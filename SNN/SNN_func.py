import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from typing import Callable, Union
from collections.abc import Iterable
from itertools import accumulate
from tqdm import tqdm
from torchmetrics.classification import MulticlassConfusionMatrix
from matplotlib.colors import SymLogNorm

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#if torch.backends.mps.is_available():
#    device = torch.device("mps")
#    torch.set_default_dtype(torch.float32)


#####################################
##                                 ##
##  RANDOM SEED                    ##
##                                 ##
#####################################

# def set_global_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

###############################################################################
##                                                                           ##
##     SPIKING NEURAL NETWORK
##                                                                           ##
###############################################################################

class Spiking_Net(nn.Module):
    """FCN with variable neural model and number of layers."""

    def __init__(self, net_desc, spikegen_fn):
        super().__init__()
        
        self.n_neurons = net_desc["layers"]
        self.timesteps = net_desc["timesteps"]
        self.output = net_desc["output"]

        modules = []
        for i_layer in range(1, len(self.n_neurons)):
            modules.append(nn.Linear(in_features=self.n_neurons[i_layer-1], out_features=self.n_neurons[i_layer]))
            if "model" in net_desc:
                modules.append(net_desc["model"](**net_desc["neuron_params"][i_layer]))
            else:
                modules.append(net_desc["neuron_params"][i_layer][0](**(net_desc["neuron_params"][i_layer][1])))
        self.network = nn.Sequential(*modules)

        self.spikegen_fn = spikegen_fn

    
    def forward(self, data):
        """Forward pass for several time steps."""

        x = self.spikegen_fn(data)

        # Initalize membrane potential
        mem = []
        for i, module in enumerate(self.network):
            if i%2==1:
                res = module.reset_mem()
                if type(res) is tuple:
                    mem.append(list(res))
                else:
                    mem.append([res])

        # Record the final layer
        spk_rec = []
        mem_rec = []

        # Loop over 
        spk = None
        for step in range(self.timesteps):
            for i_layer in range(len(self.network)//2):
                if i_layer == 0:
                    cur = self.network[2*i_layer](x[step])
                else:
                    cur = self.network[2*i_layer](spk)
                
                spk, *(mem[i_layer]) = self.network[2*i_layer+1](cur, *(mem[i_layer]))

                if i_layer == len(self.network)//2-1:
                    spk_rec.append(spk)
                    mem_rec.append(mem[i_layer][-1])

        if self.output == "spike":
            return torch.stack(spk_rec, dim=0)
        elif self.output == "membrane":
            return torch.stack(mem_rec, dim=0)
        


###############################################################################
##                                                                           ##
##     PREDICTOR
##                                                                           ##
###############################################################################

class Predictor():

    def __init__(self, prediction_fn, accuracy_fn, population_sizes: Union[int, Iterable[int]] = -1):
        self.prediction_fn = prediction_fn
        self.accuracy_fn = accuracy_fn
        if isinstance(population_sizes, int):
            self.population_sizes = population_sizes
            return
        if isinstance(population_sizes, Iterable):
            if isinstance(population_sizes, (np.ndarray, torch.Tensor)):
                population_sizes = population_sizes.tolist()
            if all(isinstance(x, int) for x in population_sizes):
                self.population_sizes = population_sizes
                return
        raise TypeError("Input must be an int or an iterable of int.")


    def _predict_singletask(self, output, targets, reduction: str = "mean"):
        prediction = self.prediction_fn(output)
        accuracy = self.accuracy_fn(prediction, targets)
        if reduction == "mean":
            return prediction, torch.mean(accuracy, 0)
        elif reduction == "sum":
            return prediction, torch.sum(accuracy, 0)
        else:
            return prediction, accuracy
        
    
    def __call__(self, output, targets, reduction: str = "mean"):
        #check if multiple task must be handled
        if len(targets.shape) == 1:
            return self._predict_singletask(output, targets, reduction)
        
        # populations of different size
        if isinstance(self.population_sizes, (list, tuple)):
            if sum(self.population_sizes) != output.shape[-1]:
                raise ValueError("Population sizes must add up to last layer size!")
            if len(self.population_sizes) != targets.shape[-1]:
                raise ValueError("Number of populations must be equal to number of tasks!")
            prediction = torch.zeros(size=targets.shape)
            if reduction == "none":
                accuracy = torch.zeros(size=targets.shape)
            else:
                accuracy = torch.zeros(size=(targets.shape[1],))
            chunks = list(accumulate([0]+list(self.population_sizes)))
            for i, _ in enumerate(self.population_sizes):
                if reduction == "none":
                    prediction[:, i], accuracy[:, i] = \
                        self._predict_singletask(output[chunks[i]:chunks[i+1]], targets[:, i], reduction)
                else:
                    prediction[:, i], accuracy[i] = \
                        self._predict_singletask(output[chunks[i]:chunks[i+1]], targets[:, i], reduction)
            return  prediction, accuracy          

        # populations of the same size
        if isinstance(self.population_sizes, int):
            output = output.reshape(output.shape[0], output.shape[1], self.population_sizes, -1)
            if output.shape[-1] != targets.shape[-1]:
                raise ValueError("Number of populations must be equal to number of tasks!")
        return self._predict_singletask(output, targets, reduction)



###############################################################################
##                                                                           ##
##     MULTIPLE TASK LOSS FUNCTION
##                                                                           ##
###############################################################################
    
class multi_MSELoss(torch.nn.Module):

    def __init__(self, reduction: str = "mean", weights : torch.tensor = torch.ones(1),
                 set_mse : list = [0,1,1,0]) -> None:
        super(multi_MSELoss, self).__init__()
        self.reduction = reduction
        self.weights = weights
        self.func = []
        for i in range(len(set_mse)):
            if set_mse[i]:
                self.func.append(F.mse_loss)
            else:
                self.func.append(F.l1_loss)


    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        
        if len(target.shape) < 2:
            target = target.unsqueeze(-1)

        losses = torch.zeros(target.shape[-1])
        for i in range(target.shape[-1]):
            losses[i] = self.func[i](input[:, i], target[:, i], reduction=self.reduction)
        
        return (self.weights*losses).sum()



###############################################################################
##                                                                           ##
##     TRAINER AND TESTER CLASS
##     # use_beta_clamp = False: Beta added here                             ##
###############################################################################
    
class Trainer():

    def __init__(self, net, loss_fn, optimizer, predict,
                 train_dataset, val_dataset, test_dataset, task="Regression"):
        self.net = net
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.predict = predict    
        self.datasets = {"train": train_dataset, "validation": val_dataset, "test": test_dataset}
        self.task = task
        # self.use_beta_clamp = use_beta_clamp

        self.current_epoch = 0
        self.loss_hist = {"train": {}, "validation": {}, "test": {}}
        self.acc_hist = {"validation": {}, "test": {}}
        self.par_hist = {}
        for name, param in self.net.named_parameters():
            if param.requires_grad:
                self.par_hist[name] = []
                
        # Beta History is added here
        #self.beta_hist = []
                     

    def test(self, dataset_name):

        # by default, computes accuracy on test dataset
        try:
            if dataset_name == "validation" or dataset_name == "test":
                dataset = self.datasets[dataset_name]
            else:
                raise NameError("Unidentified dataset name. Please choose between \"validation\" or \"test\".")
        except NameError as n:
            print(f"Error: {n}")

        temp_loss = []
        temp_acc = []

        self.net.eval()
        with torch.no_grad():
            temp_loss = []
            for data, targets in dataset:
                data = data.to(device)
                targets = targets.to(device)

                # forward pass
                output = self.net(data)
                pred, acc = self.predict(output, targets)

                # compute loss
                loss = self.loss_fn(pred, targets)
                temp_loss.append(loss.item())
                temp_acc.append(acc)

        self.loss_hist[dataset_name][self.current_epoch] = np.mean(temp_loss, 0)
        self.acc_hist[dataset_name][self.current_epoch] = np.mean(temp_acc, 0)


    def train(self, num_epochs, verbosity=1):

        self.net.to(device)

        # Validation
        self.test("validation")
        if verbosity:
            task_metric = "Average Error" if self.task == "Regression" else "Accuracy"
            print(f"Epoch {self.current_epoch}:")
            print(f"Validation Loss = {self.loss_hist['validation'][self.current_epoch]}")
            print(f"Validation {task_metric} = {self.acc_hist['validation'][self.current_epoch]}")
            print("\n-------------------------------\n")

        # Save value of optimizable parameters
        for name, param in self.net.named_parameters():
            if param.requires_grad:
                seld.par_hist[name].append(param.cpu().detach().numpy())

        # initial Beta
        # for name, param in self.net.named_parameters():
        #      if "beta" in name and para,.requires_grad:
        #          self.beta_hist.append(param.detach().cpu().clone.numpy())

        for epoch in tqdm(range(num_epochs), desc="Epoch"):
            self.net.train()
            # Minibatch training loop
            for data, targets in tqdm(self.datasets["train"], desc="Batches", leave=False):
                data = data.to(device)
                targets = targets.to(device)

                # forward pass
                output = self.net(data)
                pred, _ = self.predict(output, targets)

                # compute loss
                loss_val = self.loss_fn(pred, targets)

                # Gradient calculation + weight update
                self.optimizer.zero_grad()
                loss_val.backward()
                self.optimizer.step()

            # End epoch: Log Beta
            # for name, param in self.net.named_parameters():
            #     if "beta" in name.lower() and param.requires_grad:
            #         self.beta_hist.append(param.detach().cpu().clone().numpy())
                
                  # Here the Beta clamping Process starts
                  # with torch.no_grad():
                #       for name, param in self.net.named_parameters():
                #           if 'beta' in name.lower():
                #               param.clamp_(0.0,1.0)

                  # Updating the Beta Process for testing here
                 #  if self.use_beta_clamp:
                #       with torch.no_grad():
                #           for name, param in self.net.named_parameters():
                #               if 'beta' in name.lower():
                #                   param.clamp_(0.0,1.0)

                # Here the Beta clamping Process starts (This Beta clamping is used only if the beta values go beyond 1 and if the values do not go beyond 1 then no need,
                # as the snntorch library has inbuilt beta clamping depending upon the version of the library used. Better to check the library version 1st before doing
                # external beta clamping
                
                 # beta_tol = 1.10
                
                  # with torch.no_grad():
                  #    for name, param in self.net.named_parameters():
                  #        if 'beta' in name.lower():
                  #            param.clamp_(0.0,1.0)


                # Store loss history for future plotting
                if self.current_epoch in self.loss_hist["train"]:
                    self.loss_hist["train"][self.current_epoch].append(loss_val.item())
                else:
                    self.loss_hist["train"][self.current_epoch] = [loss_val.item()]

            self.current_epoch += 1

            # Validation
            self.test("validation")

            if verbosity:
                print(f"Epoch {self.current_epoch}:")
                print(f"Validation Loss = {self.loss_hist['validation'][self.current_epoch]}")
                print(f"Validation {task_metric} = {self.acc_hist['validation'][self.current_epoch]}")
                print("\n-------------------------------\n")

            # After training - compute stats
            self.plot_parameter.statistics()

        # check if the Beta Exists
        if len(self.beta_hist) == 0:
            raise RuntimeError("No beta parameter found in model.")

        # Correct Beta Snapshots
        if len(self.beta_hist) != num_epochs +1:
            raise RuntimeError(
                f'Beta history lenght mismatch'
                f'expected {num_epochs + 1}, got {len(self.beta_hist)}'
            )

        return{
            "beta": self.beta_hist,
            "loss": self.loss_hist,
            "acc": self.acc_hist,
        }
# This function is added now for testing purpose to see whether our approach is correct or not
    def plot_parameter_statistics(self):
        """Compute and plot min, mean, and variance of all learnable parameters across epochs"""
        print("\n Computing parameter statistics...\n")

        stats = {}
        for name, param_snapshots in self.par_hist.items():
            param_snapshots = np.array(param_snapshots)
            stats[name] = {
                'min': np.min(param_snapshots, axis=0),
                'mean': np.mean(param_snapshots, axis=0),
                'var': np.var(param_snapshots, axis=0)
            }

        # Aggregate Stats for Plot

        # param_names = list(stats.keys())
        # min_vals = [stats[p]['min'] for p in param_names]
        # mean_vals = [stats[p]['mean'] for p in param_names]
        # var_vals = [stats[p]['var'] for p in param_names]

        # Create subplots for min, mean, variance

        #fig, axes = plt.subplots(1,3, figsize=9)

        # Plot mean & variance trends (aggregated)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        mean_vals = [np.mean(v['mean']) for v in stats.values()]
        var_vals = [np.mean(v['var']) for v in stats.values()]
        param_names = list(stats.keys())

        axes[0].bar(range(len(param_names)), mean_vals)
        axes[0].set_xticks(range(len(param_names)))
        axes[0].set_xticklabels(param_names, rotation=45, ha='right')
        axes[0].set_title("Average of Mean Parameter Values")
        axes[0].set_ylabel("Mean Value")    

        axes[1].bar(range(len(param_names)), var_vals)
        axes[1].set_xticks(range(len(param_names)))
        axes[1].set_xticklabels(param_names, rotation=45, ha='right')
        axes[1].set_title("Average of Parameter Variance")
        axes[1].set_ylabel("Variance")

        plt.tight_layout()
        plt.show()
    
    def plot_loss(self, validation=True, logscale=True):

        loss = [l for l_per_epoch in self.loss_hist["train"].values() for l in l_per_epoch]
        fig = plt.figure(facecolor="w", figsize=(4, 3))
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        if logscale:
            plt.yscale("log")
        plt.plot(loss, label="Training")
        if validation:
            x = [i*len(self.datasets["train"]) for i in self.loss_hist["validation"]]
            plt.plot(x, list(self.loss_hist["validation"].values()), color='orange', marker='o', linestyle='dashed', label="Validation")
        
        plt.legend(loc='upper right')
        plt.show()

    
    def ConfusionMatrix(self, *args, **kwargs):

        cm = MulticlassConfusionMatrix(*args, **kwargs)

        self.net.eval()
        with torch.no_grad():
            for data, targets in self.datasets["test"]:
                data = data.to(device)
                targets = targets.to(device)

                # forward pass
                output = self.net(data)

                # calculate total accuracy
                pred, _ = self.predict(output, targets)
                cm.update(pred, targets)
        
        return cm


    def _get_all(self, transform: Callable = lambda *args, **kwargs: args[0]):
        self.net.eval()
        all_targets = []
        all_predictions = []
        all_accuracy = []
        with torch.no_grad():
            for data, targets in self.datasets["test"]:
                data = data.to(device)
                targets = targets.to(device)

                # forward pass
                output = self.net(data)

                # calculate total accuracy
                pred, acc = self.predict(output, targets, reduction='none')

                all_targets.append(transform(targets))
                all_predictions.append(transform(pred))
                all_accuracy.append(acc)

        all_targets = torch.cat(all_targets, dim=0)
        all_predictions = torch.cat(all_predictions, dim=0)
        all_accuracy = torch.cat(all_accuracy, dim=0)

        if len(all_targets.shape) < 2:
            all_targets = all_targets.unsqueeze(-1)
            all_predictions = all_predictions.unsqueeze(-1)
            all_accuracy = all_accuracy.unsqueeze(-1)

        return all_targets, all_predictions, all_accuracy
    

    def _plot_results(self, targets = torch.tensor([]), predictions = torch.tensor([]),
                      accuracy = torch.tensor([]), plot_type : str = "2D",
                      nbins : int = 50, title : Union[str, list[str]] = "",
                      logscale : bool = False, select : list = [],
                      *args, **kwargs):
        
        n_tasks = max(targets.shape[-1], accuracy.shape[-1])
        if select:
            n_tasks = min(n_tasks, len(select))
        else:
            select = [i for i in range(n_tasks)]
            
        ncols = math.ceil(math.sqrt(n_tasks))
        nrows = math.ceil(n_tasks / ncols)
        
        fig, axs = plt.subplots(ncols=ncols, nrows=nrows, facecolor="w", figsize=(5*ncols, 4*nrows),constrained_layout=True)
        if not isinstance(axs, np.ndarray):
            axs = np.array([axs])
            title = [title]
        axs = axs.flatten()

        for i in range(n_tasks):
            # plot 2D histogram of prediction vs targets
            if plot_type == "2D":
                axs[i].set_xlabel("Targets", fontsize=15)
                axs[i].set_ylabel("Prediction", fontsize=15)
                axs[i].set_title(title[i], fontsize=15)
                if "E" in title[i] or "sigma" in title[i]:
                    r = [min(targets[:, select[i]].min(), predictions[:, select[i]].min()),
                         max(targets[:, select[i]].max(), predictions[:, select[i]].max())]
                else:
                    r = [0, 9]
                    axs[i].set_xticks([i for i in range(10)])
                    axs[i].set_yticks([i for i in range(10)])
                if logscale:
                    hist = axs[i].hist2d(targets[:, select[i]], predictions[:, select[i]],
                                         nbins, norm=SymLogNorm(*args, **kwargs), cmap='viridis',
                                         range=[r, r])
                else:
                    hist = axs[i].hist2d(targets[:, select[i]], predictions[:, select[i]], nbins,
                                         range=[r, r])
                                   
                axs[i].plot([0, 1e5], [0, 1e5], color='white', linewidth=1, linestyle='--')

            # plot 1D histogram of residuals
            if plot_type == "1D":
                axs[i].set_xlabel("Residuals", fontsize=15)
                axs[i].set_ylabel("Counts", fontsize=15)
                axs[i].set_title(title[i], fontsize=15)
                hist = axs[i].hist(accuracy[:, select[i]], nbins, edgecolor='black', alpha=0.7)
                axs[i].grid(True, linestyle='--', alpha=0.6)
                axs[i].axvline(0, color='black', linewidth=1, linestyle='--', alpha=0.5)
                axs[i].spines['top'].set_visible(False)
                axs[i].spines['right'].set_visible(False)
                axs[i].tick_params(axis='both', which='major', labelsize=12)

        for i in range(n_tasks, len(axs)):
            fig.delaxes(axs[i])

        if plot_type == "2D":
            # Add a color bar
            if hist:
                cbar = fig.colorbar(hist[3], ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
                cbar.set_label("Counts", fontsize=15)  # Customize as needed


    def plot_pred_vs_target(self, transform: Callable = lambda *args, **kwargs: args[0], *args, **kwargs):
        targets, predictions, _ = self._get_all(transform=transform)
        self._plot_results(targets=targets, predictions=predictions, plot_type="2D", *args, **kwargs) 


    def plot_residuals(self, transform: Callable = lambda *args, **kwargs: args[0], *args, **kwargs):
        _, _ , accuracy = self._get_all(transform=transform)
        self._plot_results(accuracy=accuracy, plot_type="1D", *args, **kwargs)


    def show_results(self, transform: Callable = lambda *args, **kwargs: args[0], *args, **kwargs):
        print(f"Test loss: {self.loss_hist['test'][self.current_epoch]}")
        print(f"Test relative error: {self.acc_hist['test'][self.current_epoch]*100}%")
        self.plot_loss()
        targets, predictions, accuracy = self._get_all(transform=transform)
        self._plot_results(targets=targets, predictions=predictions, plot_type="2D", *args, **kwargs) 
        self._plot_results(accuracy=accuracy, plot_type="1D", *args, **kwargs)

    def get_par_hist(self):
        return self.par_hist
