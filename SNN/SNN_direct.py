import snntorch as snn
from snntorch import surrogate
from copy import deepcopy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.show = lambda: None
import numpy as np
import torch
from typing import Callable
import visualize
import dataset as ds
import SNN_func_modified as snnfn
from torch.utils.data import DataLoader
import torch.nn as nn
from snntorch import spikeplot as splt
import torchvision as tv


print('empieza')
nCublets = 1000
nSensors = 100
max_t = 20
dt = 0.2
timesteps = int(max_t/dt)
batch_size = 50
num_epochs = 4
idx = 100  
'''
labels_map = {
  -1: "unclassified",
   0: "proton",
   1: "kaon",
   2: "pion",
   3: "other"
}
'''
labels_map = {
   0: "proton",
   1: "kaon",
   2: "pion"}
nClasses = len(labels_map)


population = 20

net_desc = {
    "layers" : [400, 120 ,100 , nClasses*population],
    "timesteps": 100,
    "neuron_params" : {

                1: [snn.Leaky, 
                    {"beta" : 0.5,
                    "learn_beta": True,
                    "threshold" : 1.0,
                    "learn_threshold": True,
                    "spike_grad": surrogate.atan(),
                    }],
                    
                2: [snn.Leaky, 
                    {"beta" : 0.5,
                    "learn_beta": True,
                    "threshold" : 1.0,
                    "learn_threshold": True,
                    "spike_grad": surrogate.atan(),
                    }],

                3: [snn.Leaky, 
                    {"beta" : 0.5,
                    "learn_beta": True,
                    "threshold" : 1.0,
                    "learn_threshold": True,
                    "spike_grad": surrogate.atan(),
                    }],

                4: [snn.Leaky, 
                    {"beta" : 0.5,
                    "learn_beta": True,
                    "threshold" : 1.0,
                    "learn_threshold": True,
                    "spike_grad": surrogate.atan(),
                    }],

                },
    }



net_desc_spikefreq = deepcopy(net_desc)
net_desc_spikefreq["output"] = "spike"



def spikegen_multi(data, multiplicity=4):
    data = torch.where(data > 0, data, torch.zeros_like(data))
    og_shape = data.shape
    spike_data = torch.zeros(og_shape[1], og_shape[0], multiplicity*og_shape[2], device = data.device)
    for i in range(multiplicity):
        condition = data > np.power(10, i+2)
        batch_idx, time_idx, sensor_idx = torch.nonzero(condition, as_tuple=True)
        spike_data[time_idx, batch_idx, multiplicity*sensor_idx+i] = 1

    return spike_data
'''
def predict_spikefreq(output):
    spikes = output.sum(dim=0)
    spikes = spikes.view(spikes.shape[0], nClasses, population)
    prediction = spikes.mean(dim=2)
    #return output.sum(dim=0)
    return prediction
'''
'''
def predict_spikefreq(output):
    prediction = output.sum(dim=0)
    return prediction
'''
def predict_spikefreq(output):
    return output
'''
def spikegen_filtrado(x):
    spikes = spikegen_multi(x, 4)
    mask = (spikes != 0).view(spikes.size(0), -1).any(dim=1)
    
    return spikes[mask]
'''

def comp_accuracy(output, targets, *args, **kwargs):
    #_, predicted = output.max(1) 
    predicted = output.argmax(dim=1)
    correct = (predicted == targets).to(torch.float32)

    return correct


class Hybrid_Net(nn.Module):
    def __init__(self, snn_network):
        super().__init__()

        self.softmax = nn.Softmax(dim=1)
        self.snn = snn_network

        self.ann = nn.Sequential(
            nn.Linear(self.snn.n_neurons[-1], 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.LeakyReLU(),
            nn.Linear(16, 4)
        )

        self.softmax = nn.LogSoftmax(dim=1)
        self.snn = snn_network

    def forward(self, data):

        features_t = self.snn(data)

        features = features_t.sum(dim=0)

        logits = self.ann(features)

        probs = self.softmax(logits)

        return probs

dataset = ds.build_dataset( path="./Data/PrimaryOnly/Uniform", max_files=10000, primary_only = True, target="particle")

train_loader, test_loader, val_loader = ds.build_loaders(dataset, split=(0.7, 0.15), batch_size=batch_size, shuffle=True)

net_Epos_spk = snnfn.Spiking_Net(net_desc_spikefreq, spikegen_multi)
modelo_completo = Hybrid_Net(net_Epos_spk)

Pred_Epos_spk = snnfn.Predictor(predict_spikefreq, comp_accuracy)
#loss_Epos = nn.CrossEntropyLoss()
loss_Epos = nn.NLLLoss()
opt_Epos_spk = torch.optim.Adam(modelo_completo.parameters(), lr=5e-3, betas=(0.9, 0.999), weight_decay=0)
sche_Epos_spk = torch.optim.lr_scheduler.ExponentialLR(opt_Epos_spk, gamma=0.9)
train_Epos_spk = snnfn.Trainer(modelo_completo, loss_Epos, opt_Epos_spk, Pred_Epos_spk,
                    train_loader, val_loader, test_loader, task = "Accuracy")

train_Epos_spk.train(num_epochs)
train_Epos_spk.predict.accuracy_fn = lambda p, t: comp_accuracy(p, t)
train_Epos_spk.test("test")
train_Epos_spk.predict.accuracy_fn = lambda p, t: comp_accuracy(p, t)
cm_metric = train_Epos_spk.ConfusionMatrix(num_classes=4)
cm = cm_metric.compute().cpu().numpy()
np.savetxt("confusion_matrix.txt", cm, fmt="%d")


'''
data_batch, targets_batch = next(iter(val_loader))
spiked_data = spikegen_multi(data_batch)

with torch.no_grad():
    output_batch = net_Epos_spk(data_batch.to(snnfn.device))
    prediction_batch = predict_spikefreq(output_batch)

T = output_batch.shape[0]
target_cls = int(targets_batch[idx].item())
print(target_cls)
pred_cls = int(prediction_batch.argmax(dim=1)[idx].item())
print(pred_cls)

spikes_evt = output_batch[:, idx, :].detach().cpu().view(T, nClasses, population)

# PLOT 1: Input Spikes
fig_in, ax_in = plt.subplots(figsize=(20, 10))
splt.raster(spiked_data[:, idx, :], ax_in, c="black", marker="|")
ax_in.set_ylabel("Input spikes") 
fig_in.savefig("input_spikes.png", dpi=300, bbox_inches="tight")
plt.close(fig_in) 

# PLOT 2: Output Spikes
fig_all, axes_all = plt.subplots(nClasses, 1, figsize=(12, 10), sharex=True)

for c in range(nClasses):
    splt.raster(spikes_evt[:, c, :], axes_all[c], c="black", marker="|")
    
    tag = " (TARGET)" if c == target_cls else ""
    tag += " (GANADORA)" if c == pred_cls else ""
    
    axes_all[c].set_ylabel(f"Clase {c}")
    axes_all[c].set_title(f"{labels_map[c]}{tag}")

axes_all[-1].set_xlabel("Tiempo")
fig_all.suptitle(f"Output spikes | target={labels_map[target_cls]} | pred={labels_map[pred_cls]}", y=0.995)
fig_all.tight_layout(rect=[0, 0, 1, 0.97])
fig_all.savefig("output_spikes_raster_4_clases.png", dpi=300, bbox_inches="tight")
plt.close(fig_all)

# PLOT 3: Tasas Totales por Clase

rates_total = spikes_evt.sum(dim=(0, 2)).numpy()
print("Tasas totales por clase para el evento", idx, ":", rates_total)

fig_rates, ax_rates = plt.subplots(figsize=(6, 4))
ax_rates.bar(range(nClasses), rates_total, tick_label=[labels_map[i] for i in range(nClasses)])
ax_rates.set_ylabel("Spikes totales (tiempo x población)")
ax_rates.set_title("Actividad total por clase")
fig_rates.tight_layout()
fig_rates.savefig("rates_totales_por_clase_evento_idx.png", dpi=300, bbox_inches="tight")
plt.close(fig_rates)

# PLOT 4: Clase Ganadora por Timestep

class_winner = spikes_evt.mean(dim=2).argmax(dim=1).numpy()

fig_cls, ax_cls = plt.subplots()
ax_cls.scatter(range(T), class_winner, s=15)
ax_cls.hlines(target_cls, -0.5, T-0.5, colors="red", linestyles="--", label="Clase real")
ax_cls.legend(markerscale=2)
ax_cls.set_yticks(range(nClasses))
ax_cls.set_yticklabels([labels_map[i] for i in range(nClasses)])
ax_cls.set_xlabel("Tiempo")
ax_cls.set_ylabel("Clase ganadora")
ax_cls.set_title("Clase ganadora por timestep")
fig_cls.tight_layout()
fig_cls.savefig("output_spikes_target_vs_pred.png", dpi=300, bbox_inches="tight")
plt.close(fig_cls)
'''

# Plot de la matriz de confusión
fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
im = ax_cm.imshow(cm, interpolation="nearest", cmap="Blues")
fig_cm.colorbar(im, ax=ax_cm)

ax_cm.set_xlabel("Predicción")
ax_cm.set_ylabel("Valor real")
ax_cm.set_title("Matriz de confusión")

tick_marks = np.arange(3)
ax_cm.set_xticks(tick_marks)
ax_cm.set_yticks(tick_marks)
ax_cm.set_xticklabels(["proton", "kaon", "pion"], rotation=45, ha="right")
ax_cm.set_yticklabels(["proton", "kaon", "pion"])

# Anotar valores en cada celda
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax_cm.text(
            j,
            i,
            int(cm[i, j]),
            ha="center",
            va="center",
            color="white" if cm[i, j] > cm.max() / 2.0 else "black",
        )

fig_cm.tight_layout()
fig_cm.savefig("confusion_matrix_SNN_40_epocas.png", dpi=300, bbox_inches="tight")



modelo_completo.eval()

all_probs = []
all_targets = []

with torch.no_grad():
    for data_batch, targets_batch in test_loader:
        data_batch = data_batch.to(snnfn.device)
        targets_batch = targets_batch.to(snnfn.device)

        log_probs = modelo_completo(data_batch)

        probs = torch.exp(log_probs)

        all_probs.append(probs.cpu())
        all_targets.append(targets_batch.view(-1).cpu())

all_probs = torch.cat(all_probs, dim=0).numpy()      
all_targets = torch.cat(all_targets, dim=0).numpy()  


class_names = ["proton", "kaon", "pion"]


fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
axes = axes.flatten()

for true_cls in range(3):
    ax = axes[true_cls]
    mask = (all_targets == true_cls)

    print(mask.sum())
    for pred_cls in range(3):
        ax.hist(
            all_probs[mask, pred_cls],
            bins=1000,
            range=(0.0, 1.0),
            density=True,
            histtype="step",
            linewidth=2,
            label=f"P({class_names[pred_cls]})"
        )

    ax.set_title(f"Eventos reales = {class_names[true_cls]}")
    ax.set_xlabel("Probabilidad predicha")
    ax.set_ylabel("Densidad")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

fig.suptitle("Distribución de probabilidades predichas condicionada a la clase real", y=0.98)
fig.tight_layout()
fig.savefig("probabilidades.png", dpi=300, bbox_inches="tight")
print("terminado")
plt.close(fig)




