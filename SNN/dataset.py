import numpy as np
import math
import struct
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from collections.abc import Iterable
import torch.nn.functional as F
###############################################################################


nCublets = 1000
nSensors = 100
max_t = 20
dt = 0.2
timesteps = int(max_t/dt)


###############################################################################


def readfile(filename, primary_only, merged = True):

  ph_list = []
  E_list  = []
  ct_list = []
  sE_list = []
  N_list  = []
  p_class = []
  cublet_id = []
  event_id = []

  if not primary_only:
    primary_list = []

  delimiter = b'EOE '

  with open(filename, 'rb') as file:
    event = 1
    data = file.read(4)
    while data:
      if data == delimiter:
        event = struct.unpack('i', file.read(4))[0]
        data = file.read(4)

        if not data:
            break
        continue

      ph_matrix = np.zeros(shape=(timesteps, nSensors), dtype=np.int32)

      # read photon count data
      while True:
        t = struct.unpack('i', data)[0]
        # check stop condition
        if t == 2147483647:
          break
        sensor = struct.unpack('i', file.read(4))[0]
        n_photons = struct.unpack('i', file.read(4))[0]
        ph_matrix[t, sensor] = n_photons

        data = file.read(4)
      
      ph_list.append(ph_matrix)

      # Read cublet_id
      cublet_id.append(struct.unpack('i', file.read(4))[0])

      # Read total energy released
      E_list.append(struct.unpack('d', file.read(8))[0])
      
      # Read energy centroid
      x = struct.unpack('d', file.read(8))[0]
      y = struct.unpack('d', file.read(8))[0]
      z = struct.unpack('d', file.read(8))[0]
      ct_list.append((x,y,z))
      
      # Read energy dispersion
      sX = struct.unpack('d', file.read(8))[0]
      sY = struct.unpack('d', file.read(8))[0]
      sZ = struct.unpack('d', file.read(8))[0]
      sE_list.append((sX, sY, sZ))
    
      # Read number of interactions
      N_list.append(struct.unpack('i', file.read(4))[0])
    
      # Read particle class
      p_class.append(struct.unpack('i', file.read(4))[0]-1)

      # Read primary vertex indicator
      if not primary_only:
        primary_list.append(struct.unpack('i', file.read(4))[0])
      
      if merged: 
        event_id.append(event)
      
      data = file.read(4)

  res = [ph_list, E_list, ct_list, sE_list, N_list, p_class, cublet_id]

  if merged:
    res.append(event_id)
    
  if not primary_only:
    res.append(primary_list)

  return res


###############################################################################



# converts to Torch tensor of desired type
def to_tensor_and_dtype(input, target_dtype=torch.float32):
    
    # Convert to PyTorch tensor
    if not torch.is_tensor(input):
        input = torch.tensor(input)

    # Force the tensor to have the specified dtype
    if input.dtype is not target_dtype:
        input = input.to(target_dtype)
    
    return input

class CustomDataset(Dataset):
    def __init__(self, filelist, primary_only=True, target="energy",transform=None):
        
        targets_dict = {
            "energy":1,
            "centroid":2,
            "dispersion":3,
            "N_int":4,
            "particle":5,
            "primary":6
        }
        
        samples = []
        targets = []
        for file in filelist:
            info = readfile(file, primary_only)
            samples += info[0]

            if isinstance(target, Iterable) and not isinstance(target, (str, bytes)):
                temp = [info[targets_dict[key]] for key in target]
                targets += list(zip(*temp))
            else:
                targets += info[targets_dict[target]]

        samples = to_tensor_and_dtype(np.array(samples))
        if isinstance(targets, str):
            targets = to_tensor_and_dtype(targets, target_dtype = torch.long)
        self.data = list(zip(samples, targets))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): 

        if isinstance(idx, slice):
            return [self.__getitem__(i) for i in range(*idx.indices(len(self)))]  # Slicing directly
        
        if isinstance(idx, (list, np.ndarray)):
            if (isinstance(idx, np.ndarray) and idx.dtype == bool) or \
               (isinstance(idx, list)       and all(isinstance(item, bool) for item in idx)):  # Boolean mask
                if len(idx) != len(self.data):
                    raise ValueError("Boolean mask must have the same length as the dataset")
                return [self.__getitem__(i) for i, mask in enumerate(idx) if mask]
            return [self.__getitem__(i) for i in idx]  # Indexing with list or array
        
        if isinstance(idx, (int, np.int64, torch.int64)):
            sample = self.data[idx]
            if self.transform:
                sample = self.transform(sample)
            return sample
        
        raise TypeError("Invalid index type: {}".format(type(idx)))

    def clean(self, selection_fn):
        """
        Filters the dataset by applying a selection function to each sample.

        Args:
            selection_fn (callable): A function that takes a single sample as input
                                     (sample, target) and returns a boolean indicating
                                     whether to keep the sample.
        """
        self.data = [sample for sample in self.data if selection_fn(sample)]
    

###############################################################################


def build_dataset(path, max_files=50, *args, **kwargs):
    
    filelist = []

    for subdir, _, files in os.walk(path):
        subdir_name = os.path.basename(subdir)
        if files:
            filelist += [os.path.join(path, subdir_name, f) for f in files[:max_files]]

    dataset = CustomDataset(filelist, *args, **kwargs)

    return dataset

    
def build_loaders(dataset, split=(0.6, 0.2), batch_size=50, *args, **kwargs):
    
    if isinstance(split, Iterable) and not isinstance(split, (str, bytes)):
        if len(split) > 2:
            raise ValueError("Split should be provided for training and validation datasets")
        if sum(split) > 1:
            raise ValueError("Split fractions should sum up to 1 at most")
        train_size = int(len(dataset)*split[0])
        try:
            val_size = int(len(dataset)*split[1])
        except:
            val_size = 0
        test_size = len(dataset) - train_size - val_size
    else:
        train_size = int(len(dataset)*split)
        test_size = len(dataset) - train_size
        val_size = 0

    train_dataset, test_dataset, val_dataset = random_split(dataset, [train_size, test_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, *args, **kwargs)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, *args, **kwargs)
    if len(val_dataset) > 0:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, *args, **kwargs)
        return train_loader, test_loader, val_loader
    
    return train_loader, test_loader
    

