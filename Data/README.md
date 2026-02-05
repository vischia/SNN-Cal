# Dataset 

The datasets are created through a multi-step process:
1. Initial GEANT4 simulation (see [this page](https://github.com/Tungcg1906/GEANT4-simulation) for more information). 
2. <a name="pre-processing"></a> Pre-processing of the simulation output to prune unused branches of the *TTree* and re-format it (see [this page](https://github.com/Tungcg1906/Particle-Detectors-optimization-with-Deep-Learning-techniques/blob/main/src/README.md) for more information).
3. Process the lightweight files using the [*genPhotons.cpp*](https://github.com/enlupi/SNN-Cal/blob/main/GenerateDataset/genPhotons.cpp) script to produce the actual dataset used in this study.

The datasets are stored in a **CERNBox** folder, available at the following link: https://cernbox.cern.ch/s/3XHkVMwn8iATKuU. You may use the *download_data.py* script to download and set them up locally. 

## Dataset Organization

The folder is organized based on the following schema:

```
Data/
├── AllCublets/
└── PrimaryCublets/
    ├── Uniform/
    ├── Side/
    └── Centred/
```

(the same schema will be used by default when downloading the data). Each of the final subdirectories contains three folders (*kaon/*, *pion/* and *proton/*), whose name corresponds to the primary particle that was used in the *GEANT4* simulation to generate the data stored within them.

### AllCublets

The **AllCublets** folder contains the dataset where every interaction is recorded, regardess of the cublet where it took place. \
 The name convention used for the files is *\<primary particle>\_\<file index>\_\<event index>.dat*, where *file index* refers to the index of the pre-processed file produced in [step 2](#pre-processing) of the dataset creation process. 

>   ⚠️ **Warning** ⚠️ \
> The files in this directory have a different format than the ones in **PrimaryCublets**! Please do not forget to set the [*primary_only*](https://github.com/enlupi/SNN-Cal/blob/main/SNN/dataset.py) argument correctly when generating the dataset.

### PrimaryCublets

The **PrimaryCublets** folder contains only the interactions that took place in the cublet containing the primary vertex of the event, which typically contains a large share of the released energy and offers the highest information content on the identity of the incident particle. \
The name convention used for the files is *\<primary particle>\_\<file index>.dat*, and each file contains the information of all the 1,000  events contained in the original pre-processed file.

This folders contains three different subfolders, depending on the position of the particle gun with respect to the detector (or its distribution):
- **Centered**: the initial position of the particle gun is centered on one cublet \
  (⚠️this data is not used, and is present only for testing purposes⚠️) 
- **Fixed**: the initial position of the particle gun is centered on the whole face of the detector, which corresponds to the intersection of four cubelets. 
- **Uniform**: the initial position of the particle gun is randomized, so that it is always contained inside the face of a particular cublets but uniformely distributed inside it.
