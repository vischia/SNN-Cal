#!/bin/bash

source /lhome/ext/uovi123/`whoami`/.bashrc
conda activate snn_hgcal
# Adapt to the path where you installed it:
cd /lhome/ext/uovi123/`whoami`/snn_calo/SNN-Cal/
mkdir -p logs
python $1