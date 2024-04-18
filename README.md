# Baseline Models for Solving the Inverse Scattering Problem

## Overview
This repository currently contains four baseline deterministic models for solving the wideband inverse scattering problem. The models included are:

- **SwitchNet**
- **Uncompressed Equivariant Model**
- **Compressed Equivariant Model**
- **Wideband Butterfly Network**

## Installation
Project Environment can be installed by running
!pip install git+https://github.com/google-research/swirl-dynamics.git@main

## Demos
Demos for these models can be found in the `colabs` folder.

## Comments on the uncompressed and compressed rotationally equivariant models
-The two models are sensitive to the order of the source dimension (s) and the receiver dimension (r) in the far-field pattern data. If the models yield very poor results, try training with the perturbation data transposed.

-Using the warmup_cosine_decay_schedule scheduler to train the two models yields much better results (compared to the exponential_decay scheduler used in the TensorFlow codes).


## Owen's Setup Notes

```
#Needs to be python 3.11 even though the docs say >=3.10
conda create -n jax_inv_scat python=3.11 
conda activate jax_inv_scat
pip install git+https://github.com/google-research/swirl-dynamics.git@main
# CUDA version of jax
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
conda install jupyter matplotlib natsort 
conda install chardet
pip install pysteps
```

## Owen's Data Generation Notes

### Training Dataset

Here are the details for my reconstruction of the "Square (3, 5, 10)" dataset from the Wide-Band Butterfly Network paper. 
I was working in this private GitHub repository, which is a fork of Matt Li's repository. 
```
https://github.com/meliao/Inverse_Scattering_ML_TF2
```

In MATLAB, I ran the following commands:
```
scatter_width_per_mesh_list = [3 5 10];
freq_list = [2.5 5 10];
dset = 'train';
gendata_multifreq_mixh(3,10, scatter_width_per_mesh_list, 'square', freq_list, dset)
```
Note that I used L=3, s=10 instead of the usual L=4, s=5. I don't believe this had a material effect, because these parameters are only used to determine the grid size = (2**L) * s.

After the MATLAB code completed (~ 3 hours) I ran the post-processing scripts:
```
# Go to the appropriate data dir
cd /Users/owen/projects/Inverse_Scattering_ML_TF2/data/xxxdata_L3s10_multifreq_square_3_5_10_h_freq_2.5_5_10
# post-process script to collect data
sh ../../post-processing-scripts/collect_data_2.5_5_10.sh
# post-process script to put data into HDF5
python ../../post-processing-scripts/merge_wavedatacsv_to_hdf5.py
```
These post-processing scripts were edited to reflect the different hostnames in Matt's compute environment and my own.

### Test Dataset

I was using the 3,000 samples of the "Square (3, 5, 10)" dataset provided by Matt in the repository:
```
https://github.com/mtcli/Inverse_Scattering_ML_TF2/tree/master/public-example/testdata
```