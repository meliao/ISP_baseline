#!/bin/bash

#SBATCH --job-name=debug_ptns_train_widebnet
#SBATCH --time=3:00:00
# SBATCH --account=oortsang
#SBATCH --partition=willett-contrib
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100GB
#SBATCH --output=logs/2024-04-30_debug_ptns_train_widebnet.out
#SBATCH --error=logs/2024-04-30_debug_ptns_train_widebnet.err
#SBATCH --mail-type=end
#SBATCH --mail-user=oortsang@uchicago.edu


WORKDIR=tmp/replicate_widebnet_experiment_ptns
# TEST_DIR=/home/meliao/projects/Inverse_Scattering_ML_TF2/public-example/testdata
TEST_DIR=/home/oortsang/widebnet-repo/public-example/testdata
rm -rf $WORKDIR
python train_widebnet_model.py \
    -train_data_dir /net/projects/willettlab/meliao/recursive-linearization/traindata_L3s10_multifreq_square_3_5_10_h_freq_2.5_5_10/ \
    -test_data_dir ${TEST_DIR} \
    -truncate_num 1000 \
    -truncate_num_val 100 \
    -num_train_steps 1000 \
    -workdir $WORKDIR \
    --blur_test_eta \
    -wavenumbers 10 \
    -wavenumber_low 0.0 \
    -wavenumber_high 10.0

# -wavenumbers 2.5 5 10 \