WORKDIR=tmp/replicate_widebnet_experiment
rm -rf $WORKDIR
python train_widebnet_model.py \
    -train_data_dir /net/projects/willettlab/meliao/recursive-linearization/traindata_L3s10_multifreq_square_3_5_10_h_freq_2.5_5_10/ \
    -test_data_dir /home/meliao/projects/Inverse_Scattering_ML_TF2/public-example/testdata/ \
    -workdir $WORKDIR \
    --blur_test_eta