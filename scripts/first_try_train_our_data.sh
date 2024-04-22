WORKDIR=tmp/first_try_train_widebnet_model_our_data
rm -rf $WORKDIR

DIR_OURDATA_TEST=/net/projects/willettlab/meliao/recursive-linearization/dataset/test_measurements_nu_{}
DIR_OURDATA_TRAIN=/net/projects/willettlab/meliao/recursive-linearization/dataset/train_measurements_nu_{}

python train_widebnet_model.py \
    -train_data_dir $DIR_OURDATA_TRAIN \
    -test_data_dir $DIR_OURDATA_TEST \
    -workdir $WORKDIR \
    -truncate_num 3333 \
    -wavenumbers 4 8 16 \
    -s 12 \
    --use_wandb