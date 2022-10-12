#!/bin/bash
groups=(0 1 2 3 4 5 6 7)
dslevels=(1 2 3)
batchsize=(32 32 64)
noise_range=(1.0 1.5 2.0)
dataset64=("./datasets/ModelNet40/10bitdepth_2_oct4/" "./datasets/Microsoft/10bitdepth_2_oct4/"  "./datasets/MPEG/10bitdepth_2_oct4/" )
test_data=("./testdata/")

# TBPV
# for gr in "${groups[@]}";
# do
#   for lv in "${dslevels[@]}";
#     do
#     python3 -m training.ms_voxel_transformer_training -usecuda 1 -dataset ${dataset64[0]} -dataset ${dataset64[1]} -dataset ${dataset64[2]}  -outputmodel Model/TBPVoxel/ -epoch 20 --scratch=1  -batch ${batchsize[${lv}-1]}  -group $gr -downsample $lv  -dim 1024 -depth 3 -mlp_dim 512 -heads 2
#     done
# done

# MS_Voxel
# for gr in "${groups[@]}"
# do
#   for lv in "${dslevels[@]}";
#     do
#     python3 -m training.ms_voxel_cnn_training -usecuda 1 -dataset ${dataset64[0]} -dataset ${dataset64[1]} -dataset ${dataset64[2]}  -outputmodel Model/MSVoxelCNN/ -epoch 20 --scratch=0  -batch ${batchsize[${lv}-1]}   -nopatches 2 -group $gr -downsample $lv -noresnet 4
#     done
# done

# voxel8
# python3 -m training.voxel_dnn_training_torch -blocksize 8 -nfilters 64 -inputmodel Model/voxelDNN -outputmodel Model/voxelDNN -dataset "./datasets/ModelNet40/10bitdepth_2_oct4/" -dataset "./datasets/MPEG/10bitdepth_2_oct4/" -dataset "./datasets/Microsoft/10bitdepth_2_oct4/"  -batch 8 -epochs 3

############ test


# try1
for gr in "${groups[@]}";
do
  # for lv in "${dslevels[@]}";
  #   do
  python3 -m training.try1 -usecuda 1 -dataset "./little_datasets/MPEG/10bitdepth_2_oct4/" -dataset "./little_datasets/Microsoft/10bitdepth_2_oct4/"  -outputmodel Model/TEST1_TRY1/ -epoch 20 --scratch=1  -batch ${batchsize[${lv}-1]}  -group $gr -downsample 2  -dim 128 -depth 3 -mlp_dim 64 -heads 2
    # done
done

# Ms
# for gr in "${groups[@]}";
# do
#   for lv in "${dslevels[@]}";
#     do
#     python3 -m training.ms_voxel_cnn_training -usecuda 1 -dataset "./little_datasets/MPEG/10bitdepth_2_oct4/" -dataset "./little_datasets/Microsoft/10bitdepth_2_oct4/"  -outputmodel Model/TEST1_MS/ -epoch 20 --scratch=1  -batch ${batchsize[${lv}-1]}  -nopatches 2 -group $gr -downsample $lv -noresnet 4
#     done
# done

# TB
# for gr in "${groups[@]}";
# do
#   for lv in "${dslevels[@]}";
#     do
#     python3 -m training.ms_voxel_transformer_training -usecuda 1 -dataset "./little_datasets/MPEG/10bitdepth_2_oct4/" -dataset "./little_datasets/Microsoft/10bitdepth_2_oct4/"  -outputmodel Model/TEST_TB/ -epoch 20 --scratch=1  -batch ${batchsize[${lv}-1]}  -group $gr -downsample $lv  -dim 1024 -depth 3 -mlp_dim 512 -heads 2
#     done
# done