#!/usr/bin/env bash
now=$(date +"%Y%m%d_%H%M%S")
EXP_DIR=./ADE20K_Resnet-101
mkdir -p ${EXP_DIR}
# Example on Camvid,  fine tune from Cityscapes

python -m torch.distributed.launch --nproc_per_node=8 train.py \
  --dataset ade20k \
  --cv 2 \
  --arch network.semantic_segmentation.Transformer_dual_branch \
  --class_uniform_pct 0.5 \
  --class_uniform_tile 512 \
  --lr 0.002 \
  --lr_schedule poly \
  --poly_exp 1.0 \
  --syncbn \
  --sgd \
  --crop_size 512 \
  --scale_min 1.0 \
  --scale_max 2.0 \
  --color_aug 0.25 \
  --max_epoch 180 \
  --jointwtborder \
  --joint_edgeseg_loss \
  --wt_bound 1.0 \
  --bs_mult 1 \
  --apex \
  --exp bdd \
  --ckpt ${EXP_DIR}/ \
  --tb_path ${EXP_DIR}/ \
  2>&1 | tee  ${EXP_DIR}/log_${now}.txt &

