#!/usr/bin/env bash
now=$(date +"%Y%m%d_%H%M%S")
EXP_DIR=./body_edge/deepv3_decouple_r101_F_score
mkdir -p ${EXP_DIR}

python -m torch.distributed.launch --nproc_per_node=8 train.py \
  --dataset cityscapes \
  --cv 0 \
  --evaluateF \
  --snapshot /home/user/pretrained \
  --arch network.semantic_segmentation.Transformer_dual_branch \
  --lr 0.01 \
  --lr_schedule poly \
  --poly_exp 1.0 \
  --repoly 1.5  \
  --rescale 1.0 \
  --sgd \
  --apex \
  --crop_size 1024 \
  --scale_min 0.5 \
  --max_epoch 180 \
  --ohem \
  --jointwtborder \
  --joint_edgeseg_loss \
  --wt_bound 1.0 \
  --bs_mult 1 \
  --exp cityscapes_ft \
  --ckpt ${EXP_DIR}/ \
  --tb_path ${EXP_DIR}/ \
  2>&1 | tee  ${EXP_DIR}/log_${now}.txt &