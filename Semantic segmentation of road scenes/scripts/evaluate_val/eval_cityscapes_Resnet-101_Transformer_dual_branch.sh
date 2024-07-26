#!/usr/bin/env bash
echo "Running inference on" ${1}
echo "Saving Results :" ${2}
python3 eval.py \
	  --dataset cityscapes \
    --arch network.semantic_segmentation.Transformer_dual_branch \
    --inference_mode  sliding \
    --scales 1.0 \
    --split val \
    --cv_split 0 \
    --dump_images \
    --ckpt_path ${2} \
    --snapshot ${1}