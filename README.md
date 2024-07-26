# Semantic segmentation of road scenarios based on Transformer and dual-branch network

![The overall network architecture](images/1.jpg)

# DataSet preparation
Dataloaders for Cityscapes, Camvid and ADE20K are available in [datasets](./datasets). 

## Requirements

pytorch >= 1.2.0
apex
opencv-python


# Model Checkpoint

## Pretrained Models

Baidu Pan Link: https://pan.baidu.com/s/1s_BdU9PzMTpZoGJtCJZeEA  p24v


# Training

To be note that, our resnet-based methods including fcn, u-net, deeplabv3+, pspnet can be trained by 8-3080-TI gpus with batchsize 8.
Our training contains two steps(Here I give the ):


## 1, Train the base model.
    We found 150-180 epoch is good enough for warm up traning.
```bash
sh ./scripts/train/train_cityscapes_Resnet-101_Transformer_dual_branch.sh
```

## 2, Re-Train with our module with lower LR using pretrained models.


### For Transformer_dual_branch:
  You can use the pretrained ckpt in previous step.
  

# Evaluation


## 1, Single-Scale Evaluation
```bash
sh ./scripts/evaluate_val/eval_cityscapes_Resnet-101_Transformer_dual_branch.sh 
```

## 2, Multi-Scale Evaluation
```bash
sh ./scripts/evaluate_val/eval_cityscapes_Resnet-101_Transformer_dual_branch_ms.sh
```
## 3, Evaluate F-score on Segmentation Boundary.(change the path of snapshot)
```bash
sh ./scripts/evaluate_boundy_fscore/evaluate_cityscapes_deeplabv3_r101_decouple
```

# Submission on Cityscapes

You can submit the results using our checkpoint by running 

```bash
sh ./scripts/submit_tes/submit_cityscapes_ResNet101_Transformer_dual_branch.sh
```

# Demo 
Here we give some demo scripts for using our checkpoints.
You can change the scripts according to your needs.

```bash
python ./test.py
```
