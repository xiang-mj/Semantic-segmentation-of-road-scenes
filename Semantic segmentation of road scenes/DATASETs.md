
## Here we give Three different datasets example including Cityscapes, Camvid, ADE20K.

## Cityscapes Dataset

### Download Dataset
First of all, please request the dataset from [here](https://www.cityscapes-dataset.com/). You need multiple files.
Both Coarse data and Fine data are used. 
```
- leftImg8bit_trainvaltest.zip
- gtFine_trainvaltest.zip
- gtCoarse.zip
```

### Prepare Folder Structure

Now unzip those files, the desired folder structure will look like,
```
Cityscapes
├── leftImg8bit_trainvaltest
│   ├── leftImg8bit
│   │   ├── train
│   │   │   ├── aachen
│   │   │   │   ├── aachen_000000_000019_leftImg8bit.png
│   │   │   │   ├── aachen_000001_000019_leftImg8bit.png
│   │   │   │   ├── ...
│   │   │   ├── bochum
│   │   │   ├── ...
│   │   ├── val
│   │   ├── test
├── gtFine_trainvaltest
│   ├── gtFine
│   │   ├── train
│   │   │   ├── aachen
│   │   │   │   ├── aachen_000000_000019_gtFine_color.png
│   │   │   │   ├── aachen_000000_000019_gtFine_instanceIds.png
│   │   │   │   ├── aachen_000000_000019_gtFine_labelIds.png
│   │   │   │   ├── aachen_000000_000019_gtFine_polygons.json
│   │   │   │   ├── ...
│   │   │   ├── bochum
│   │   │   ├── ...
│   │   ├── val
│   │   ├── test
├── gtCoarse
│   ├── gtCoarse
│   │   ├── train
│   │   ├── train_extra
│   │   ├── val
```

## CamVid Dataset

Please download and prepare this dataset according to the [tutorial](https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid). The desired folder structure will look like,
```
CamVid
├── train
├── trainannot
├── val
├── valannot
├── test
├── testannot
```

## ADE20K Dataset

Please download this dataset at ADE20K [webpage](https://groups.csail.mit.edu/vision/datasets/ADE20K/)

and we use the semantic segmentation parts. Unzip the file, the desired folder structure will look like this.

```
ADE20K
├── images
│   ├── training
│   ├── validation
├── annotation
│   ├── training
│   ├── validation
```
After that, you can either change the `config.py` or do the soft link according to the default path in config.

For example, 

Suppose you store your dataset at `~/username/data/Cityscapes`, please update the dataset path in `config.py`,
```
__C.DATASET.MAPILLARY_DIR = '~/username/data/Cityscapes'
``` 
