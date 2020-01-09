# Deep Image Matting

Deep Image Matting [paper](https://arxiv.org/abs/1703.03872) implementation in PyTorch.

## Differences

1. "fc6" is dropped.
2. Indices pooling.

<p>"fc6" is clumpy, over 100 millions parameters, makes the model hard to converge. I guess it is the reason why the model (paper) has to be trained stagewisely.

## Performance
- The Composition-1k testing dataset.
- Evaluate with whole image.
- SAD normalized by 1000.
- Input image is normalized with mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
- Both erode and dialte to generate trimap.

|Models|SAD|MSE|Download|
|---|---|---|---|
|paper-stage0|59.6|0.019||
|paper-stage1|54.6|0.017||
|paper-stage3|50.4|0.014||
|my-stage0|66.8|0.024|[Link](https://github.com/foamliu/Deep-Image-Matting-PyTorch/releases/download/v1.0/BEST_checkpoint.tar)|

## Dependencies

- Python 3.5.2
- PyTorch 1.1.0

## Dataset
### Adobe Deep Image Matting Dataset
Follow the [instruction](https://sites.google.com/view/deepimagematting) to contact author for the dataset.

### MSCOCO
Go to [MSCOCO](http://cocodataset.org/#download) to download:
* [2014 Train images](http://images.cocodataset.org/zips/train2014.zip)


### PASCAL VOC
Go to [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) to download:
* VOC challenge 2008 [training/validation data](http://host.robots.ox.ac.uk/pascal/VOC/voc2008/VOCtrainval_14-Jul-2008.tar)
* The test data for the VOC2008 challenge

## Usage
### Data Pre-processing
Extract training images:
```bash
$ python pre_process.py
```

### Train
```bash
$ python train.py
```

If you want to visualize during training, run in your terminal:
```bash
$ tensorboard --logdir runs
```

## Experimental results

### The Composition-1k testing dataset

1. Test:
```bash
$ python test.py
```

It prints out average SAD and MSE errors when finished.

### The alphamatting.com dataset

1. Download the evaluation datasets: Go to the [Datasets page](http://www.alphamatting.com/datasets.php) and download the evaluation datasets. Make sure you pick the low-resolution dataset.

2. Extract evaluation images:
```bash
$ python extract.py
```

3. Evaluate:
```bash
$ python eval.py
```

Click to view whole images:

Image | Trimap1 | Trimap2 | Trimap3|
|---|---|---|---|
|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/data/alphamatting/input_lowres/doll.png) |![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/data/alphamatting/trimap_lowres/Trimap1/doll.png)|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/data/alphamatting/trimap_lowres/Trimap2/doll.png)|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/data/alphamatting/trimap_lowres/Trimap3/doll.png)|
|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/data/alphamatting/input_lowres/doll.png) |![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/alphamatting/output_lowres/Trimap1/doll.png)|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/alphamatting/output_lowres/Trimap2/doll.png)|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/alphamatting/output_lowres/Trimap3/doll.png)|
|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/data/alphamatting/input_lowres/donkey.png) |![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/data/alphamatting/trimap_lowres/Trimap1/donkey.png)|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/data/alphamatting/trimap_lowres/Trimap2/donkey.png)|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/data/alphamatting/trimap_lowres/Trimap3/donkey.png)|
|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/data/alphamatting/input_lowres/donkey.png) |![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/alphamatting/output_lowres/Trimap1/donkey.png)|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/alphamatting/output_lowres/Trimap2/donkey.png)|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/alphamatting/output_lowres/Trimap3/donkey.png)|
|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/data/alphamatting/input_lowres/elephant.png) |![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/data/alphamatting/trimap_lowres/Trimap1/elephant.png)|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/data/alphamatting/trimap_lowres/Trimap2/elephant.png)|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/data/alphamatting/trimap_lowres/Trimap3/elephant.png)|
|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/data/alphamatting/input_lowres/elephant.png) |![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/alphamatting/output_lowres/Trimap1/elephant.png)|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/alphamatting/output_lowres/Trimap2/elephant.png)|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/alphamatting/output_lowres/Trimap3/elephant.png)|
|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/data/alphamatting/input_lowres/net.png) |![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/data/alphamatting/trimap_lowres/Trimap1/net.png)|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/data/alphamatting/trimap_lowres/Trimap2/net.png)|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/data/alphamatting/trimap_lowres/Trimap3/net.png)|
|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/data/alphamatting/input_lowres/net.png) |![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/alphamatting/output_lowres/Trimap1/net.png)|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/alphamatting/output_lowres/Trimap2/net.png)|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/alphamatting/output_lowres/Trimap3/net.png)|
|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/data/alphamatting/input_lowres/pineapple.png) |![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/data/alphamatting/trimap_lowres/Trimap1/pineapple.png)|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/data/alphamatting/trimap_lowres/Trimap2/pineapple.png)|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/data/alphamatting/trimap_lowres/Trimap3/pineapple.png)|
|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/data/alphamatting/input_lowres/pineapple.png) |![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/alphamatting/output_lowres/Trimap1/pineapple.png)|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/alphamatting/output_lowres/Trimap2/pineapple.png)|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/alphamatting/output_lowres/Trimap3/pineapple.png)|
|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/data/alphamatting/input_lowres/plant.png) |![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/data/alphamatting/trimap_lowres/Trimap1/plant.png)|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/data/alphamatting/trimap_lowres/Trimap2/plant.png)|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/data/alphamatting/trimap_lowres/Trimap3/plant.png)|
|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/data/alphamatting/input_lowres/plant.png) |![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/alphamatting/output_lowres/Trimap1/plant.png)|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/alphamatting/output_lowres/Trimap2/plant.png)|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/alphamatting/output_lowres/Trimap3/plant.png)|
|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/data/alphamatting/input_lowres/plasticbag.png) |![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/data/alphamatting/trimap_lowres/Trimap1/plasticbag.png)|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/data/alphamatting/trimap_lowres/Trimap2/plasticbag.png)|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/data/alphamatting/trimap_lowres/Trimap3/plasticbag.png)|
|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/data/alphamatting/input_lowres/plasticbag.png) |![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/alphamatting/output_lowres/Trimap1/plasticbag.png)|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/alphamatting/output_lowres/Trimap2/plasticbag.png)|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/alphamatting/output_lowres/Trimap3/plasticbag.png)|
|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/data/alphamatting/input_lowres/troll.png) |![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/data/alphamatting/trimap_lowres/Trimap1/troll.png)|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/data/alphamatting/trimap_lowres/Trimap2/troll.png)|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/data/alphamatting/trimap_lowres/Trimap3/troll.png)|
|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/data/alphamatting/input_lowres/troll.png) |![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/alphamatting/output_lowres/Trimap1/troll.png)|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/alphamatting/output_lowres/Trimap2/troll.png)|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/alphamatting/output_lowres/Trimap3/troll.png)|

### Demo
Download pre-trained Deep Image Matting [Link](https://github.com/foamliu/Deep-Image-Matting-PyTorch/releases/download/v1.0/BEST_checkpoint.tar) then run:
```bash
$ python demo.py
```

Image/Trimap | Output/GT | New BG/Compose | 
|---|---|---|
|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/0_image.png)  | ![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/0_out.png)   | ![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/0_new_bg.png) |
|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/0_trimap.png) | ![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/0_alpha.png) | ![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/0_compose.png)|
|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/1_image.png)  | ![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/1_out.png)   | ![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/1_new_bg.png) | 
|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/1_trimap.png) | ![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/1_alpha.png) | ![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/1_compose.png)|
|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/2_image.png)  | ![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/2_out.png)   | ![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/2_new_bg.png) |
|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/2_trimap.png) | ![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/2_alpha.png) | ![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/2_compose.png)|
|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/3_image.png)  | ![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/3_out.png)   | ![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/3_new_bg.png) |
|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/3_trimap.png) | ![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/3_alpha.png) | ![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/3_compose.png)|
|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/4_image.png)  | ![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/4_out.png)   | ![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/4_new_bg.png) |
|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/4_trimap.png) | ![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/4_alpha.png) | ![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/4_compose.png)|
|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/5_image.png)  | ![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/5_out.png)   | ![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/5_new_bg.png) |
|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/5_trimap.png) | ![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/5_alpha.png) | ![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/5_compose.png)|
|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/6_image.png)  | ![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/6_out.png)   | ![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/6_new_bg.png) |
|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/6_trimap.png) | ![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/6_alpha.png) | ![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/6_compose.png)|
|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/7_image.png)  | ![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/7_out.png)   | ![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/7_new_bg.png) |
|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/7_trimap.png) | ![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/7_alpha.png) | ![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/7_compose.png)|
|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/8_image.png)  | ![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/8_out.png)   | ![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/8_new_bg.png) |
|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/8_trimap.png) | ![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/8_alpha.png) | ![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/8_compose.png)|
|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/9_image.png)  | ![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/9_out.png)   | ![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/9_new_bg.png) |
|![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/9_trimap.png) | ![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/9_alpha.png) | ![image](https://github.com/foamliu/Deep-Image-Matting-PyTorch/raw/master/images/9_compose.png)|
