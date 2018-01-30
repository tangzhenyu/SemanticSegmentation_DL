## semantic-segmentation-implementations

### DataSets Usage
[Kitti Dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php)
[MIT ADE20K scene parsing dataset](https://github.com/hangzhaomit/semantic-segmentation-pytorch)
[Cityscapes dataset](https://www.cityscapes-dataset.com/benchmarks/)

## Models
+ [FCN:Fully Convolutional Networks for Semantic Segmentation](http://arxiv.org/abs/1411.4038) [Paper2](http://arxiv.org/abs/1605.06211)
+ [RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation](https://arxiv.org/abs/1611.06612)
+ [PSPNet:Pyramid Scene Parsing Network](https://arxiv.org/abs/1612.01105)
+ [ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation](https://arxiv.org/pdf/1606.02147.pdf)
+ [ICNet for Real-Time Semantic Segmentation on High-Resolution Images](https://arxiv.org/abs/1704.08545)  
+ [FC-DenseNet:The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation](https://arxiv.org/abs/1611.09326)
+ [Mobile UNet for Semantic Segmentation](https://arxiv.org/abs/1704.04861)
+ [Encoder-Decoder with skip connections based on SegNet](https://arxiv.org/abs/1511.00561)
+ [Encoder-Decoder based on SegNet](https://arxiv.org/abs/1511.00561)
+ [Full-Resolution Residual Networks for Semantic Segmentation in Street Scenes](https://arxiv.org/abs/1611.08323)

## Installation
This project has the following dependencies:

- Numpy `sudo pip install numpy`

- OpenCV Python `sudo apt-get install python-opencv`

- TensorFlow `sudo pip install --upgrade tensorflow-gpu`

## Training
The only thing you have to do to get started is set up the folders in the following structure:

    ├── "dataset_name"    
    |   ├── train
    |   ├── train_labels
    |   ├── val
    |   ├── val_labels
    |   ├── test
    |   ├── test_labels

python train.py

## Test
Get corresponding transformed pre-trained weights, and put into `model` directory:   
```
python inference.py --img-path /Path/To/Image --dataset Model_Type
```
### optional arguments:
```
--dataset - choose from "RefineNet-Res50"/"icnet"/"PSPNet"/"fcn"/"enet"  
--model MODEL  -The model you are using
```

### Import module in your code:
```python
from model import FCN8s, PSPNet50, ICNet, ENet

model = PSPNet50() # or another model

model.read_input(img_path)  # read image data from path

sess = tf.Session(config=config)
init = tf.global_variables_initializer()
sess.run(init)

model.load(model_path, sess)  # load pretrained model
preds = model.forward(sess) # Get prediction 
```

## Referrence
+ [PSPNet:Pyramid Scene Parsing Network](https://github.com/hszhao/PSPNet),
+ [FCN](https://github.com/CSAILVision/sceneparsing)
+ [ICNet](https://github.com/hszhao/ICNet)
+ [ENet](https://github.com/fregu856/segmentation).
+ [Semantic understanding of scenes through the ade20k dataset](http://people.csail.mit.edu/bzhou/publication/scene-parse-camera-ready.pdf)
+ [Semantic understanding of scenes through the ade20k dataset](https://arxiv.org/pdf/1608.05442.pdf)
