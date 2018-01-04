
基于v2版本的deeplab,使用VGG16模型，在VOC2012，Pascal-context，NYU-v2等多个数据集上进行训练。
好记性不如烂笔头, 最近用Deeplab v2跑的图像分割，现记录如下。

----------
官方源码地址如下：https://bitbucket.org/aquariusjay/deeplab-public-ver2/overview 
但是此源码只是为deeplab网络做相应变形的caffe,如果需要fine tuning微调网络，还需要准备以下文件：

 - **txt文件**：文件中有数据集的名字列表的txt文件,[训练测试集列表](https://ucla.box.com/s/rd9z2xvwsfpksi7mi08i2xqrj7ab4keb)
 
 - **训练好的init.caffemodel**: 针对deeplab v2，作者有已经预训练好的两个模型参数：[DeepLabv2_VGG16 ](http://liangchiehchen.com/projects/released/deeplab_aspp_vgg16/prototxt_and_model.zip)和[DeepLabv2_ResNet101
](http://liangchiehchen.com/projects/released/deeplab_aspp_resnet101/prototxt_and_model.zip) 

 - **网络结构prototxt文件**: train.prototxt和solver.prototxt，分别在：[DeepLabv2_VGG16 ](http://liangchiehchen.com/projects/released/deeplab_aspp_vgg16/prototxt_and_model.zip)和 [DeepLabv2_ResNet101
](http://liangchiehchen.com/projects/released/deeplab_aspp_resnet101/prototxt_and_model.zip) 
 - **官网脚本文件**: [三个sh文件](https://ucla.box.com/s/4grlj8yoodv95936uybukjh5m0tdzvrf)，建议使用脚本文件，初看虽不懂，但是比[python版本](https://github.com/TheLegendAli/CCVL)的运行简单很多
注：本博客只涉及脚本版本的训练


----------


----------


## 准备工作
### 1.必要工具
   下载安装matio,[下载地址](http://sourceforge.net/projects/matio/files/matio/1.5.2/)
### 2.数据集准备
文章中使用的数据集并不全是pascal-voc2012,而是由voc2012和另外一个数据集合并而成
 **数据下载**

```
# augmented PASCAL VOC
mkdir -p ~/DL_dataset
cd ~/DL_dataset       #save datasets 为$DATASETS
wget http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz # 1.3 GB
tar -zxvf benchmark.tgz
mv benchmark_RELEASE VOC_aug

# original PASCAL VOC 2012
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar # 2 GB
tar -xvf VOCtrainval_11-May-2012.tar
mv VOCdevkit/VOC2012 VOC2012_orig && rm -r VOCdevkit
```


**数据转换**
因为pascal voc2012增强数据集的label是mat格式的文件，所以我们需要把mat格式的label转为png格式的图片.

```
cd ~/DL_dataset/VOC_aug/dataset
mkdir cls_png
cd ~/deeplab_v2/voc2012/
./mat2png.py ~/DL_dataset/VOC_aug/dataset/cls ~/DL_dataset/VOC_aug/dataset/cls_png
```
pascal voc2012原始数据集的label为三通道RGB图像，但是caffe最后一层softmax loss 层只能识别一通道的label,所以此处我们需要对原始数据集的label进行降维

```
cd ~/DL_dataset/VOC2012_orig
mkdir SegmentationClass_1D

cd ~/deeplab_v2/voc2012
./convert_labels.py ~/DL_dataset/VOC2012_orig/SegmentationClass/ \  ~/DL_dataset/VOC2012_orig/ImageSets/Segmentation/trainval.txt \ ~/DL_dataset/VOC2012_orig/SegmentationClass_1D/

```

**数据融合**
此时我们已经处理好好pascal voc2012 增强数据集和pascal voc2012的原始数据集，为了便于train.txt等文件的调用，我们需要将两个文件夹数据合并到同一个文件中.目前已有数据文件如下：

 

 1. ~/DL_dataset/VOC2012_orig 为原始pascal voc2012文件夹
  - images数据集的文件名为：JPEGImages 
 - labels数据集文件名为：SegmentationClass_1D

 2.   ~/DL_dataset/VOC_aug/dataset为pascal voc2012增强数据集文件夹
   - images数据集的文件名为：img
 ，jpg图片数为5073
  - labels数据集文件名为：cls_png，png图片数11355
 
 现分别pascal voc2012增强数据集里的images和labels复制到增强数据集中，若重复则覆盖，合将并数据集的操作如下：
```
cp ~/DL_dataset/VOC2012_orig/SegmentationClass_1D/* ~/DL_dataset/VOC_aug/dataset/cls_png
cp ~/DL_dataset/VOC2012_orig/JPEGImages/* ~/DL_dataset/VOC_aug/dataset/img/

```
**文件名修改**
对应[train.txt文件](https://github.com/xmojiao/deeplab_v2/blob/master/voc2012/list/train_aug.txt)的数据集文件名，修改文件名。

```
cd ~/DL_dataset/VOC_aug/dataset
mv ./img ./JPEGImages
mv ./cls_png ./SegmentationClassAug
```


![这里写图片描述](http://img.blog.csdn.net/20170908235046830?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvWG1vX2ppYW8=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)


----------
到此处， ~/DL_dataset/VOC_aug/dataset文件夹中
   - images数据集的文件名为：JPEGImages ,jpg图片数由5073变为17125
 - labels数据集文件名为：cls_png ，png图片数由11355变为12031


----------


----------


### 1.从github克隆train deeplab_v2文件夹
此github已经将文件夹结构建好，并下载放置对应的txt文件，prototxt文件，脚本sh文件，放置到对应文件夹下。由于官方model文件大，不宜放GitHub，将在第3步下载。
```
cd ~
git clone git@github.com:xmojiao/deeplab_v2.git
```


----------


### 2.将源码下载到此文件夹下，并编译安装deeplab caffe 

```
cd deeplab_v2
git clone https://bitbucket.org/aquariusjay/deeplab-public-ver2.git
cd deeplab-public-ver2
make all
make pycaffe
make test # NOT mandatory
make runtest # NOT mandatory

```


----------


### 3.将官方预训练的model放置到voc2012的model/deeplab_largeFOV

此处以VGG16训练为例，model[下载地址](http://liangchiehchen.com/projects/released/deeplab_aspp_vgg16/prototxt_and_model.zip)
也可在命令行下载并移动到相应文件夹，如下：
```
wget http://liangchiehchen.com/projects/released/deeplab_aspp_vgg16/prototxt_and_model.zip
unzip prototxt_and_model.zip
mv *caffemodel ~/deeplab_v2/model/deeplab_largeFOV
rm *prototxt
```



-------------------

### 4.deeplab2的script脚本文件run_pascal.sh 解析
目前我们已经准备好数据集和数据txt文件，参数文件model，网络结构文件prototxt,和三个sh脚本文件，接下来只需要修改run_pascal.sh文件，deeplabv2就可以run起来了。啦啦啦，让我们拭目以待吧！

```
#!/bin/sh

## MODIFY PATH for YOUR SETTING
ROOT_DIR=/home/dl/DL_dataset  #此处为voc数据集主路径

CAFFE_DIR=../deeplab-public-ver2 #此处为官方deeplab源码的路径
CAFFE_BIN=${CAFFE_DIR}/.build_release/tools/caffe.bin

EXP=.   #此目录路径~/deeplab_v2/voc2012
if [ "${EXP}" = "." ]; then  #若数据集为voc2012，则分类数为21，数DATA_ROOT据集具体路径
    NUM_LABELS=21
    DATA_ROOT=${ROOT_DIR}/VOC_aug/dataset/
else
    NUM_LABELS=0
    echo "Wrong exp name"
fi
 

## Specify which model to train
########### voc12 ################
NET_ID=deeplab_largeFOV ##此处原文件名有问题应该改为deeplab_largeFOV



## Variables used for weakly or semi-supervisedly training
#TRAIN_SET_SUFFIX=
TRAIN_SET_SUFFIX=_aug  #此处为选择train_aug.txt数据集


#TRAIN_SET_STRONG=train
#TRAIN_SET_STRONG=train200
#TRAIN_SET_STRONG=train500
#TRAIN_SET_STRONG=train1000
#TRAIN_SET_STRONG=train750

#TRAIN_SET_WEAK_LEN=5000

DEV_ID=0

#####

## Create dirs

CONFIG_DIR=${EXP}/config/${NET_ID} #此处目录为/voc2012/config/deeplab_largeFOV
MODEL_DIR=${EXP}/model/${NET_ID}
mkdir -p ${MODEL_DIR}
LOG_DIR=${EXP}/log/${NET_ID}
mkdir -p ${LOG_DIR}
export GLOG_log_dir=${LOG_DIR}

## Run

RUN_TRAIN=1 #为1说明执行train
RUN_TEST=0 #为1说明执行test
RUN_TRAIN2=0
RUN_TEST2=0

## Training #1 (on train_aug)

if [ ${RUN_TRAIN} -eq 1 ]; then  #r如果RUN_TRAIN为1
    #
    LIST_DIR=${EXP}/list
    TRAIN_SET=train${TRAIN_SET_SUFFIX}
    if [ -z ${TRAIN_SET_WEAK_LEN} ]; then #如果TRAIN_SET_WEAK_LEN长度为零则为真
				TRAIN_SET_WEAK=${TRAIN_SET}_diff_${TRAIN_SET_STRONG}
				comm -3 ${LIST_DIR}/${TRAIN_SET}.txt ${LIST_DIR}/${TRAIN_SET_STRONG}.txt > ${LIST_DIR}/${TRAIN_SET_WEAK}.txt
    else
				TRAIN_SET_WEAK=${TRAIN_SET}_diff_${TRAIN_SET_STRONG}_head${TRAIN_SET_WEAK_LEN}
				comm -3 ${LIST_DIR}/${TRAIN_SET}.txt ${LIST_DIR}/${TRAIN_SET_STRONG}.txt | head -n ${TRAIN_SET_WEAK_LEN} > ${LIST_DIR}/${TRAIN_SET_WEAK}.txt
    fi
    #
    MODEL=${EXP}/model/${NET_ID}/init.caffemodel #下载的vgg16或者ResNet101中的 model
    #
    echo Training net ${EXP}/${NET_ID}
    for pname in train solver; do
				sed "$(eval echo $(cat sub.sed))" \
						${CONFIG_DIR}/${pname}.prototxt > ${CONFIG_DIR}/${pname}_${TRAIN_SET}.prototxt
    done  #此部分运行时如以下命令
        CMD="${CAFFE_BIN} train \
         --solver=${CONFIG_DIR}/solver_${TRAIN_SET}.prototxt \
         --gpu=${DEV_ID}"
		if [ -f ${MODEL} ]; then
				CMD="${CMD} --weights=${MODEL}"
		fi
		echo Running ${CMD} && ${CMD}
fi
#train部分运行时，即以下运行命令 ../deeplab-public-ver2/.build_release/tools/caffe.bin train --solver=volab_largeFOV/solver_train_aug.prototxt --gpu=0 --weights=voc12/model/deeplab_largeFOV/init.caf   femodel
#上述命令中，solver_train_aug.prototxt由solve.prototxt文件复制而来，init.caffemodel为原始下载了的VGG16的model

## Test #1 specification (on val or test)

if [ ${RUN_TEST} -eq 1 ]; then
    #
    for TEST_SET in val; do
				TEST_ITER=`cat ${EXP}/list/${TEST_SET}.txt | wc -l` #此处计算val.txt文件中测试图片个数，共1449个
				MODEL=${EXP}/model/${NET_ID}/test.caffemodel
				if [ ! -f ${MODEL} ]; then
						MODEL=`ls -t ${EXP}/model/${NET_ID}/train_iter_*.caffemodel | head -n 1`
				fi
				#
				echo Testing net ${EXP}/${NET_ID}
				FEATURE_DIR=${EXP}/features/${NET_ID}
				mkdir -p ${FEATURE_DIR}/${TEST_SET}/fc8
        mkdir -p ${FEATURE_DIR}/${TEST_SET}/fc9
				mkdir -p ${FEATURE_DIR}/${TEST_SET}/seg_score
				sed "$(eval echo $(cat sub.sed))" \
						${CONFIG_DIR}/test.prototxt > ${CONFIG_DIR}/test_${TEST_SET}.prototxt
				CMD="${CAFFE_BIN} test \
             --model=${CONFIG_DIR}/test_${TEST_SET}.prototxt \
             --weights=${MODEL} \
             --gpu=${DEV_ID} \
             --iterations=${TEST_ITER}"
				echo Running ${CMD} && ${CMD}
    done
fi
#test部分运行时，即以下运行命令../deeplab-public-ver2/.build_release/tools/caffe.bin test --model=voc12/config/deeplab_largeFOV/test_val.prototxt --weights=voc12/model/deeplab_largeFOV/train_iter_20000.caffemodel --gpu=0 --iterations=1449
#上述命令中，test_val.prototxt由test.prototxt文件复制而来，train_iter_20000.caffemode由第一部分train得到的model
```


----------


### 5.deeplab跑起来

此处我将train和test分开操作，即是修改run_pascal.sh脚本中的如下代码：
![这里写图片描述](http://img.blog.csdn.net/20170910225002960?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvWG1vX2ppYW8=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)


----------


 - RUN_TRAIN=1 时 
 

```
cd ~/deeplab_v2/voc2012
sh run_pascal.sh 2>&1|tee train.log

```

2>&1|tee train.log
 指令的作用为在命令行展示log的同时，保存log到当前目录的train.log文件夹。前工作做的顺利的话，你就能看到如下结果。

![这里写图片描述](http://img.blog.csdn.net/20170909004756855?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvWG1vX2ppYW8=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

![这里写图片描述](http://img.blog.csdn.net/20170909004817659?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvWG1vX2ppYW8=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

----------




 - RUN_TEST=1

目前没发现作者有写单张图片测试的代码，但是当我们跑此部分run_test时，会得到png格式的测试结果 

**跑出测试结果**

 

```
sh run_pascal.sh 2>&1|tee train.log

```



前工作做的顺利的话，你就能看到如下log
![这里写图片描述](http://img.blog.csdn.net/20170910225603453?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvWG1vX2ppYW8=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)


----------


----------


### 6.将test的结果mat文件转换为png文件
test结束，你会在~/deeplab_v2/voc2012/features/deeplab_largeFOV/val/fc8目录下跑出mat格式的结果。
![这里写图片描述](http://img.blog.csdn.net/20170910225908184?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvWG1vX2ppYW8=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)


----------


**mat转png图片**
-修改creat_labels.py中文件目录

```
cd ~/deeplab_v2/voc2012/
vim create_labels.py
```




![这里写图片描述](http://img.blog.csdn.net/20170910230640226?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvWG1vX2ppYW8=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)


----------


-在此目录运行creat_labels.py

`python create_labels.py`

大功告成，可以看到结果如下图：
![这里写图片描述](http://img.blog.csdn.net/20170910232511018?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvWG1vX2ppYW8=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)


取第一张图片2007_000033.jpg将其放大对比：
![这里写图片描述](http://img.blog.csdn.net/20170910232609797?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvWG1vX2ppYW8=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
