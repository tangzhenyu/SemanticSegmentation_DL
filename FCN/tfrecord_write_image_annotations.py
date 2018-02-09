# Important: We are using PIL to read .png files later.
# This was done on purpose to read indexed png files
# in a special way -- only indexes and not map the indexes
# to actual rgb values. This is specific to PASCAL VOC
# dataset data. If you don't want thit type of behaviour
# consider using skimage.io.imread()
from PIL import Image
import numpy as np
import skimage.io as io
import tensorflow as tf
import os
import sys

filename_pairs = [
('/data_b/bd-recommend/zhenyutang/logs/VOCdevkit/VOC2007/JPEGImages/000032.jpg',
'/data_b/bd-recommend/zhenyutang/logs/VOCdevkit/VOC2007/SegmentationClass/000032.png'),
('/data_b/bd-recommend/zhenyutang/logs/VOCdevkit/VOC2007/JPEGImages/000039.jpg',
'/data_b/bd-recommend/zhenyutang/logs/VOCdevkit/VOC2007/SegmentationClass/000039.png'),
('/data_b/bd-recommend/zhenyutang/logs/VOCdevkit/VOC2007/JPEGImages/000063.jpg',
'/data_b/bd-recommend/zhenyutang/logs/VOCdevkit/VOC2007/SegmentationClass/000063.png')
]

def load_file_names_pairs(dir_path,data_list):
    JPEGImagesPath=os.path.join(dir_path,"JPEGImages")
    SegmentationClassPath=os.path.join(dir_path,"SegmentationClass")
    JPEGImagesArr=[]
    SegmentationClassArr=[]
    print(JPEGImagesPath)
    for filename in os.listdir(JPEGImagesPath):
        JPEGImagesArr.append(os.path.join(JPEGImagesPath,filename))
    print(SegmentationClassPath)
    for filename in os.listdir(SegmentationClassPath):
        SegmentationClassArr.append(os.path.join(SegmentationClassPath,filename))
    print(len(JPEGImagesArr))
    print(len(SegmentationClassArr))
    assert len(JPEGImagesArr) == len(SegmentationClassArr)
    PairArr=[]
    
    fout=open(data_list,"w")

    for i in range(len(JPEGImagesArr)):
        PairArr.append((JPEGImagesArr[i],SegmentationClassArr[i]))
        fout.write(JPEGImagesArr[i] + " " + SegmentationClassArr[i] + "\n")
    fout.close()

    return PairArr 
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


tfrecords_filename = 'pascal_voc_segmentation.tfrecords'

writer = tf.python_io.TFRecordWriter(tfrecords_filename)

# Let's collect the real images to later on compare
# to the reconstructed ones
original_images = []

filename_pairs=load_file_names_pairs("/data_b/bd-recommend/zhenyutang/logs/VOCdevkit/VOC2007/","./data_list")

print(len(filename_pairs))
sys.exit()

for img_path, annotation_path in filename_pairs:
    img = np.array(Image.open(img_path))
    annotation = np.array(Image.open(annotation_path))

    # The reason to store image sizes was demonstrated
    # in the previous example -- we have to know sizes
    # of images to later read raw serialized string,
    # convert to 1d array and convert to respective
    # shape that image used to have.
    height = img.shape[0]
    width = img.shape[1]

    # Put in the original images into array
    # Just for future check for correctness
    original_images.append((img, annotation))

    img_raw = img.tostring()
    annotation_raw = annotation.tostring()

    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(height),
        'width': _int64_feature(width),
        'image_raw': _bytes_feature(img_raw),
        'mask_raw': _bytes_feature(annotation_raw)}))

    writer.write(example.SerializeToString())

writer.close()
