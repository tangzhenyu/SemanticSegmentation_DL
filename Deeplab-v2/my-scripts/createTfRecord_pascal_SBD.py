import tensorflow as tf
import numpy as np
import os
import scipy.io as spio
from matplotlib import pyplot as plt
from scipy.misc import imread

# define base paths for pascal the original VOC dataset training images
base_dataset_dir_voc = '<path-to-voc-2012>/PascalVoc2012/train/VOC2012'
images_folder_name_voc = "JPEGImages/"
annotations_folder_name_voc = "SegmentationClass_1D/"
images_dir_voc = os.path.join(base_dataset_dir_voc, images_folder_name_voc)
annotations_dir_voc = os.path.join(base_dataset_dir_voc, annotations_folder_name_voc)

# define base paths for pascal augmented VOC images
# download: http://home.bharathh.info/pubs/codes/SBD/download.html
base_dataset_dir_aug_voc = '<path-to-aug-voc>/benchmark_RELEASE/dataset'
images_folder_name_aug_voc = "img/"
annotations_folder_name_aug_voc = "cls/"
images_dir_aug_voc = os.path.join(base_dataset_dir_aug_voc, images_folder_name_aug_voc)
annotations_dir_aug_voc = os.path.join(base_dataset_dir_aug_voc, annotations_folder_name_aug_voc)



def get_files_list(base_dataset_dir, images_folder_name, annotations_folder_name, filename):
    images_dir = os.path.join(base_dataset_dir, images_folder_name)
    annotations_dir = os.path.join(base_dataset_dir, annotations_folder_name)

    file = open(filename, 'r')
    images_filename_list = [line for line in file]
    return images_filename_list

images_filename_list = get_files_list(base_dataset_dir_aug_voc, images_folder_name_aug_voc, annotations_folder_name_aug_voc, "custom_train.txt")
print("Total number of training images:", len(images_filename_list))

# shuffle array and separate 10% to validation
np.random.shuffle(images_filename_list)
val_images_filename_list = images_filename_list[:int(0.10*len(images_filename_list))]
train_images_filename_list = images_filename_list[int(0.10*len(images_filename_list)):]


print("train set size:", len(train_images_filename_list))
print("val set size:", len(val_images_filename_list))

TRAIN_DATASET_DIR="./dataset/"
TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'validation.tfrecords'
train_writer = tf.python_io.TFRecordWriter(os.path.join(TRAIN_DATASET_DIR,TRAIN_FILE))
val_writer = tf.python_io.TFRecordWriter(os.path.join(TRAIN_DATASET_DIR,VALIDATION_FILE))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def read_annotation_from_mat_file(annotations_dir, image_name):
    annotations_path = os.path.join(annotations_dir, (image_name.strip() + ".mat"))
    mat = spio.loadmat(annotations_path)
    img = mat['GTcls']['Segmentation'][0][0]
    return img

def create_tfrecord_dataset(filename_list, writer):

    # create training tfrecord
    for i, image_name in enumerate(filename_list):

        try:
            image_np = imread(os.path.join(images_dir_aug_voc, image_name.strip() + ".jpg"))
        except FileNotFoundError:
            # read from Pascal VOC path
            image_np = imread(os.path.join(images_dir_voc, image_name.strip() + ".jpg"))
            
        try:
            annotation_np = read_annotation_from_mat_file(annotations_dir_aug_voc, image_name)
        except FileNotFoundError:
            # read from Pascal VOC path
            annotation_np = imread(os.path.join(annotations_dir_voc, image_name.strip() + ".png"))
        
        image_h = image_np.shape[0]
        image_w = image_np.shape[1]

        img_raw = image_np.tostring()
        annotation_raw = annotation_np.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(image_h),
                'width': _int64_feature(image_w),
                'image_raw': _bytes_feature(img_raw),
                'annotation_raw': _bytes_feature(annotation_raw)}))

        writer.write(example.SerializeToString())
    
    print("End of TfRecord. Total of image written:", i)
    writer.close()

create_tfrecord_dataset(train_images_filename_list, train_writer)

create_tfrecord_dataset(val_images_filename_list, val_writer)