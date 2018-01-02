# coding: utf-8
pascal_root = '/media/D/DataSet/IS/VOCdevkit/VOC2012'

from utils.pascal_voc import get_augmented_pascal_image_annotation_filename_pairs
from utils.tf_records import write_image_annotation_pairs_to_tfrecord

# Returns a list of (image, annotation) filename pairs (filename.jpg, filename.png)
overall_train_image_annotation_filename_pairs, overall_val_image_annotation_filename_pairs =get_augmented_pascal_image_annotation_filename_pairs(pascal_root=pascal_root)

# You can create your own tfrecords file by providing
# your list with (image, annotation) filename pairs here
write_image_annotation_pairs_to_tfrecord(filename_pairs=overall_val_image_annotation_filename_pairs,
                                         tfrecords_filename='pascal_val.tfrecords')

write_image_annotation_pairs_to_tfrecord(filename_pairs=overall_train_image_annotation_filename_pairs,
                                         tfrecords_filename='pascal_train.tfrecords')

