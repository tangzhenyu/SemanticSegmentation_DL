import os


def pascal_segmentation_lut():
    """Return look-up table with number and correspondng class names
    for PASCAL VOC segmentation dataset. Two special classes are: 0 -
    background and 255 - ambigious region. All others are numerated from
    1 to 20.
    
    Returns
    -------
    classes_lut : dict
        look-up table with number and correspondng class names
    """

    class_names = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                   'dog', 'horse', 'motorbike', 'person', 'potted-plant',
                   'sheep', 'sofa', 'train', 'tv/monitor', 'ambigious']
    
    enumerated_array = enumerate(class_names[:-1])
    
    classes_lut = list(enumerated_array)
    
    # Add a special class representing ambigious regions
    # which has index 255.
    classes_lut.append((255, class_names[-1]))
    
    classes_lut = dict(classes_lut)

    return classes_lut


def get_pascal_segmentation_images_lists_txts(pascal_root):
    segmentation_images_lists_relative_folder = 'ImageSets/Segmentation'
    
    segmentation_images_lists_folder = os.path.join(pascal_root,
                                                    segmentation_images_lists_relative_folder)
    
    pascal_train_list_filename = os.path.join(segmentation_images_lists_folder,
                                              'train.txt')

    pascal_validation_list_filename = os.path.join(segmentation_images_lists_folder,
                                                   'val.txt')
    
    pascal_trainval_list_filname = os.path.join(segmentation_images_lists_folder,
                                                'trainval.txt')
    
    return [
            pascal_train_list_filename,
            pascal_validation_list_filename,
            pascal_trainval_list_filname
           ]


def readlines_with_strip(filename):
    # Get raw filnames from the file
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Clean filenames from whitespaces and newline symbols
    clean_lines = map(lambda x: x.strip(), lines)
    
    return clean_lines


def readlines_with_strip_array_version(filenames_array):
    multiple_files_clean_lines = map(readlines_with_strip, filenames_array)
    
    return multiple_files_clean_lines


def add_full_path_and_extention_to_filenames(filenames_array, full_path, extention):
    full_filenames = map(lambda x: os.path.join(full_path, x) + '.' + extention, filenames_array)
    
    return full_filenames


def add_full_path_and_extention_to_filenames_array_version(filenames_array_array, full_path, extention):
    result = map(lambda x: add_full_path_and_extention_to_filenames(x, full_path, extention),
                 filenames_array_array)
    
    return result


def get_pascal_segmentation_image_annotation_filenames_pairs(pascal_root):
    
    pascal_relative_images_folder = 'JPEGImages'
    pascal_relative_class_annotations_folder = 'SegmentationClass'
    
    images_extention = 'jpg'
    annotations_extention = 'png'
    
    pascal_images_folder = os.path.join(pascal_root, pascal_relative_images_folder)
    pascal_class_annotations_folder = os.path.join(pascal_root, pascal_relative_class_annotations_folder)
    
    pascal_images_lists_txts = get_pascal_segmentation_images_lists_txts(pascal_root)
    
    pascal_image_names = readlines_with_strip_array_version(pascal_images_lists_txts)
    
    images_full_names = add_full_path_and_extention_to_filenames_array_version(pascal_image_names,
                                                                               pascal_images_folder,
                                                                               images_extention)
    
    annotations_full_names = add_full_path_and_extention_to_filenames_array_version(pascal_image_names,
                                                                                    pascal_class_annotations_folder,
                                                                                    annotations_extention)
    

    temp = zip(images_full_names, annotations_full_names)

    image_annotation_filename_pairs = map(lambda x: zip(*x), temp)
    
    return image_annotation_filename_pairs


def get_pascal_selected_image_annotation_filenames_pairs(pascal_root, selected_names):
    """Returns (image, annotation) filenames pairs from PASCAL VOC segmentation dataset for selected names.
    The function accepts the selected file names from PASCAL VOC segmentation dataset
    and returns image, annotation pairs with fullpath and extention for those names.
    Parameters
    ----------
    pascal_root : string
        Path to the PASCAL VOC dataset root that is usually named 'VOC2012'
        after being extracted from tar file.
    selected_names : array of strings
        Selected filenames from PASCAL VOC that can be read from txt files that
        come with dataset.
    Returns
    -------
    image_annotation_pairs : 
        Array with filename pairs with fullnames.
    """
    pascal_relative_images_folder = 'JPEGImages'
    pascal_relative_class_annotations_folder = 'SegmentationClass'
    
    images_extention = 'jpg'
    annotations_extention = 'png'
    
    pascal_images_folder = os.path.join(pascal_root, pascal_relative_images_folder)
    pascal_class_annotations_folder = os.path.join(pascal_root, pascal_relative_class_annotations_folder)
    
    images_full_names = add_full_path_and_extention_to_filenames(selected_names,
                                                                 pascal_images_folder,
                                                                 images_extention)
    
    annotations_full_names = add_full_path_and_extention_to_filenames(selected_names,
                                                                      pascal_class_annotations_folder,
                                                                      annotations_extention)
    
    image_annotation_pairs = zip(images_full_names, 
                                 annotations_full_names)
    
    return image_annotation_pairs

def get_augmented_pascal_image_annotation_filename_pairs(pascal_root):

    pascal_txts = get_pascal_segmentation_images_lists_txts(pascal_root=pascal_root)

    pascal_name_lists = readlines_with_strip_array_version(pascal_txts)

    pascal_train_name_set, pascal_val_name_set, _ = map(lambda x: set(x), pascal_name_lists)

    all_pascal = pascal_train_name_set | pascal_val_name_set

    everything = all_pascal
    validation = pascal_val_name_set

    # The rest of the dataset is for training
    train = everything - validation


    # The rest of the data will be loaded from pascal
    train_from_pascal = train


    train_from_pascal_image_annotation_pairs = \
    get_pascal_selected_image_annotation_filenames_pairs(pascal_root,
                                                         list(train_from_pascal))

    overall_train_image_annotation_filename_pairs = train_from_pascal_image_annotation_pairs

    overall_val_image_annotation_filename_pairs = \
    get_pascal_selected_image_annotation_filenames_pairs(pascal_root,
                                                         validation)

    return overall_train_image_annotation_filename_pairs, overall_val_image_annotation_filename_pairs