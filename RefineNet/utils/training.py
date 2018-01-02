import tensorflow as tf


def get_labels_from_annotation(annotation_tensor, class_labels):
    """Returns tensor of size (width, height, num_classes) derived from annotation tensor.
    The function returns tensor that is of a size (width, height, num_classes) which
    is derived from annotation tensor with sizes (width, height) where value at
    each position represents a class. The functions requires a list with class
    values like [0, 1, 2 ,3] -- they are used to derive labels. Derived values will
    be ordered in the same way as the class numbers were provided in the list. Last
    value in the aforementioned list represents a value that indicate that the pixel
    should be masked out. So, the size of num_classes := len(class_labels) - 1.
    
    Parameters
    ----------
    annotation_tensor : Tensor of size (width, height)
        Tensor with class labels for each element
    class_labels : list of ints
        List that contains the numbers that represent classes. Last
        value in the list should represent the number that was used
        for masking out.
        
    Returns
    -------
    labels_2d_stacked : Tensor of size (width, height, num_classes).
        Tensor with labels for each pixel.
    """
    
    # Last value in the classes list should show
    # which number was used in the annotation to mask out
    # the ambigious regions or regions that should not be
    # used for training.
    # TODO: probably replace class_labels list with some custom object
    valid_entries_class_labels = class_labels[:-1]
    
    # Stack the binary masks for each class
    labels_2d = map(lambda x: tf.equal(annotation_tensor, x),
                    valid_entries_class_labels)

    # Perform the merging of all of the binary masks into one matrix
    labels_2d_stacked = tf.stack(labels_2d, axis=2)
    
    # Convert tf.bool to tf.float
    # Later on in the labels and logits will be used
    # in tf.softmax_cross_entropy_with_logits() function
    # where they have to be of the float type.
    labels_2d_stacked_float = tf.to_float(labels_2d_stacked)
    
    return labels_2d_stacked_float

def get_labels_from_annotation_batch(annotation_batch_tensor, class_labels):
    """Returns tensor of size (batch_size, width, height, num_classes) derived
    from annotation batch tensor. The function returns tensor that is of a size
    (batch_size, width, height, num_classes) which is derived from annotation tensor
    with sizes (batch_size, width, height) where value at each position represents a class.
    The functions requires a list with class values like [0, 1, 2 ,3] -- they are
    used to derive labels. Derived values will be ordered in the same way as
    the class numbers were provided in the list. Last value in the aforementioned
    list represents a value that indicate that the pixel should be masked out.
    So, the size of num_classes len(class_labels) - 1.
    
    Parameters
    ----------
    annotation_batch_tensor : Tensor of size (batch_size, width, height)
        Tensor with class labels for each element
    class_labels : list of ints
        List that contains the numbers that represent classes. Last
        value in the list should represent the number that was used
        for masking out.
        
    Returns
    -------
    batch_labels : Tensor of size (batch_size, width, height, num_classes).
        Tensor with labels for each batch.
    """
    
    batch_labels = tf.map_fn(fn=lambda x: get_labels_from_annotation(annotation_tensor=x, class_labels=class_labels),
                             elems=annotation_batch_tensor,
                             dtype=tf.float32)
    
    return batch_labels

def get_valid_entries_indices_from_annotation_batch(annotation_batch_tensor, class_labels):
    """Returns tensor of size (num_valid_eintries, 3).
    Returns tensor that contains the indices of valid entries according
    to the annotation tensor. This can be used to later on extract only
    valid entries from logits tensor and labels tensor. This function is
    supposed to work with a batch input like [b, w, h] -- where b is a
    batch size, w, h -- are width and height sizes. So the output is
    a tensor which contains indexes of valid entries. This function can
    also work with a single annotation like [w, h] -- the output will
    be (num_valid_eintries, 2).
    
    Parameters
    ----------
    annotation_batch_tensor : Tensor of size (batch_size, width, height)
        Tensor with class labels for each batch
    class_labels : list of ints
        List that contains the numbers that represent classes. Last
        value in the list should represent the number that was used
        for masking out.
        
    Returns
    -------
    valid_labels_indices : Tensor of size (num_valid_eintries, 3).
        Tensor with indices of valid entries
    """
    
    # Last value in the classes list should show
    # which number was used in the annotation to mask out
    # the ambigious regions or regions that should not be
    # used for training.
    # TODO: probably replace class_labels list with some custom object
    mask_out_class_label = class_labels[-1]
    
    # Get binary mask for the pixels that we want to
    # use for training. We do this because some pixels
    # are marked as ambigious and we don't want to use
    # them for trainig to avoid confusing the model
    valid_labels_mask = tf.not_equal(annotation_batch_tensor,
                                        mask_out_class_label)
    
    valid_labels_indices = tf.where(valid_labels_mask)
    
    return tf.to_int32(valid_labels_indices)


def get_valid_logits_and_labels(annotation_batch_tensor,
                                logits_batch_tensor,
                                class_labels):
    """Returns two tensors of size (num_valid_entries, num_classes).
    The function converts annotation batch tensor input of the size
    (batch_size, height, width) into label tensor (batch_size, height,
    width, num_classes) and then selects only valid entries, resulting
    in tensor of the size (num_valid_entries, num_classes). The function
    also returns the tensor with corresponding valid entries in the logits
    tensor. Overall, two tensors of the same sizes are returned and later on
    can be used as an input into tf.softmax_cross_entropy_with_logits() to
    get the cross entropy error for each entry.
    
    Parameters
    ----------
    annotation_batch_tensor : Tensor of size (batch_size, width, height)
        Tensor with class labels for each batch
    logits_batch_tensor : Tensor of size (batch_size, width, height, num_classes)
        Tensor with logits. Usually can be achived after inference of fcn network.
    class_labels : list of ints
        List that contains the numbers that represent classes. Last
        value in the list should represent the number that was used
        for masking out.
        
    Returns
    -------
    (valid_labels_batch_tensor, valid_logits_batch_tensor) : Two Tensors of size (num_valid_eintries, num_classes).
        Tensors that represent valid labels and logits.
    """
    
    
    labels_batch_tensor = get_labels_from_annotation_batch(annotation_batch_tensor=annotation_batch_tensor,
                                                           class_labels=class_labels)
    
    valid_batch_indices = get_valid_entries_indices_from_annotation_batch(annotation_batch_tensor=annotation_batch_tensor,
                                                                          class_labels=class_labels)
    
    valid_labels_batch_tensor = tf.gather_nd(params=labels_batch_tensor, indices=valid_batch_indices)
    
    valid_logits_batch_tensor = tf.gather_nd(params=logits_batch_tensor, indices=valid_batch_indices)
    
    return valid_labels_batch_tensor, valid_logits_batch_tensor