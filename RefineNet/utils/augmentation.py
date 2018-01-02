import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

slim = tf.contrib.slim


def apply_with_random_selector(x, func, num_cases):
  """Computes func(x, sel), with sel sampled from [0...num_cases-1].
  Args:
    x: input Tensor.
    func: Python function to apply.
    num_cases: Python int32, number of cases to sample sel from.
  Returns:
    The result of func(x, sel), where func receives the value of the
    selector as a python integer, but sel is sampled dynamically.
  """
  sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
  # Pass the real x only to one of the func calls.
  return control_flow_ops.merge([
      func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
      for case in range(num_cases)])[0]


def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
  """Distort the color of a Tensor image.
  Each color distortion is non-commutative and thus ordering of the color ops
  matters. Ideally we would randomly permute the ordering of the color ops.
  Rather then adding that level of complication, we select a distinct ordering
  of color ops for each preprocessing thread.
  Args:
    image: 3-D Tensor containing single image in [0, 1].
    color_ordering: Python int, a type of distortion (valid values: 0-3).
    fast_mode: Avoids slower ops (random_hue and random_contrast)
    scope: Optional scope for name_scope.
  Returns:
    3-D Tensor color-distorted image on range [0, 1]
  Raises:
    ValueError: if color_ordering not in [0, 3]
  """
  with tf.name_scope(scope, 'distort_color', [image]):
    if fast_mode:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      else:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    else:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
      elif color_ordering == 2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      elif color_ordering == 3:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
      else:
        raise ValueError('color_ordering must be in [0, 3]')

    # The random_* ops do not necessarily clamp.
    return tf.clip_by_value(image, 0.0, 1.0)


def flip_randomly_left_right_image_with_annotation(image_tensor, annotation_tensor):
    """Accepts image tensor and annotation tensor and returns randomly flipped tensors of both.
    The function performs random flip of image and annotation tensors with probability of 1/2
    The flip is performed or not performed for image and annotation consistently, so that
    annotation matches the image.
    
    Parameters
    ----------
    image_tensor : Tensor of size (width, height, 3)
        Tensor with image
    annotation_tensor : Tensor of size (width, height, 1)
        Tensor with annotation
        
    Returns
    -------
    randomly_flipped_img : Tensor of size (width, height, 3) of type tf.float.
        Randomly flipped image tensor
    randomly_flipped_annotation : Tensor of size (width, height, 1)
        Randomly flipped annotation tensor
        
    """
    
    # Random variable: two possible outcomes (0 or 1)
    # with a 1 in 2 chance
    random_var = tf.random_uniform(maxval=2, dtype=tf.int32, shape=[])


    randomly_flipped_img = control_flow_ops.cond(pred=tf.equal(random_var, 0),
                                                 fn1=lambda: tf.image.flip_left_right(image_tensor),
                                                 fn2=lambda: image_tensor)

    randomly_flipped_annotation = control_flow_ops.cond(pred=tf.equal(random_var, 0),
                                                        fn1=lambda: tf.image.flip_left_right(annotation_tensor),
                                                        fn2=lambda: annotation_tensor)
    
    return randomly_flipped_img, randomly_flipped_annotation


def distort_randomly_image_color(image_tensor, fast_mode=False):
    """Accepts image tensor of (width, height, 3) and returns color distorted image.
    The function performs random brightness, saturation, hue, contrast change as it is performed
    for inception model training in TF-Slim (you can find the link below in comments). All the
    parameters of random variables were originally preserved. There are two regimes for the function
    to work: fast and slow. Slow one performs only saturation and brightness random change is performed.
    
    Parameters
    ----------
    image_tensor : Tensor of size (width, height, 3) of tf.int32 or tf.float
        Tensor with image with range [0,255]
    fast_mode : boolean
        Boolean value representing whether to use fast or slow mode
        
    Returns
    -------
    img_float_distorted_original_range : Tensor of size (width, height, 3) of type tf.float.
        Image Tensor with distorted color in [0,255] intensity range
    """
    
    # Make the range to be in [0,1]
    img_float_zero_one_range = tf.to_float(image_tensor) / 255
    
    # Randomly distort the color of image. There are 4 ways to do it.
    # Credit: TF-Slim
    # https://github.com/tensorflow/models/blob/master/slim/preprocessing/inception_preprocessing.py#L224
    # Most probably the inception models were trainined using this color augmentation:
    # https://github.com/tensorflow/models/tree/master/slim#pre-trained-models
    distorted_image = apply_with_random_selector(img_float_zero_one_range,
                                                 lambda x, ordering: distort_color(x, ordering, fast_mode=fast_mode),
                                                 num_cases=4)
    
    img_float_distorted_original_range = distorted_image * 255
    
    return img_float_distorted_original_range
    
    


def scale_randomly_image_with_annotation_with_fixed_size_output(img_tensor,
                                                                annotation_tensor,
                                                                output_shape,
                                                                min_relative_random_scale_change=0.9,
                                                                max_realtive_random_scale_change=1.1,
                                                                mask_out_number=255):
    """Returns tensor of a size (output_shape, output_shape, depth) and (output_shape, output_shape, 1).
    The function returns tensor that is of a size (output_shape, output_shape, depth)
    which is randomly scaled by a factor that is sampled from a uniform distribution
    between values [min_relative_random_scale_change, max_realtive_random_scale_change] multiplied
    by the factor that is needed to scale image to the output_shape. When the rescaled image
    doesn't fit into the [output_shape] size, the image is either padded or cropped. Also, the
    function returns scaled annotation tensor of the size (output_shape, output_shape, 1). Both,
    the image tensor and the annotation tensor are scaled using nearest neighbour interpolation.
    This was done to preserve the annotation labels. Be careful when specifying the big sample
    space for the random variable -- aliasing effects can appear. When scaling, this function
    preserves the aspect ratio of the original image. When performing all of those manipulations
    there will be some regions in the output image with blank regions -- the function masks out
    those regions in the annotation using mask_out_number. Overall, the function performs the
    rescaling neccessary to get image of output_shape, adds random scale jitter, preserves
    scale ratio, masks out unneccassary regions that appear.
    
    Parameters
    ----------
    img_tensor : Tensor of size (width, height, depth)
        Tensor with image
    annotation_tensor : Tensor of size (width, height, 1)
        Tensor with respective annotation
    output_shape : Tensor or list [int, int]
        Tensor of list representing desired output shape
    min_relative_random_scale_change : float
        Lower bound for uniform distribution to sample from
        when getting random scaling jitter
    max_realtive_random_scale_change : float
        Upper bound for uniform distribution to sample from
        when getting random scaling jitter
    mask_out_number : int
        Number representing the mask out value.
        
    Returns
    -------
    cropped_padded_img : Tensor of size (output_shape[0], output_shape[1], 3).
        Image Tensor that was randomly scaled
    cropped_padded_annotation : Tensor of size (output_shape[0], output_shape[1], 1)
        Respective annotation Tensor that was randomly scaled with the same parameters
    """
    
    # tf.image.resize_nearest_neighbor needs
    # first dimension to represent the batch number
    img_batched = tf.expand_dims(img_tensor, 0)
    annotation_batched = tf.expand_dims(annotation_tensor, 0)

    # Convert to int_32 to be able to differentiate
    # between zeros that was used for padding and
    # zeros that represent a particular semantic class
    annotation_batched = tf.to_int32(annotation_batched)

    # Get height and width tensors
    input_shape = tf.shape(img_batched)[1:3]

    input_shape_float = tf.to_float(input_shape)

    scales = output_shape / input_shape_float

    rand_var = tf.random_uniform(shape=[1],
                                 minval=min_relative_random_scale_change,
                                 maxval=max_realtive_random_scale_change)

    final_scale = tf.reduce_min(scales) * rand_var

    scaled_input_shape = tf.to_int32(tf.round(input_shape_float * final_scale))
    
    # Resize the image and annotation using nearest neighbour
    # Be careful -- may cause aliasing.
    
    # TODO: try bilinear resampling for image only
    resized_img = tf.image.resize_nearest_neighbor( img_batched, scaled_input_shape )
    resized_annotation = tf.image.resize_nearest_neighbor( annotation_batched, scaled_input_shape )

    resized_img = tf.squeeze(resized_img, axis=0)
    resized_annotation = tf.squeeze(resized_annotation, axis=0)

    # Shift all the classes by one -- to be able to differentiate
    # between zeros representing padded values and zeros representing
    # a particular semantic class.
    annotation_shifted_classes = resized_annotation + 1

    cropped_padded_img = tf.image.resize_image_with_crop_or_pad( resized_img, output_shape[0], output_shape[1] )

    cropped_padded_annotation = tf.image.resize_image_with_crop_or_pad(annotation_shifted_classes,
                                                                       output_shape[0],
                                                                       output_shape[1])
    
    # TODO: accept the classes lut instead of mask out
    # value as an argument
    annotation_additional_mask_out = tf.to_int32(tf.equal(cropped_padded_annotation, 0)) * (mask_out_number+1)

    cropped_padded_annotation = cropped_padded_annotation + annotation_additional_mask_out - 1
    
    return cropped_padded_img, cropped_padded_annotation