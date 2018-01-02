import tensorflow as tf
from tensorflow.contrib import slim
from nets import resnet_v1
from utils.training import get_valid_logits_and_labels
FLAGS = tf.app.flags.FLAGS

def unpool(inputs,scale):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*scale,  tf.shape(inputs)[2]*scale])


def ResidualConvUnit(inputs,features=256,kernel_size=3):
    net=tf.nn.relu(inputs)
    net=slim.conv2d(net, features, kernel_size)
    net=tf.nn.relu(net)
    net=slim.conv2d(net,features,kernel_size)
    net=tf.add(net,inputs)

    return net

def MultiResolutionFusion(high_inputs=None,low_inputs=None,up0=2,up1=1,n_i=256):

    g0 = unpool(slim.conv2d(high_inputs, n_i, 3), scale=up0)

    if low_inputs is None:
        return g0

    g1=unpool(slim.conv2d(low_inputs,n_i,3),scale=up1)
    return tf.add(g0,g1)

def ChainedResidualPooling(inputs,n_i=256):
    net_relu=tf.nn.relu(inputs)
    net=slim.max_pool2d(net_relu, [5, 5],stride=1,padding='SAME')
    net=slim.conv2d(net,n_i,3)
    return tf.add(net,net_relu)

def RefineBlock(high_inputs=None,low_inputs=None):
    if low_inputs is not None:
        print(high_inputs.shape)
        rcu_high=ResidualConvUnit(high_inputs,features=256)
        rcu_low=ResidualConvUnit(low_inputs,features=256)
        fuse=MultiResolutionFusion(rcu_high,rcu_low,up0=2,up1=1,n_i=256)
        fuse_pooling=ChainedResidualPooling(fuse,n_i=256)
        output=ResidualConvUnit(fuse_pooling,features=256)
        return output
    else:
        rcu_high = ResidualConvUnit(high_inputs, features=256)
        fuse = MultiResolutionFusion(rcu_high, low_inputs=None, up0=1,  n_i=256)
        fuse_pooling = ChainedResidualPooling(fuse, n_i=256)
        output = ResidualConvUnit(fuse_pooling, features=256)
        return output


def model(images, weight_decay=1e-5, is_training=True):
    images = mean_image_subtraction(images)

    with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay)):
        logits, end_points = resnet_v1.resnet_v1_101(images, is_training=is_training, scope='resnet_v1_101')

    with tf.variable_scope('feature_fusion', values=[end_points.values]):
        batch_norm_params = {
        'decay': 0.997,
        'epsilon': 1e-5,
        'scale': True,
        'is_training': is_training
        }
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
            f = [end_points['pool5'], end_points['pool4'],
                 end_points['pool3'], end_points['pool2']]
            for i in range(4):
                print('Shape of f_{} {}'.format(i, f[i].shape))

            g = [None, None, None, None]
            h = [None, None, None, None]

            for i in range(4):
                h[i]=slim.conv2d(f[i], 256, 1)
            for i in range(4):
                print('Shape of h_{} {}'.format(i, h[i].shape))

            g[0]=RefineBlock(h[0])
            g[1]=RefineBlock(g[0],h[1])
            g[2]=RefineBlock(g[1],h[2])
            g[3]=RefineBlock(g[2],h[3])
            g[3]=unpool(g[3],scale=4)
            F_score = slim.conv2d(g[3], 21, 1, activation_fn=tf.nn.relu, normalizer_fn=None)

    return F_score


def mean_image_subtraction(images, means=[123.68, 116.78, 103.94]):
    num_channels = images.get_shape().as_list()[-1]
    if len(means) != num_channels:
      raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)

def loss(annotation_batch,upsampled_logits_batch,class_labels):
    valid_labels_batch_tensor, valid_logits_batch_tensor = get_valid_logits_and_labels(
        annotation_batch_tensor=annotation_batch,
        logits_batch_tensor=upsampled_logits_batch,
        class_labels=class_labels)

    cross_entropies = tf.nn.softmax_cross_entropy_with_logits(logits=valid_logits_batch_tensor,
                                                              labels=valid_labels_batch_tensor)

    cross_entropy_sum = tf.reduce_mean(cross_entropies)
    tf.summary.scalar('cross_entropy_loss', cross_entropy_sum)

    return cross_entropy_sum

