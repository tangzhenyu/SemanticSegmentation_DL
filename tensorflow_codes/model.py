from network import Network
from network import PReLU, spatial_dropout, max_unpool
from tools import *
import tensorflow as tf
import os
import cv2

class FCN8s(Network):
    def __init__(self, is_training=False, num_classes=151, input_size=[384, 384]):
        self.input_size = input_size
    
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, None, 3])
        self.img_tf, self.shape = preprocess(self.x, self.input_size, 'fcn-8s')
        
        super().__init__({'data': self.img_tf}, num_classes, is_training)

    def setup(self, is_training, num_classes):
        (self.feed('data')
             .zero_padding(paddings=100, name='padding1')
             .conv(3, 3, 64, 1, 1, name='conv1_1')
             .zero_padding(paddings=1, name='padding2')
             .conv(3, 3, 64, 1, 1, name='conv1_2')
             .max_pool(2, 2, 2, 2, name='pool1')
             .zero_padding(paddings=1, name='padding3')
             .conv(3, 3, 128, 1, 1, name='conv2_1')
             .zero_padding(paddings=1, name='padding4')
             .conv(3, 3, 128, 1, 1, name='conv2_2')
             .zero_padding(paddings=1, name='padding5')
             .max_pool(2, 2, 2, 2, name='pool2')
             .zero_padding(paddings=1, name='padding6')
             .conv(3, 3, 256, 1, 1, name='conv3_1')
             .zero_padding(paddings=1, name='padding7')
             .conv(3, 3, 256, 1, 1, name='conv3_2')
             .zero_padding(paddings=1, name='padding8')
             .conv(3, 3, 256, 1, 1, name='conv3_3')
             .max_pool(2, 2, 2, 2, name='pool3')
             .scale(0.00001, name='scale_pool3')
             .conv(1, 1, num_classes, 1, 1, name='score_pool3'))

        (self.feed('pool3')
             .zero_padding(paddings=1, name='padding9')
             .conv(3, 3, 512, 1, 1, name='conv4_1')
             .zero_padding(paddings=1, name='padding10')
             .conv(3, 3, 512, 1, 1, name='conv4_2')
             .zero_padding(paddings=1, name='padding11')
             .conv(3, 3, 512, 1, 1, name='conv4_3')
             .zero_padding(paddings=1, name='padding12')
             .max_pool(2, 2, 2, 2, name='pool4')
             .scale(0.01, name='scale_pool4')
             .conv(1, 1, num_classes, 1, 1, name='score_pool4'))

        (self.feed('pool4')
             .zero_padding(paddings=1, name='padding13')
             .conv(3, 3, 512, 1, 1, name='conv5_1')
             .zero_padding(paddings=1, name='padding14')
             .conv(3, 3, 512, 1, 1, name='conv5_2')
             .zero_padding(paddings=1, name='padding15')
             .conv(3, 3, 512, 1, 1, name='conv5_3')
             .zero_padding(paddings=1, name='padding16')
             .max_pool(2, 2, 2, 2, name='pool5')
             .conv(7, 7, 4096, 1, 1, name='fc6')
             .conv(1, 1, 4096, 1, 1, name='fc7')
             .conv(1, 1, num_classes, 1, 1, name='score_fr')
             .deconv(4, 4, num_classes, 2, 2, name='upscore2'))

        (self.feed('upscore2', 'score_pool4')
             .crop(5, name='score_pool4c'))

        (self.feed('upscore2', 'score_pool4c')
             .add(name='fuse_pool4')
             .deconv(4, 4, num_classes, 2, 2, name='upscore_pool4'))

        (self.feed('upscore_pool4', 'score_pool3')
             .crop(9, name='score_pool3c'))

        (self.feed('upscore_pool4', 'score_pool3c')
             .add(name='fuse_pool3')
             .deconv(16, 16, num_classes, 8, 8, name='upscore8'))

        (self.feed('data', 'upscore8')
             .crop(31, name='score'))

        score = self.layers['score']
        score = tf.image.resize_bilinear(score, size=self.shape[0:2], align_corners=True)
        score = tf.argmax(score, axis=3)
        self.pred = decode_labels(score, self.shape[0:2], num_classes)

    def read_input(self, img_path):
        self.img, self.img_name = load_img(img_path)

    def forward(self, sess):
        return sess.run(self.pred, feed_dict={self.x: self.img})

class PSPNet50(Network):
    def __init__(self, is_training=False, num_classes=150, input_size=[473, 473]):
        self.input_size = input_size
    
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, None, 3])
        self.img_tf, self.n_h, self.n_w, self.o_shape = preprocess(self.x, self.input_size, 'pspnet50')
        # return new height, new width, original input shape

        super().__init__({'data': self.img_tf}, num_classes, is_training)

    def setup(self, is_training, num_classes):
        (self.feed('data')
             .conv(3, 3, 64, 2, 2, biased=False, relu=False, padding='SAME', name='conv1_1_3x3_s2')
             .batch_normalization(relu=False, name='conv1_1_3x3_s2_bn')
             .relu(name='conv1_1_3x3_s2_bn_relu')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, padding='SAME', name='conv1_2_3x3')
             .batch_normalization(relu=True, name='conv1_2_3x3_bn')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, padding='SAME', name='conv1_3_3x3')
             .batch_normalization(relu=True, name='conv1_3_3x3_bn')
             .max_pool(3, 3, 2, 2, padding='SAME', name='pool1_3x3_s2')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv2_1_1x1_proj')
             .batch_normalization(relu=False, name='conv2_1_1x1_proj_bn'))

        (self.feed('pool1_3x3_s2')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='conv2_1_1x1_reduce')
             .batch_normalization(relu=True, name='conv2_1_1x1_reduce_bn')
             .zero_padding(paddings=1, name='padding1')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='conv2_1_3x3')
             .batch_normalization(relu=True, name='conv2_1_3x3_bn')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv2_1_1x1_increase')
             .batch_normalization(relu=False, name='conv2_1_1x1_increase_bn'))

        (self.feed('conv2_1_1x1_proj_bn',
                   'conv2_1_1x1_increase_bn')
             .add(name='conv2_1')
             .relu(name='conv2_1/relu')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='conv2_2_1x1_reduce')
             .batch_normalization(relu=True, name='conv2_2_1x1_reduce_bn')
             .zero_padding(paddings=1, name='padding2')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='conv2_2_3x3')
             .batch_normalization(relu=True, name='conv2_2_3x3_bn')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv2_2_1x1_increase')
             .batch_normalization(relu=False, name='conv2_2_1x1_increase_bn'))

        (self.feed('conv2_1/relu',
                   'conv2_2_1x1_increase_bn')
             .add(name='conv2_2')
             .relu(name='conv2_2/relu')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='conv2_3_1x1_reduce')
             .batch_normalization(relu=True, name='conv2_3_1x1_reduce_bn')
             .zero_padding(paddings=1, name='padding3')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='conv2_3_3x3')
             .batch_normalization(relu=True, name='conv2_3_3x3_bn')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv2_3_1x1_increase')
             .batch_normalization(relu=False, name='conv2_3_1x1_increase_bn'))

        (self.feed('conv2_2/relu',
                   'conv2_3_1x1_increase_bn')
             .add(name='conv2_3')
             .relu(name='conv2_3/relu')
             .conv(1, 1, 512, 2, 2, biased=False, relu=False, name='conv3_1_1x1_proj')
             .batch_normalization(relu=False, name='conv3_1_1x1_proj_bn'))

        (self.feed('conv2_3/relu')
             .conv(1, 1, 128, 2, 2, biased=False, relu=False, name='conv3_1_1x1_reduce')
             .batch_normalization(relu=True, name='conv3_1_1x1_reduce_bn')
             .zero_padding(paddings=1, name='padding4')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='conv3_1_3x3')
             .batch_normalization(relu=True, name='conv3_1_3x3_bn')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv3_1_1x1_increase')
             .batch_normalization(relu=False, name='conv3_1_1x1_increase_bn'))

        (self.feed('conv3_1_1x1_proj_bn',
                   'conv3_1_1x1_increase_bn')
             .add(name='conv3_1')
             .relu(name='conv3_1/relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv3_2_1x1_reduce')
             .batch_normalization(relu=True, name='conv3_2_1x1_reduce_bn')
             .zero_padding(paddings=1, name='padding5')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='conv3_2_3x3')
             .batch_normalization(relu=True, name='conv3_2_3x3_bn')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv3_2_1x1_increase')
             .batch_normalization(relu=False, name='conv3_2_1x1_increase_bn'))

        (self.feed('conv3_1/relu',
                   'conv3_2_1x1_increase_bn')
             .add(name='conv3_2')
             .relu(name='conv3_2/relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv3_3_1x1_reduce')
             .batch_normalization(relu=True, name='conv3_3_1x1_reduce_bn')
             .zero_padding(paddings=1, name='padding6')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='conv3_3_3x3')
             .batch_normalization(relu=True, name='conv3_3_3x3_bn')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv3_3_1x1_increase')
             .batch_normalization(relu=False, name='conv3_3_1x1_increase_bn'))

        (self.feed('conv3_2/relu',
                   'conv3_3_1x1_increase_bn')
             .add(name='conv3_3')
             .relu(name='conv3_3/relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv3_4_1x1_reduce')
             .batch_normalization(relu=True, name='conv3_4_1x1_reduce_bn')
             .zero_padding(paddings=1, name='padding7')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='conv3_4_3x3')
             .batch_normalization(relu=True, name='conv3_4_3x3_bn')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv3_4_1x1_increase')
             .batch_normalization(relu=False, name='conv3_4_1x1_increase_bn'))

        (self.feed('conv3_3/relu',
                   'conv3_4_1x1_increase_bn')
             .add(name='conv3_4')
             .relu(name='conv3_4/relu')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_1_1x1_proj')
             .batch_normalization(relu=False, name='conv4_1_1x1_proj_bn'))

        (self.feed('conv3_4/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_1_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_1_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding8')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_1_3x3')
             .batch_normalization(relu=True, name='conv4_1_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_1_1x1_increase')
             .batch_normalization(relu=False, name='conv4_1_1x1_increase_bn'))

        (self.feed('conv4_1_1x1_proj_bn',
                   'conv4_1_1x1_increase_bn')
             .add(name='conv4_1')
             .relu(name='conv4_1/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_2_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_2_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding9')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_2_3x3')
             .batch_normalization(relu=True, name='conv4_2_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_2_1x1_increase')
             .batch_normalization(relu=False, name='conv4_2_1x1_increase_bn'))

        (self.feed('conv4_1/relu',
                   'conv4_2_1x1_increase_bn')
             .add(name='conv4_2')
             .relu(name='conv4_2/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_3_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_3_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding10')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_3_3x3')
             .batch_normalization(relu=True, name='conv4_3_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_3_1x1_increase')
             .batch_normalization(relu=False, name='conv4_3_1x1_increase_bn'))

        (self.feed('conv4_2/relu',
                   'conv4_3_1x1_increase_bn')
             .add(name='conv4_3')
             .relu(name='conv4_3/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_4_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_4_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding11')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_4_3x3')
             .batch_normalization(relu=True, name='conv4_4_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_4_1x1_increase')
             .batch_normalization(relu=False, name='conv4_4_1x1_increase_bn'))

        (self.feed('conv4_3/relu',
                   'conv4_4_1x1_increase_bn')
             .add(name='conv4_4')
             .relu(name='conv4_4/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_5_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_5_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding12')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_5_3x3')
             .batch_normalization(relu=True, name='conv4_5_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_5_1x1_increase')
             .batch_normalization(relu=False, name='conv4_5_1x1_increase_bn'))

        (self.feed('conv4_4/relu',
                   'conv4_5_1x1_increase_bn')
             .add(name='conv4_5')
             .relu(name='conv4_5/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_6_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_6_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding13')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_6_3x3')
             .batch_normalization(relu=True, name='conv4_6_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_6_1x1_increase')
             .batch_normalization(relu=False, name='conv4_6_1x1_increase_bn'))

        (self.feed('conv4_5/relu',
                   'conv4_6_1x1_increase_bn')
             .add(name='conv4_6')
             .relu(name='conv4_6/relu')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='conv5_1_1x1_proj')
             .batch_normalization(relu=False, name='conv5_1_1x1_proj_bn'))

        (self.feed('conv4_6/relu')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv5_1_1x1_reduce')
             .batch_normalization(relu=True, name='conv5_1_1x1_reduce_bn')
             .zero_padding(paddings=4, name='padding31')
             .atrous_conv(3, 3, 512, 4, biased=False, relu=False, name='conv5_1_3x3')
             .batch_normalization(relu=True, name='conv5_1_3x3_bn')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='conv5_1_1x1_increase')
             .batch_normalization(relu=False, name='conv5_1_1x1_increase_bn'))

        (self.feed('conv5_1_1x1_proj_bn',
                   'conv5_1_1x1_increase_bn')
             .add(name='conv5_1')
             .relu(name='conv5_1/relu')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv5_2_1x1_reduce')
             .batch_normalization(relu=True, name='conv5_2_1x1_reduce_bn')
             .zero_padding(paddings=4, name='padding32')
             .atrous_conv(3, 3, 512, 4, biased=False, relu=False, name='conv5_2_3x3')
             .batch_normalization(relu=True, name='conv5_2_3x3_bn')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='conv5_2_1x1_increase')
             .batch_normalization(relu=False, name='conv5_2_1x1_increase_bn'))

        (self.feed('conv5_1/relu',
                   'conv5_2_1x1_increase_bn')
             .add(name='conv5_2')
             .relu(name='conv5_2/relu')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv5_3_1x1_reduce')
             .batch_normalization(relu=True, name='conv5_3_1x1_reduce_bn')
             .zero_padding(paddings=4, name='padding33')
             .atrous_conv(3, 3, 512, 4, biased=False, relu=False, name='conv5_3_3x3')
             .batch_normalization(relu=True, name='conv5_3_3x3_bn')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='conv5_3_1x1_increase')
             .batch_normalization(relu=False, name='conv5_3_1x1_increase_bn'))

        (self.feed('conv5_2/relu',
                   'conv5_3_1x1_increase_bn')
             .add(name='conv5_3')
             .relu(name='conv5_3/relu'))

        conv5_3 = self.layers['conv5_3/relu']
        shape = tf.shape(conv5_3)[1:3]

        (self.feed('conv5_3/relu')
             .avg_pool(60, 60, 60, 60, name='conv5_3_pool1')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv5_3_pool1_conv')
             .batch_normalization(relu=True, name='conv5_3_pool1_conv_bn')
             .resize_bilinear(shape, name='conv5_3_pool1_interp'))

        (self.feed('conv5_3/relu')
             .avg_pool(30, 30, 30, 30, name='conv5_3_pool2')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv5_3_pool2_conv')
             .batch_normalization(relu=True, name='conv5_3_pool2_conv_bn')
             .resize_bilinear(shape, name='conv5_3_pool2_interp'))

        (self.feed('conv5_3/relu')
             .avg_pool(20, 20, 20, 20, name='conv5_3_pool3')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv5_3_pool3_conv')
             .batch_normalization(relu=True, name='conv5_3_pool3_conv_bn')
             .resize_bilinear(shape, name='conv5_3_pool3_interp'))

        (self.feed('conv5_3/relu')
             .avg_pool(10, 10, 10, 10, name='conv5_3_pool6')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv5_3_pool6_conv')
             .batch_normalization(relu=True, name='conv5_3_pool6_conv_bn')
             .resize_bilinear(shape, name='conv5_3_pool6_interp'))

        (self.feed('conv5_3/relu',
                   'conv5_3_pool6_interp',
                   'conv5_3_pool3_interp',
                   'conv5_3_pool2_interp',
                   'conv5_3_pool1_interp')
             .concat(axis=-1, name='conv5_3_concat')
             .conv(3, 3, 512, 1, 1, biased=False, relu=False, padding='SAME', name='conv5_4')
             .batch_normalization(relu=True, name='conv5_4_bn')
             .conv(1, 1, num_classes, 1, 1, biased=True, relu=False, name='conv6'))

        raw_output = self.layers['conv6']
        raw_output_up = tf.image.resize_bilinear(raw_output, size=[self.n_h, self.n_w], align_corners=True)
        raw_output_up = tf.image.crop_to_bounding_box(raw_output_up, 0, 0, self.o_shape[0], self.o_shape[1])
        raw_output_up = tf.argmax(raw_output_up, axis=3)
        self.pred = decode_labels(raw_output_up, self.o_shape, num_classes)

    def read_input(self, img_path):
        self.img, self.img_name = load_img(img_path)

    def forward(self, sess):
        return sess.run(self.pred, feed_dict={self.x: self.img})

class ENet(object):
    def __init__(self, img_height=512, img_width=1024, batch_size=1):
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.img_mean = np.array((103.939, 116.779, 123.68), dtype=np.float32)
        self.no_of_classes = 20
        self.batch_imgs = np.zeros((batch_size, img_height, img_width, 3), dtype=np.float32)

        self.wd = 2e-4

        # add placeholders to the comp. graph:
        self.add_placeholders()

        # define the forward pass, compute logits and add to the comp. graph:
        self.add_logits()

    def read_input(self, img_path):
        self.img, self.img_name = load_img(img_path)
        self.img = cv2.resize(self.img, (self.img_width, self.img_height))
        self.img = self.img - self.img_mean
        self.batch_imgs[0] = self.img

    def forward(self, sess):
        batch_feed_dict = self.create_feed_dict(imgs_batch=self.batch_imgs,
                    early_drop_prob=0.0, late_drop_prob=0.0)
        pred = sess.run(self.logits, feed_dict=batch_feed_dict)

        return pred

    def load(self, model_path, sess):
        saver = tf.train.Saver(tf.trainable_variables(), write_version=tf.train.SaverDef.V2)
        saver.restore(sess, model_path)
        print('restore from: ', model_path)

    """
    def read_input(self):
        ---------- nothing ---------

    def forward(self, sess, img):
        # suppose img shape = h x w x 3
        img = cv2.resize(img, (self.img_width, self.img_height))
        img = img - self.img_mean
        self.batch_imgs[0] = img
        
        batch_feed_dict = self.create_feed_dict(imgs_batch=self.batch_imgs,
                    early_drop_prob=0.0, late_drop_prob=0.0)
        pred = sess.run(self.logits, feed_dict=batch_feed_dict)

        return pred
    """
    def add_placeholders(self):
        self.imgs_ph = tf.placeholder(tf.float32,
                    shape=[self.batch_size, self.img_height, self.img_width, 3],
                    name="imgs_ph")

        self.onehot_labels_ph = tf.placeholder(tf.float32,
                    shape=[self.batch_size, self.img_height, self.img_width, self.no_of_classes],
                    name="onehot_labels_ph")

        # dropout probability in the early layers of the network:
        self.early_drop_prob_ph = tf.placeholder(tf.float32, name="early_drop_prob_ph")

        # dropout probability in the later layers of the network:
        self.late_drop_prob_ph = tf.placeholder(tf.float32, name="late_drop_prob_ph")

    def create_feed_dict(self, imgs_batch, early_drop_prob, late_drop_prob, onehot_labels_batch=None):
        # return a feed_dict mapping the placeholders to the actual input data:
        feed_dict = {}
        feed_dict[self.imgs_ph] = imgs_batch
        feed_dict[self.early_drop_prob_ph] = early_drop_prob
        feed_dict[self.late_drop_prob_ph] = late_drop_prob
        if onehot_labels_batch is not None:
            # only add the labels data if it's specified (during inference, we
            # won't have any labels):
            feed_dict[self.onehot_labels_ph] = onehot_labels_batch

        return feed_dict

    def add_logits(self):
        # encoder:
        # # initial block:
        network = self.initial_block(x=self.imgs_ph, scope="inital")
        #print(network.get_shape().as_list())


        # # layer 1:
        # # # save the input shape to use in max_unpool in the decoder:
        inputs_shape_1 = network.get_shape().as_list()
        network, pooling_indices_1 = self.encoder_bottleneck_regular(x=network,
                    output_depth=64, drop_prob=self.early_drop_prob_ph,
                    scope="bottleneck_1_0", downsampling=True)

        network = self.encoder_bottleneck_regular(x=network, output_depth=64,
                    drop_prob=self.early_drop_prob_ph, scope="bottleneck_1_1")
        network = self.encoder_bottleneck_regular(x=network, output_depth=64,
                    drop_prob=self.early_drop_prob_ph, scope="bottleneck_1_2")
        network = self.encoder_bottleneck_regular(x=network, output_depth=64,
                    drop_prob=self.early_drop_prob_ph, scope="bottleneck_1_3")
        network = self.encoder_bottleneck_regular(x=network, output_depth=64,
                    drop_prob=self.early_drop_prob_ph, scope="bottleneck_1_4")

        # # layer 2:
        # # # save the input shape to use in max_unpool in the decoder:
        inputs_shape_2 = network.get_shape().as_list()
        network, pooling_indices_2 = self.encoder_bottleneck_regular(x=network,
                    output_depth=128, drop_prob=self.late_drop_prob_ph,
                    scope="bottleneck_2_0", downsampling=True)
        network = self.encoder_bottleneck_regular(x=network, output_depth=128,
                        drop_prob=self.late_drop_prob_ph, scope="bottleneck_2_1")
        network = self.encoder_bottleneck_dilated(x=network, output_depth=128,
                    drop_prob=self.late_drop_prob_ph, scope="bottleneck_2_2",
                    dilation_rate=2)
        network = self.encoder_bottleneck_asymmetric(x=network, output_depth=128,
                    drop_prob=self.late_drop_prob_ph, scope="bottleneck_2_3")
        network = self.encoder_bottleneck_dilated(x=network, output_depth=128,
                    drop_prob=self.late_drop_prob_ph, scope="bottleneck_2_4",
                    dilation_rate=4)
        network = self.encoder_bottleneck_regular(x=network, output_depth=128,
                    drop_prob=self.late_drop_prob_ph, scope="bottleneck_2_5")
        network = self.encoder_bottleneck_dilated(x=network, output_depth=128,
                    drop_prob=self.late_drop_prob_ph, scope="bottleneck_2_6",
                    dilation_rate=8)
        network = self.encoder_bottleneck_asymmetric(x=network, output_depth=128,
                    drop_prob=self.late_drop_prob_ph, scope="bottleneck_2_7")
        network = self.encoder_bottleneck_dilated(x=network, output_depth=128,
                    drop_prob=self.late_drop_prob_ph, scope="bottleneck_2_8",
                    dilation_rate=16)

        # layer 3:
        network = self.encoder_bottleneck_regular(x=network, output_depth=128,
                    drop_prob=self.late_drop_prob_ph, scope="bottleneck_3_1")
        network = self.encoder_bottleneck_dilated(x=network, output_depth=128,
                    drop_prob=self.late_drop_prob_ph, scope="bottleneck_3_2",
                    dilation_rate=2)
        network = self.encoder_bottleneck_asymmetric(x=network, output_depth=128,
                    drop_prob=self.late_drop_prob_ph, scope="bottleneck_3_3")
        network = self.encoder_bottleneck_dilated(x=network, output_depth=128,
                    drop_prob=self.late_drop_prob_ph, scope="bottleneck_3_4",
                    dilation_rate=4)
        network = self.encoder_bottleneck_regular(x=network, output_depth=128,
                    drop_prob=self.late_drop_prob_ph, scope="bottleneck_3_5")
        network = self.encoder_bottleneck_dilated(x=network, output_depth=128,
                    drop_prob=self.late_drop_prob_ph, scope="bottleneck_3_6",
                    dilation_rate=8)
        network = self.encoder_bottleneck_asymmetric(x=network, output_depth=128,
                    drop_prob=self.late_drop_prob_ph, scope="bottleneck_3_7")
        network = self.encoder_bottleneck_dilated(x=network, output_depth=128,
                    drop_prob=self.late_drop_prob_ph, scope="bottleneck_3_8",
                    dilation_rate=16)

        # decoder:
        # # layer 4:
        network = self.decoder_bottleneck(x=network, output_depth=64,
                    scope="bottleneck_4_0", upsampling=True,
                    pooling_indices=pooling_indices_2, output_shape=inputs_shape_2)
        network = self.decoder_bottleneck(x=network, output_depth=64,
                    scope="bottleneck_4_1")
        network = self.decoder_bottleneck(x=network, output_depth=64,
                    scope="bottleneck_4_2")

        # # layer 5:
        network = self.decoder_bottleneck(x=network, output_depth=16,
                    scope="bottleneck_5_0", upsampling=True,
                    pooling_indices=pooling_indices_1, output_shape=inputs_shape_1)
        network = self.decoder_bottleneck(x=network, output_depth=16,
                    scope="bottleneck_5_1")

        # fullconv:
        network = tf.contrib.slim.conv2d_transpose(network, self.no_of_classes,
                    [2, 2], stride=2, scope="fullconv", padding="SAME")

        network = tf.argmax(network, axis=3)
        pred = decode_labels(network, [512, 1024], 20)

        self.logits = pred

    def initial_block(self, x, scope):
        # convolution branch:
        W_conv = self.get_variable_weight_decay(scope + "/W",
                    shape=[3, 3, 3, 13], # ([filter_height, filter_width, in_depth, out_depth])
                    initializer=tf.contrib.layers.xavier_initializer(),
                    loss_category="encoder_wd_losses")
        b_conv = self.get_variable_weight_decay(scope + "/b", shape=[13], # ([out_depth])
                    initializer=tf.constant_initializer(0),
                    loss_category="encoder_wd_losses")
        conv_branch = tf.nn.conv2d(x, W_conv, strides=[1, 2, 2, 1],
                    padding="SAME") + b_conv

        # max pooling branch:
        pool_branch = tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding="VALID")

        # concatenate the branches:
        concat = tf.concat([conv_branch, pool_branch], axis=3) # (3: the depth axis)

        # apply batch normalization and PReLU:
        output = tf.contrib.slim.batch_norm(concat)
        output = PReLU(output, scope=scope)

        return output

    def encoder_bottleneck_regular(self, x, output_depth, drop_prob, scope,
                proj_ratio=4, downsampling=False):
        input_shape = x.get_shape().as_list()
        input_depth = input_shape[3]

        internal_depth = int(output_depth/proj_ratio)

        # convolution branch:
        conv_branch = x

        # # 1x1 projection:
        if downsampling:
            W_conv = self.get_variable_weight_decay(scope + "/W_proj",
                        shape=[2, 2, input_depth, internal_depth], # ([filter_height, filter_width, in_depth, out_depth])
                        initializer=tf.contrib.layers.xavier_initializer(),
                        loss_category="encoder_wd_losses")
            conv_branch = tf.nn.conv2d(conv_branch, W_conv, strides=[1, 2, 2, 1],
                        padding="VALID") # NOTE! no bias terms
        else:
            W_proj = self.get_variable_weight_decay(scope + "/W_proj",
                        shape=[1, 1, input_depth, internal_depth], # ([filter_height, filter_width, in_depth, out_depth])
                        initializer=tf.contrib.layers.xavier_initializer(),
                        loss_category="encoder_wd_losses")
            conv_branch = tf.nn.conv2d(conv_branch, W_proj, strides=[1, 1, 1, 1],
                        padding="VALID") # NOTE! no bias terms
        # # # batch norm and PReLU:
        conv_branch = tf.contrib.slim.batch_norm(conv_branch)
        conv_branch = PReLU(conv_branch, scope=scope + "/proj")

        # # conv:
        W_conv = self.get_variable_weight_decay(scope + "/W_conv",
                    shape=[3, 3, internal_depth, internal_depth], # ([filter_height, filter_width, in_depth, out_depth])
                    initializer=tf.contrib.layers.xavier_initializer(),
                    loss_category="encoder_wd_losses")
        b_conv = self.get_variable_weight_decay(scope + "/b_conv", shape=[internal_depth], # ([out_depth])
                    initializer=tf.constant_initializer(0),
                    loss_category="encoder_wd_losses")
        conv_branch = tf.nn.conv2d(conv_branch, W_conv, strides=[1, 1, 1, 1],
                    padding="SAME") + b_conv
        # # # batch norm and PReLU:
        conv_branch = tf.contrib.slim.batch_norm(conv_branch)
        conv_branch = PReLU(conv_branch, scope=scope + "/conv")

        # # 1x1 expansion:
        W_exp = self.get_variable_weight_decay(scope + "/W_exp",
                    shape=[1, 1, internal_depth, output_depth], # ([filter_height, filter_width, in_depth, out_depth])
                    initializer=tf.contrib.layers.xavier_initializer(),
                    loss_category="encoder_wd_losses")
        conv_branch = tf.nn.conv2d(conv_branch, W_exp, strides=[1, 1, 1, 1],
                    padding="VALID") # NOTE! no bias terms
        # # # batch norm:
        conv_branch = tf.contrib.slim.batch_norm(conv_branch)
        # NOTE! no PReLU here

        # # regularizer:
        conv_branch = spatial_dropout(conv_branch, drop_prob)


        # main branch:
        main_branch = x

        if downsampling:
            # max pooling with argmax (for use in max_unpool in the decoder):
            main_branch, pooling_indices = tf.nn.max_pool_with_argmax(main_branch,
                        ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
            # (everytime we downsample, we also increase the feature block depth)

            # pad with zeros so that the feature block depth matches:
            depth_to_pad = output_depth - input_depth
            paddings = tf.convert_to_tensor([[0, 0], [0, 0], [0, 0], [0, depth_to_pad]])
            # (paddings is an integer tensor of shape [4, 2] where 4 is the rank
            # of main_branch. For each dimension D (D = 0, 1, 2, 3) of main_branch,
            # paddings[D, 0] is the no of values to add before the contents of
            # main_branch in that dimension, and paddings[D, 0] is the no of
            # values to add after the contents of main_branch in that dimension)
            main_branch = tf.pad(main_branch, paddings=paddings, mode="CONSTANT")


        # add the branches:
        merged = conv_branch + main_branch

        # apply PReLU:
        output = PReLU(merged, scope=scope + "/output")

        if downsampling:
            return output, pooling_indices
        else:
            return output

    def encoder_bottleneck_dilated(self, x, output_depth, drop_prob, scope,
                dilation_rate, proj_ratio=4):
        input_shape = x.get_shape().as_list()
        input_depth = input_shape[3]

        internal_depth = int(output_depth/proj_ratio)

        # convolution branch:
        conv_branch = x

        # # 1x1 projection:
        W_proj = self.get_variable_weight_decay(scope + "/W_proj",
                    shape=[1, 1, input_depth, internal_depth], # ([filter_height, filter_width, in_depth, out_depth])
                    initializer=tf.contrib.layers.xavier_initializer(),
                    loss_category="encoder_wd_losses")
        conv_branch = tf.nn.conv2d(conv_branch, W_proj, strides=[1, 1, 1, 1],
                    padding="VALID") # NOTE! no bias terms
        # # # batch norm and PReLU:
        conv_branch = tf.contrib.slim.batch_norm(conv_branch)
        conv_branch = PReLU(conv_branch, scope=scope + "/proj")

        # # dilated conv:
        W_conv = self.get_variable_weight_decay(scope + "/W_conv",
                    shape=[3, 3, internal_depth, internal_depth], # ([filter_height, filter_width, in_depth, out_depth])
                    initializer=tf.contrib.layers.xavier_initializer(),
                    loss_category="encoder_wd_losses")
        b_conv = self.get_variable_weight_decay(scope + "/b_conv", shape=[internal_depth], # ([out_depth])
                    initializer=tf.constant_initializer(0),
                    loss_category="encoder_wd_losses")
        conv_branch = tf.nn.atrous_conv2d(conv_branch, W_conv, rate=dilation_rate,
                    padding="SAME") + b_conv
        # # # batch norm and PReLU:
        conv_branch = tf.contrib.slim.batch_norm(conv_branch)
        conv_branch = PReLU(conv_branch, scope=scope + "/conv")

        # # 1x1 expansion:
        W_exp = self.get_variable_weight_decay(scope + "/W_exp",
                    shape=[1, 1, internal_depth, output_depth], # ([filter_height, filter_width, in_depth, out_depth])
                    initializer=tf.contrib.layers.xavier_initializer(),
                    loss_category="encoder_wd_losses")
        conv_branch = tf.nn.conv2d(conv_branch, W_exp, strides=[1, 1, 1, 1],
                    padding="VALID") # NOTE! no bias terms
        # # # batch norm:
        conv_branch = tf.contrib.slim.batch_norm(conv_branch)
        # NOTE! no PReLU here

        # # regularizer:
        conv_branch = spatial_dropout(conv_branch, drop_prob)


        # main branch:
        main_branch = x


        # add the branches:
        merged = conv_branch + main_branch

        # apply PReLU:
        output = PReLU(merged, scope=scope + "/output")

        return output

    def encoder_bottleneck_asymmetric(self, x, output_depth, drop_prob, scope, proj_ratio=4):
        input_shape = x.get_shape().as_list()
        input_depth = input_shape[3]

        internal_depth = int(output_depth/proj_ratio)

        # convolution branch:
        conv_branch = x

        # # 1x1 projection:
        W_proj = self.get_variable_weight_decay(scope + "/W_proj",
                    shape=[1, 1, input_depth, internal_depth], # ([filter_height, filter_width, in_depth, out_depth])
                    initializer=tf.contrib.layers.xavier_initializer(),
                    loss_category="encoder_wd_losses")
        conv_branch = tf.nn.conv2d(conv_branch, W_proj, strides=[1, 1, 1, 1],
                    padding="VALID") # NOTE! no bias terms
        # # # batch norm and PReLU:
        conv_branch = tf.contrib.slim.batch_norm(conv_branch)
        conv_branch = PReLU(conv_branch, scope=scope + "/proj")

        # # asymmetric conv:
        # # # asymmetric conv 1:
        W_conv1 = self.get_variable_weight_decay(scope + "/W_conv1",
                    shape=[5, 1, internal_depth, internal_depth], # ([filter_height, filter_width, in_depth, out_depth])
                    initializer=tf.contrib.layers.xavier_initializer(),
                    loss_category="encoder_wd_losses")
        conv_branch = tf.nn.conv2d(conv_branch, W_conv1, strides=[1, 1, 1, 1],
                    padding="SAME") # NOTE! no bias terms
        # # # asymmetric conv 2:
        W_conv2 = self.get_variable_weight_decay(scope + "/W_conv2",
                    shape=[1, 5, internal_depth, internal_depth], # ([filter_height, filter_width, in_depth, out_depth])
                    initializer=tf.contrib.layers.xavier_initializer(),
                    loss_category="encoder_wd_losses")
        b_conv2 = self.get_variable_weight_decay(scope + "/b_conv2", shape=[internal_depth], # ([out_depth])
                    initializer=tf.constant_initializer(0),
                    loss_category="encoder_wd_losses")
        conv_branch = tf.nn.conv2d(conv_branch, W_conv2, strides=[1, 1, 1, 1],
                    padding="SAME") + b_conv2
        # # # batch norm and PReLU:
        conv_branch = tf.contrib.slim.batch_norm(conv_branch)
        conv_branch = PReLU(conv_branch, scope=scope + "/conv")

        # # 1x1 expansion:
        W_exp = self.get_variable_weight_decay(scope + "/W_exp",
                    shape=[1, 1, internal_depth, output_depth], # ([filter_height, filter_width, in_depth, out_depth])
                    initializer=tf.contrib.layers.xavier_initializer(),
                    loss_category="encoder_wd_losses")
        conv_branch = tf.nn.conv2d(conv_branch, W_exp, strides=[1, 1, 1, 1],
                    padding="VALID") # NOTE! no bias terms
        # # # batch norm:
        conv_branch = tf.contrib.slim.batch_norm(conv_branch)
        # NOTE! no PReLU here

        # # regularizer:
        conv_branch = spatial_dropout(conv_branch, drop_prob)


        # main branch:
        main_branch = x


        # add the branches:
        merged = conv_branch + main_branch

        # apply PReLU:
        output = PReLU(merged, scope=scope + "/output")

        return output

    def decoder_bottleneck(self, x, output_depth, scope, proj_ratio=4,
                upsampling=False, pooling_indices=None, output_shape=None):
        # NOTE! decoder uses ReLU instead of PReLU

        input_shape = x.get_shape().as_list()
        input_depth = input_shape[3]

        internal_depth = int(output_depth/proj_ratio)

        # main branch:
        main_branch = x

        if upsampling:
            # # 1x1 projection (to decrease depth to the same value as before downsampling):
            W_upsample = self.get_variable_weight_decay(scope + "/W_upsample",
                        shape=[1, 1, input_depth, output_depth], # ([filter_height, filter_width, in_depth, out_depth])
                        initializer=tf.contrib.layers.xavier_initializer(),
                        loss_category="decoder_wd_losses")
            main_branch = tf.nn.conv2d(main_branch, W_upsample, strides=[1, 1, 1, 1],
                        padding="VALID") # NOTE! no bias terms
            # # # batch norm:
            main_branch = tf.contrib.slim.batch_norm(main_branch)
            # NOTE! no ReLU here

            # # max unpooling:
            main_branch = max_unpool(main_branch, pooling_indices, output_shape)

        main_branch = tf.cast(main_branch, tf.float32)


        # convolution branch:
        conv_branch = x

        # # 1x1 projection:
        W_proj = self.get_variable_weight_decay(scope + "/W_proj",
                    shape=[1, 1, input_depth, internal_depth], # ([filter_height, filter_width, in_depth, out_depth])
                    initializer=tf.contrib.layers.xavier_initializer(),
                    loss_category="decoder_wd_losses")
        conv_branch = tf.nn.conv2d(conv_branch, W_proj, strides=[1, 1, 1, 1],
                    padding="VALID") # NOTE! no bias terms
        # # # batch norm and ReLU:
        conv_branch = tf.contrib.slim.batch_norm(conv_branch)
        conv_branch = tf.nn.relu(conv_branch)

        # # conv:
        if upsampling:
            # deconvolution:
            W_conv = self.get_variable_weight_decay(scope + "/W_conv",
                        shape=[3, 3, internal_depth, internal_depth], # ([filter_height, filter_width, in_depth, out_depth])
                        initializer=tf.contrib.layers.xavier_initializer(),
                        loss_category="decoder_wd_losses")
            b_conv = self.get_variable_weight_decay(scope + "/b_conv", shape=[internal_depth], # ([out_depth]], one bias weight per out depth layer),
                        initializer=tf.constant_initializer(0),
                        loss_category="decoder_wd_losses")
            main_branch_shape = main_branch.get_shape().as_list()
            output_shape = tf.convert_to_tensor([main_branch_shape[0],
                        main_branch_shape[1], main_branch_shape[2], internal_depth])
            conv_branch = tf.nn.conv2d_transpose(conv_branch, W_conv, output_shape=output_shape,
                        strides=[1, 2, 2, 1], padding="SAME") + b_conv
        else:
            W_conv = self.get_variable_weight_decay(scope + "/W_conv",
                        shape=[3, 3, internal_depth, internal_depth], # ([filter_height, filter_width, in_depth, out_depth])
                        initializer=tf.contrib.layers.xavier_initializer(),
                        loss_category="decoder_wd_losses")
            b_conv = self.get_variable_weight_decay(scope + "/b_conv", shape=[internal_depth], # ([out_depth])
                        initializer=tf.constant_initializer(0),
                        loss_category="decoder_wd_losses")
            conv_branch = tf.nn.conv2d(conv_branch, W_conv, strides=[1, 1, 1, 1],
                        padding="SAME") + b_conv
        # # # batch norm and ReLU:
        conv_branch = tf.contrib.slim.batch_norm(conv_branch)
        conv_branch = tf.nn.relu(conv_branch)

        # # 1x1 expansion:
        W_exp = self.get_variable_weight_decay(scope + "/W_exp",
                    shape=[1, 1, internal_depth, output_depth], # ([filter_height, filter_width, in_depth, out_depth])
                    initializer=tf.contrib.layers.xavier_initializer(),
                    loss_category="decoder_wd_losses")
        conv_branch = tf.nn.conv2d(conv_branch, W_exp, strides=[1, 1, 1, 1],
                    padding="VALID") # NOTE! no bias terms
        # # # batch norm:
        conv_branch = tf.contrib.slim.batch_norm(conv_branch)
        # NOTE! no ReLU here

        # NOTE! no regularizer


        # add the branches:
        merged = conv_branch + main_branch

        # apply ReLU:
        output = tf.nn.relu(merged)

        return output

    def get_variable_weight_decay(self, name, shape, initializer, loss_category,
                dtype=tf.float32):
        variable = tf.get_variable(name, shape=shape, dtype=dtype,
                    initializer=initializer)

        # add a variable weight decay loss:
        weight_decay = self.wd*tf.nn.l2_loss(variable)
        tf.add_to_collection(loss_category, weight_decay)

        return variabl

class ICNet(Network):
    def __init__(self, is_training=False, num_classes=19, input_size=[1024, 2048]):
        self.input_size = input_size
    
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, None, 3])
        self.img_tf, self.shape = preprocess(self.x, self.input_size, 'icnet')
        
        super().__init__({'data': self.img_tf}, num_classes, is_training)

    def setup(self, is_training, num_classes):
        (self.feed('data')
             .interp(factor=0.5, name='data_sub2')
             .conv(3, 3, 32, 2, 2, biased=True, padding='SAME', relu=True, name='conv1_1_3x3_s2')
             .conv(3, 3, 32, 1, 1, biased=True, padding='SAME', relu=True, name='conv1_2_3x3')
             .conv(3, 3, 64, 1, 1, biased=True, padding='SAME', relu=True, name='conv1_3_3x3')
             .max_pool(3, 3, 2, 2, name='pool1_3x3_s2')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='conv2_1_1x1_proj'))

        (self.feed('pool1_3x3_s2')
             .conv(1, 1, 32, 1, 1, biased=True, relu=True, name='conv2_1_1x1_reduce')
             .zero_padding(paddings=1, name='padding1')
             .conv(3, 3, 32, 1, 1, biased=True, relu=True, name='conv2_1_3x3')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='conv2_1_1x1_increase'))

        (self.feed('conv2_1_1x1_proj',
                   'conv2_1_1x1_increase')
             .add(name='conv2_1')
             .relu(name='conv2_1/relu')
             .conv(1, 1, 32, 1, 1, biased=True, relu=True, name='conv2_2_1x1_reduce')
             .zero_padding(paddings=1, name='padding2')
             .conv(3, 3, 32, 1, 1, biased=True, relu=True, name='conv2_2_3x3')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='conv2_2_1x1_increase'))

        (self.feed('conv2_1/relu',
                   'conv2_2_1x1_increase')
             .add(name='conv2_2')
             .relu(name='conv2_2/relu')
             .conv(1, 1, 32, 1, 1, biased=True, relu=True, name='conv2_3_1x1_reduce')
             .zero_padding(paddings=1, name='padding3')
             .conv(3, 3, 32, 1, 1, biased=True, relu=True, name='conv2_3_3x3')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='conv2_3_1x1_increase'))

        (self.feed('conv2_2/relu',
                   'conv2_3_1x1_increase')
             .add(name='conv2_3')
             .relu(name='conv2_3/relu')
             .conv(1, 1, 256, 2, 2, biased=True, relu=False, name='conv3_1_1x1_proj'))

        (self.feed('conv2_3/relu')
             .conv(1, 1, 64, 2, 2, biased=True, relu=True, name='conv3_1_1x1_reduce')
             .zero_padding(paddings=1, name='padding4')
             .conv(3, 3, 64, 1, 1, biased=True, relu=True, name='conv3_1_3x3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='conv3_1_1x1_increase'))

        (self.feed('conv3_1_1x1_proj',
                   'conv3_1_1x1_increase')
             .add(name='conv3_1')
             .relu(name='conv3_1/relu')
             .interp(factor=0.5, name='conv3_1_sub4')
             .conv(1, 1, 64, 1, 1, biased=True, relu=True, name='conv3_2_1x1_reduce')
             .zero_padding(paddings=1, name='padding5')
             .conv(3, 3, 64, 1, 1, biased=True, relu=True, name='conv3_2_3x3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='conv3_2_1x1_increase'))

        (self.feed('conv3_1_sub4',
                   'conv3_2_1x1_increase')
             .add(name='conv3_2')
             .relu(name='conv3_2/relu')
             .conv(1, 1, 64, 1, 1, biased=True, relu=True, name='conv3_3_1x1_reduce')
             .zero_padding(paddings=1, name='padding6')
             .conv(3, 3, 64, 1, 1, biased=True, relu=True, name='conv3_3_3x3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='conv3_3_1x1_increase'))

        (self.feed('conv3_2/relu',
                   'conv3_3_1x1_increase')
             .add(name='conv3_3')
             .relu(name='conv3_3/relu')
             .conv(1, 1, 64, 1, 1, biased=True, relu=True, name='conv3_4_1x1_reduce')
             .zero_padding(paddings=1, name='padding7')
             .conv(3, 3, 64, 1, 1, biased=True, relu=True, name='conv3_4_3x3')
             .conv(1, 1, 256, 1, 1, biased=True, relu=False, name='conv3_4_1x1_increase'))

        (self.feed('conv3_3/relu',
                   'conv3_4_1x1_increase')
             .add(name='conv3_4')
             .relu(name='conv3_4/relu')
             .conv(1, 1, 512, 1, 1, biased=True, relu=False, name='conv4_1_1x1_proj'))

        (self.feed('conv3_4/relu')
             .conv(1, 1, 128, 1, 1, biased=True, relu=True, name='conv4_1_1x1_reduce')
             .zero_padding(paddings=2, name='padding8')
             .atrous_conv(3, 3, 128, 2, biased=True, relu=True, name='conv4_1_3x3')
             .conv(1, 1, 512, 1, 1, biased=True, relu=False, name='conv4_1_1x1_increase'))

        (self.feed('conv4_1_1x1_proj',
                   'conv4_1_1x1_increase')
             .add(name='conv4_1')
             .relu(name='conv4_1/relu')
             .conv(1, 1, 128, 1, 1, biased=True, relu=True, name='conv4_2_1x1_reduce')
             .zero_padding(paddings=2, name='padding9')
             .atrous_conv(3, 3, 128, 2, biased=True, relu=True, name='conv4_2_3x3')
             .conv(1, 1, 512, 1, 1, biased=True, relu=False, name='conv4_2_1x1_increase'))

        (self.feed('conv4_1/relu',
                   'conv4_2_1x1_increase')
             .add(name='conv4_2')
             .relu(name='conv4_2/relu')
             .conv(1, 1, 128, 1, 1, biased=True, relu=True, name='conv4_3_1x1_reduce')
             .zero_padding(paddings=2, name='padding10')
             .atrous_conv(3, 3, 128, 2, biased=True, relu=True, name='conv4_3_3x3')
             .conv(1, 1, 512, 1, 1, biased=True, relu=False, name='conv4_3_1x1_increase'))

        (self.feed('conv4_2/relu',
                   'conv4_3_1x1_increase')
             .add(name='conv4_3')
             .relu(name='conv4_3/relu')
             .conv(1, 1, 128, 1, 1, biased=True, relu=True, name='conv4_4_1x1_reduce')
             .zero_padding(paddings=2, name='padding11')
             .atrous_conv(3, 3, 128, 2, biased=True, relu=True, name='conv4_4_3x3')
             .conv(1, 1, 512, 1, 1, biased=True, relu=False, name='conv4_4_1x1_increase'))

        (self.feed('conv4_3/relu',
                   'conv4_4_1x1_increase')
             .add(name='conv4_4')
             .relu(name='conv4_4/relu')
             .conv(1, 1, 128, 1, 1, biased=True, relu=True, name='conv4_5_1x1_reduce')
             .zero_padding(paddings=2, name='padding12')
             .atrous_conv(3, 3, 128, 2, biased=True, relu=True, name='conv4_5_3x3')
             .conv(1, 1, 512, 1, 1, biased=True, relu=False, name='conv4_5_1x1_increase'))

        (self.feed('conv4_4/relu',
                   'conv4_5_1x1_increase')
             .add(name='conv4_5')
             .relu(name='conv4_5/relu')
             .conv(1, 1, 128, 1, 1, biased=True, relu=True, name='conv4_6_1x1_reduce')
             .zero_padding(paddings=2, name='padding13')
             .atrous_conv(3, 3, 128, 2, biased=True, relu=True, name='conv4_6_3x3')
             .conv(1, 1, 512, 1, 1, biased=True, relu=False, name='conv4_6_1x1_increase'))

        (self.feed('conv4_5/relu',
                   'conv4_6_1x1_increase')
             .add(name='conv4_6')
             .relu(name='conv4_6/relu')
             .conv(1, 1, 1024, 1, 1, biased=True, relu=False, name='conv5_1_1x1_proj'))

        (self.feed('conv4_6/relu')
             .conv(1, 1, 256, 1, 1, biased=True, relu=True, name='conv5_1_1x1_reduce')
             .zero_padding(paddings=4, name='padding14')
             .atrous_conv(3, 3, 256, 4, biased=True, relu=True, name='conv5_1_3x3')
             .conv(1, 1, 1024, 1, 1, biased=True, relu=False, name='conv5_1_1x1_increase'))

        (self.feed('conv5_1_1x1_proj',
                   'conv5_1_1x1_increase')
             .add(name='conv5_1')
             .relu(name='conv5_1/relu')
             .conv(1, 1, 256, 1, 1, biased=True, relu=True, name='conv5_2_1x1_reduce')
             .zero_padding(paddings=4, name='padding15')
             .atrous_conv(3, 3, 256, 4, biased=True, relu=True, name='conv5_2_3x3')
             .conv(1, 1, 1024, 1, 1, biased=True, relu=False, name='conv5_2_1x1_increase'))

        (self.feed('conv5_1/relu',
                   'conv5_2_1x1_increase')
             .add(name='conv5_2')
             .relu(name='conv5_2/relu')
             .conv(1, 1, 256, 1, 1, biased=True, relu=True, name='conv5_3_1x1_reduce')
             .zero_padding(paddings=4, name='padding16')
             .atrous_conv(3, 3, 256, 4, biased=True, relu=True, name='conv5_3_3x3')
             .conv(1, 1, 1024, 1, 1, biased=True, relu=False, name='conv5_3_1x1_increase'))

        (self.feed('conv5_2/relu',
                   'conv5_3_1x1_increase')
             .add(name='conv5_3')
             .relu(name='conv5_3/relu'))

        shape = self.layers['conv5_3/relu'].get_shape().as_list()[1:3]
        h, w = shape
        
        (self.feed('conv5_3/relu')
             .avg_pool(h, w, h, w, name='conv5_3_pool1')
             .resize_bilinear(shape, name='conv5_3_pool1_interp'))

        (self.feed('conv5_3/relu')
             .avg_pool(h/2, w/2, h/2, w/2, name='conv5_3_pool2')
             .resize_bilinear(shape, name='conv5_3_pool2_interp'))

        (self.feed('conv5_3/relu')
             .avg_pool(h/3, w/3, h/3, w/3, name='conv5_3_pool3')
             .resize_bilinear(shape, name='conv5_3_pool3_interp'))

        (self.feed('conv5_3/relu')
             .avg_pool(h/4, w/4, h/4, w/4, name='conv5_3_pool6')
             .resize_bilinear(shape, name='conv5_3_pool6_interp'))

        (self.feed('conv5_3/relu',
                   'conv5_3_pool6_interp',
                   'conv5_3_pool3_interp',
                   'conv5_3_pool2_interp',
                   'conv5_3_pool1_interp')
             .add(name='conv5_3_sum')
             .conv(1, 1, 256, 1, 1, biased=True, relu=True, name='conv5_4_k1')
             .interp(factor=2.0, name='conv5_4_interp')
             .zero_padding(paddings=2, name='padding17')
             .atrous_conv(3, 3, 128, 2, biased=True, relu=False, name='conv_sub4'))

        (self.feed('conv3_1/relu')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='conv3_1_sub2_proj'))

        (self.feed('conv_sub4',
                   'conv3_1_sub2_proj')
             .add(name='sub24_sum')
             .relu(name='sub24_sum/relu')
             .interp(factor=2.0, name='sub24_sum_interp')
             .zero_padding(paddings=2, name='padding18')
             .atrous_conv(3, 3, 128, 2, biased=True, relu=False, name='conv_sub2'))

        (self.feed('data')
             .conv(3, 3, 32, 2, 2, biased=True, padding='SAME', relu=True, name='conv1_sub1')
             .conv(3, 3, 32, 2, 2, biased=True, padding='SAME', relu=True, name='conv2_sub1')
             .conv(3, 3, 64, 2, 2, biased=True, padding='SAME', relu=True, name='conv3_sub1')
             .conv(1, 1, 128, 1, 1, biased=True, relu=False, name='conv3_sub1_proj'))

        (self.feed('conv_sub2',
                   'conv3_sub1_proj')
             .add(name='sub12_sum')
             .relu(name='sub12_sum/relu')
             .interp(factor=2.0, name='sub12_sum_interp')
             .conv(1, 1, num_classes, 1, 1, biased=True, relu=False, name='conv6_cls'))

        raw_output = self.layers['conv6_cls']
        raw_output_up = tf.image.resize_bilinear(raw_output, size=self.shape, align_corners=True)
        raw_output_up = tf.argmax(raw_output_up, dimension=3)
        self.pred = decode_labels(raw_output_up, self.shape, num_classes)

    def read_input(self, img_path):
        self.img, self.img_name = load_img(img_path)

    def forward(self, sess):
        return sess.run(self.pred, feed_dict={self.x: self.img})

    """
    def forward(self, img_array, sess):
        return sess.run(self.pred, feed_dict={self.x: self.img_array}) 
    """
