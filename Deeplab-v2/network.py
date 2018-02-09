import tensorflow as tf
import numpy as np
import six



"""
This script defines the segmentation network.

The encoding part is a pre-trained ResNet. This script supports several settings (you need to specify in main.py):
	
	Deeplab v2 pre-trained model (pre-trained on MSCOCO) ('deeplab_resnet_init.ckpt')
	Deeplab v2 pre-trained model (pre-trained on MSCOCO + PASCAL_train+val) ('deeplab_resnet.ckpt')
	Original ResNet-101 ('resnet_v1_101.ckpt')
	Original ResNet-50 ('resnet_v1_50.ckpt')

You may find the download links in README.

To use the pre-trained models, the name of each layer is the same as that in .ckpy file.
"""



class Deeplab_v2(object):
	"""
	Deeplab v2 pre-trained model (pre-trained on MSCOCO) ('deeplab_resnet_init.ckpt')
	Deeplab v2 pre-trained model (pre-trained on MSCOCO + PASCAL_train+val) ('deeplab_resnet.ckpt')
	"""
	def __init__(self, inputs, num_classes, phase):
		self.inputs = inputs
		self.num_classes = num_classes
		self.channel_axis = 3
		self.phase = phase # train (True) or test (False), for BN layers in the decoder
		self.build_network()

	def build_network(self):
		self.encoding = self.build_encoder()
		self.outputs = self.build_decoder(self.encoding)

	def build_encoder(self):
		print("-----------build encoder: deeplab pre-trained-----------")
		outputs = self._start_block()
		print("after start block:", outputs.shape)
		outputs = self._bottleneck_resblock(outputs, 256, '2a', identity_connection=False)
		outputs = self._bottleneck_resblock(outputs, 256, '2b')
		outputs = self._bottleneck_resblock(outputs, 256, '2c')
		print("after block1:", outputs.shape)
		outputs = self._bottleneck_resblock(outputs, 512, '3a',	half_size=True, identity_connection=False)
		for i in six.moves.range(1, 4):
			outputs = self._bottleneck_resblock(outputs, 512, '3b%d' % i)
		print("after block2:", outputs.shape)
		outputs = self._dilated_bottle_resblock(outputs, 1024, 2, '4a',	identity_connection=False)
		for i in six.moves.range(1, 23):
			outputs = self._dilated_bottle_resblock(outputs, 1024, 2, '4b%d' % i)
		print("after block3:", outputs.shape)
		outputs = self._dilated_bottle_resblock(outputs, 2048, 4, '5a',	identity_connection=False)
		outputs = self._dilated_bottle_resblock(outputs, 2048, 4, '5b')
		outputs = self._dilated_bottle_resblock(outputs, 2048, 4, '5c')
		print("after block4:", outputs.shape)
		return outputs

	def build_decoder(self, encoding):
		print("-----------build decoder-----------")
		outputs = self._ASPP(encoding, self.num_classes, [6, 12, 18, 24])
		print("after aspp block:", outputs.shape)
		return outputs

	# blocks
	def _start_block(self):
		outputs = self._conv2d(self.inputs, 7, 64, 2, name='conv1')
		outputs = self._batch_norm(outputs, name='bn_conv1', is_training=False, activation_fn=tf.nn.relu)
		outputs = self._max_pool2d(outputs, 3, 2, name='pool1')
		return outputs

	def _bottleneck_resblock(self, x, num_o, name, half_size=False, identity_connection=True):
		first_s = 2 if half_size else 1
		assert num_o % 4 == 0, 'Bottleneck number of output ERROR!'
		# branch1
		if not identity_connection:
			o_b1 = self._conv2d(x, 1, num_o, first_s, name='res%s_branch1' % name)
			o_b1 = self._batch_norm(o_b1, name='bn%s_branch1' % name, is_training=False, activation_fn=None)
		else:
			o_b1 = x
		# branch2
		o_b2a = self._conv2d(x, 1, num_o / 4, first_s, name='res%s_branch2a' % name)
		o_b2a = self._batch_norm(o_b2a, name='bn%s_branch2a' % name, is_training=False, activation_fn=tf.nn.relu)

		o_b2b = self._conv2d(o_b2a, 3, num_o / 4, 1, name='res%s_branch2b' % name)
		o_b2b = self._batch_norm(o_b2b, name='bn%s_branch2b' % name, is_training=False, activation_fn=tf.nn.relu)

		o_b2c = self._conv2d(o_b2b, 1, num_o, 1, name='res%s_branch2c' % name)
		o_b2c = self._batch_norm(o_b2c, name='bn%s_branch2c' % name, is_training=False, activation_fn=None)
		# add
		outputs = self._add([o_b1,o_b2c], name='res%s' % name)
		# relu
		outputs = self._relu(outputs, name='res%s_relu' % name)
		return outputs

	def _dilated_bottle_resblock(self, x, num_o, dilation_factor, name, identity_connection=True):
		assert num_o % 4 == 0, 'Bottleneck number of output ERROR!'
		# branch1
		if not identity_connection:
			o_b1 = self._conv2d(x, 1, num_o, 1, name='res%s_branch1' % name)
			o_b1 = self._batch_norm(o_b1, name='bn%s_branch1' % name, is_training=False, activation_fn=None)
		else:
			o_b1 = x
		# branch2
		o_b2a = self._conv2d(x, 1, num_o / 4, 1, name='res%s_branch2a' % name)
		o_b2a = self._batch_norm(o_b2a, name='bn%s_branch2a' % name, is_training=False, activation_fn=tf.nn.relu)

		o_b2b = self._dilated_conv2d(o_b2a, 3, num_o / 4, dilation_factor, name='res%s_branch2b' % name)
		o_b2b = self._batch_norm(o_b2b, name='bn%s_branch2b' % name, is_training=False, activation_fn=tf.nn.relu)

		o_b2c = self._conv2d(o_b2b, 1, num_o, 1, name='res%s_branch2c' % name)
		o_b2c = self._batch_norm(o_b2c, name='bn%s_branch2c' % name, is_training=False, activation_fn=None)
		# add
		outputs = self._add([o_b1,o_b2c], name='res%s' % name)
		# relu
		outputs = self._relu(outputs, name='res%s_relu' % name)
		return outputs

	def _ASPP(self, x, num_o, dilations):
		o = []
		for i, d in enumerate(dilations):
			o.append(self._dilated_conv2d(x, 3, num_o, d, name='fc1_voc12_c%d' % i, biased=True))
		return self._add(o, name='fc1_voc12')

	# layers
	def _conv2d(self, x, kernel_size, num_o, stride, name, biased=False):
		"""
		Conv2d without BN or relu.
		"""
		num_x = x.shape[self.channel_axis].value
		with tf.variable_scope(name) as scope:
			w = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_x, num_o])
			s = [1, stride, stride, 1]
			o = tf.nn.conv2d(x, w, s, padding='SAME')
			if biased:
				b = tf.get_variable('biases', shape=[num_o])
				o = tf.nn.bias_add(o, b)
			return o

	def _dilated_conv2d(self, x, kernel_size, num_o, dilation_factor, name, biased=False):
		"""
		Dilated conv2d without BN or relu.
		"""
		num_x = x.shape[self.channel_axis].value
		with tf.variable_scope(name) as scope:
			w = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_x, num_o])
			o = tf.nn.atrous_conv2d(x, w, dilation_factor, padding='SAME')
			if biased:
				b = tf.get_variable('biases', shape=[num_o])
				o = tf.nn.bias_add(o, b)
			return o

	def _relu(self, x, name):
		return tf.nn.relu(x, name=name)

	def _add(self, x_l, name):
		return tf.add_n(x_l, name=name)

	def _max_pool2d(self, x, kernel_size, stride, name):
		k = [1, kernel_size, kernel_size, 1]
		s = [1, stride, stride, 1]
		return tf.nn.max_pool(x, k, s, padding='SAME', name=name)

	def _batch_norm(self, x, name, is_training, activation_fn, trainable=False):
		# For a small batch size, it is better to keep 
		# the statistics of the BN layers (running means and variances) frozen, 
		# and to not update the values provided by the pre-trained model by setting is_training=False.
		# Note that is_training=False still updates BN parameters gamma (scale) and beta (offset)
		# if they are presented in var_list of the optimiser definition.
		# Set trainable = False to remove them from trainable_variables.
		with tf.variable_scope(name) as scope:
			o = tf.contrib.layers.batch_norm(
				x,
				scale=True,
				activation_fn=activation_fn,
				is_training=is_training,
				trainable=trainable,
				scope=scope)
			return o



class ResNet_segmentation(object):
	"""
	Original ResNet-101 ('resnet_v1_101.ckpt')
	Original ResNet-50 ('resnet_v1_50.ckpt')
	"""
	def __init__(self, inputs, num_classes, phase, encoder_name):
		if encoder_name not in ['res101', 'res50']:
			print('encoder_name ERROR!')
			print("Please input: res101, res50")
			sys.exit(-1)
		self.encoder_name = encoder_name
		self.inputs = inputs
		self.num_classes = num_classes
		self.channel_axis = 3
		self.phase = phase # train (True) or test (False), for BN layers in the decoder
		self.build_network()

	def build_network(self):
		self.encoding = self.build_encoder()
		self.outputs = self.build_decoder(self.encoding)

	def build_encoder(self):
		print("-----------build encoder: %s-----------" % self.encoder_name)
		scope_name = 'resnet_v1_101' if self.encoder_name == 'res101' else 'resnet_v1_50'
		with tf.variable_scope(scope_name) as scope:
			outputs = self._start_block('conv1')
			print("after start block:", outputs.shape)
			with tf.variable_scope('block1') as scope:
				outputs = self._bottleneck_resblock(outputs, 256, 'unit_1',	identity_connection=False)
				outputs = self._bottleneck_resblock(outputs, 256, 'unit_2')
				outputs = self._bottleneck_resblock(outputs, 256, 'unit_3')
				print("after block1:", outputs.shape)
			with tf.variable_scope('block2') as scope:
				outputs = self._bottleneck_resblock(outputs, 512, 'unit_1',	half_size=True, identity_connection=False)
				for i in six.moves.range(2, 5):
					outputs = self._bottleneck_resblock(outputs, 512, 'unit_%d' % i)
				print("after block2:", outputs.shape)
			with tf.variable_scope('block3') as scope:
				outputs = self._dilated_bottle_resblock(outputs, 1024, 2, 'unit_1',	identity_connection=False)
				num_layers_block3 = 23 if self.encoder_name == 'res101' else 6
				for i in six.moves.range(2, num_layers_block3+1):
					outputs = self._dilated_bottle_resblock(outputs, 1024, 2, 'unit_%d' % i)
				print("after block3:", outputs.shape)
			with tf.variable_scope('block4') as scope:
				outputs = self._dilated_bottle_resblock(outputs, 2048, 4, 'unit_1', identity_connection=False)
				outputs = self._dilated_bottle_resblock(outputs, 2048, 4, 'unit_2')
				outputs = self._dilated_bottle_resblock(outputs, 2048, 4, 'unit_3')
				print("after block4:", outputs.shape)
				return outputs

	def build_decoder(self, encoding):
		print("-----------build decoder-----------")
		with tf.variable_scope('decoder') as scope:
			outputs = self._ASPP(encoding, self.num_classes, [6, 12, 18, 24])
			print("after aspp block:", outputs.shape)
			return outputs

	# blocks
	def _start_block(self, name):
		outputs = self._conv2d(self.inputs, 7, 64, 2, name=name)
		outputs = self._batch_norm(outputs, name=name, is_training=False, activation_fn=tf.nn.relu)
		outputs = self._max_pool2d(outputs, 3, 2, name='pool1')
		return outputs

	def _bottleneck_resblock(self, x, num_o, name, half_size=False, identity_connection=True):
		first_s = 2 if half_size else 1
		assert num_o % 4 == 0, 'Bottleneck number of output ERROR!'
		# branch1
		if not identity_connection:
			o_b1 = self._conv2d(x, 1, num_o, first_s, name='%s/bottleneck_v1/shortcut' % name)
			o_b1 = self._batch_norm(o_b1, name='%s/bottleneck_v1/shortcut' % name, is_training=False, activation_fn=None)
		else:
			o_b1 = x
		# branch2
		o_b2a = self._conv2d(x, 1, num_o / 4, first_s, name='%s/bottleneck_v1/conv1' % name)
		o_b2a = self._batch_norm(o_b2a, name='%s/bottleneck_v1/conv1' % name, is_training=False, activation_fn=tf.nn.relu)

		o_b2b = self._conv2d(o_b2a, 3, num_o / 4, 1, name='%s/bottleneck_v1/conv2' % name)
		o_b2b = self._batch_norm(o_b2b, name='%s/bottleneck_v1/conv2' % name, is_training=False, activation_fn=tf.nn.relu)

		o_b2c = self._conv2d(o_b2b, 1, num_o, 1, name='%s/bottleneck_v1/conv3' % name)
		o_b2c = self._batch_norm(o_b2c, name='%s/bottleneck_v1/conv3' % name, is_training=False, activation_fn=None)
		# add
		outputs = self._add([o_b1,o_b2c], name='%s/bottleneck_v1/add' % name)
		# relu
		outputs = self._relu(outputs, name='%s/bottleneck_v1/relu' % name)
		return outputs

	def _dilated_bottle_resblock(self, x, num_o, dilation_factor, name, identity_connection=True):
		assert num_o % 4 == 0, 'Bottleneck number of output ERROR!'
		# branch1
		if not identity_connection:
			o_b1 = self._conv2d(x, 1, num_o, 1, name='%s/bottleneck_v1/shortcut' % name)
			o_b1 = self._batch_norm(o_b1, name='%s/bottleneck_v1/shortcut' % name, is_training=False, activation_fn=None)
		else:
			o_b1 = x
		# branch2
		o_b2a = self._conv2d(x, 1, num_o / 4, 1, name='%s/bottleneck_v1/conv1' % name)
		o_b2a = self._batch_norm(o_b2a, name='%s/bottleneck_v1/conv1' % name, is_training=False, activation_fn=tf.nn.relu)

		o_b2b = self._dilated_conv2d(o_b2a, 3, num_o / 4, dilation_factor, name='%s/bottleneck_v1/conv2' % name)
		o_b2b = self._batch_norm(o_b2b, name='%s/bottleneck_v1/conv2' % name, is_training=False, activation_fn=tf.nn.relu)

		o_b2c = self._conv2d(o_b2b, 1, num_o, 1, name='%s/bottleneck_v1/conv3' % name)
		o_b2c = self._batch_norm(o_b2c, name='%s/bottleneck_v1/conv3' % name, is_training=False, activation_fn=None)
		# add
		outputs = self._add([o_b1,o_b2c], name='%s/bottleneck_v1/add' % name)
		# relu
		outputs = self._relu(outputs, name='%s/bottleneck_v1/relu' % name)
		return outputs

	def _ASPP(self, x, num_o, dilations):
		o = []
		for i, d in enumerate(dilations):
			o.append(self._dilated_conv2d(x, 3, num_o, d, name='aspp/conv%d' % (i+1), biased=True))
		return self._add(o, name='aspp/add')

	# layers
	def _conv2d(self, x, kernel_size, num_o, stride, name, biased=False):
		"""
		Conv2d without BN or relu.
		"""
		num_x = x.shape[self.channel_axis].value
		with tf.variable_scope(name) as scope:
			w = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_x, num_o])
			s = [1, stride, stride, 1]
			o = tf.nn.conv2d(x, w, s, padding='SAME')
			if biased:
				b = tf.get_variable('biases', shape=[num_o])
				o = tf.nn.bias_add(o, b)
			return o

	def _dilated_conv2d(self, x, kernel_size, num_o, dilation_factor, name, biased=False):
		"""
		Dilated conv2d without BN or relu.
		"""
		num_x = x.shape[self.channel_axis].value
		with tf.variable_scope(name) as scope:
			w = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_x, num_o])
			o = tf.nn.atrous_conv2d(x, w, dilation_factor, padding='SAME')
			if biased:
				b = tf.get_variable('biases', shape=[num_o])
				o = tf.nn.bias_add(o, b)
			return o

	def _relu(self, x, name):
		return tf.nn.relu(x, name=name)

	def _add(self, x_l, name):
		return tf.add_n(x_l, name=name)

	def _max_pool2d(self, x, kernel_size, stride, name):
		k = [1, kernel_size, kernel_size, 1]
		s = [1, stride, stride, 1]
		return tf.nn.max_pool(x, k, s, padding='SAME', name=name)

	def _batch_norm(self, x, name, is_training, activation_fn, trainable=False):
		# For a small batch size, it is better to keep 
		# the statistics of the BN layers (running means and variances) frozen, 
		# and to not update the values provided by the pre-trained model by setting is_training=False.
		# Note that is_training=False still updates BN parameters gamma (scale) and beta (offset)
		# if they are presented in var_list of the optimiser definition.
		# Set trainable = False to remove them from trainable_variables.
		with tf.variable_scope(name+'/BatchNorm') as scope:
			o = tf.contrib.layers.batch_norm(
				x,
				scale=True,
				activation_fn=activation_fn,
				is_training=is_training,
				trainable=trainable,
				scope=scope)
			return o
