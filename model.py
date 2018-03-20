# -*- coding: utf-8 -*-
""" Model

This module can use for build audio U-Net as the libary. You don't need modify
any functions in the libary. Please see the description in front of each
function if you don't undersand it. Please import this module if you want use
this libary in anthor project. Note: prev_num_filters = num_channels

################################################################################
# Author: Weikun Han <weikunhan@gmail.com>
# Crate Date: 03/6/2018
# Update: 03/19/2018
# Reference: https://github.com/jhetherly/EnglishSpeechUpsampler
################################################################################
"""

import tensorflow as tf

#############################
# TENSORBOARD HELPER FUNCTION
#############################

def comprehensive_variable_summaries(var):
    """
    Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    reference: 
        https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard
    """
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def histogram_variable_summaries(var):
    """
    Attach a histogram summary to a Tensor (for TensorBoard visualization).
    reference:
        https://www.tensorflow.org/programmers_guide/tensorboard_histograms
    """
    with tf.name_scope('summaries'):
        tf.summary.histogram('histogram', var)

########################
# LAYER HELPER FUNCTIONS
########################

def subpixel_shuffling_helper(input_tensor, num_channels):
    """ The helper function for build subpixel reshuffling layer

    performs a 1-D subpixel reshuffle of the input 2-D tensor
    assumes the last dimension of X is the filter dimension
    ref: https://github.com/Tetrachrome/subpixel

    Args:
        param1 (tensor): input_tensor
        param2 (int): num_channels
        param3 (str): name

    Returns:
        tensor: output a subpixel reshuffling layer for 2D tensor

    """

    # The tensor pass '[-1,]' to flatten 'tensor'
    # The axis=1 means split the channels the same as resifual tensor channels
    return tf.transpose(
        tf.stack([tf.reshape(x, [-1, ])
                 for x in tf.split(input_tensor, num_channels, axis=1)]))

def subpixel_shuffling(input_tensor, num_channels, name=None):
    """ Entrance function of a subpixel reshuffling layer

    This function is selest each 2D tensor in input 3D tensor.

    Args:
        param1 (tensor): input_tensor
        param2 (int): num_channels
        param3 (str): name

    Returns:
        tensor: output a subpixel reshuffling layer for 3D tensor

    """

    # From 3D tensor (input_tensor) selet each 2D tensor (x)
    # The num_channels is the residual tensor filter_size
    return tf.map_fn(
        lambda x: subpixel_shuffling_helper(x, num_channels),
                  input_tensor,
                  name=name)

def stacking_subpixel_helper(input_tensor,
                             stack_prev_num_filters,
                             stack_num_filters,
                             name=None):
    """ The function for stacking subpixel reshuffling layers

    This function is to stack more 2D tensor in 3D tensor. Performs a subpixel
    restacking such that it restacks columns of a 2-D tensor onto the rows

    Args:
        param1 (tensor): input_tensor
        param2 (int): stack_prev_num_filters
        param3 (str): stack_num_filters
        param4 (str): name

    Returns:
        tensor: output stacking subpixel reshuffling layers for 3D tensor

    """

    # Record the filter size for input tensor
    filter_size = tf.shape(input_tensor)[0]

    # Get number of 2D tensor we need
    need_prev_num_filters = (stack_prev_num_filters -
                             input_tensor.get_shape().as_list()[1])

    # Calculter total new spece need for stacking
    total_new_space = need_prev_num_filters * stack_num_filters

    # Slice the input tensor for stacking, cut from number of filters need for
    # this stacking
    stack_tensor = tf.slice(input_tensor,
                            [0, 0, stack_num_filters],
                            [-1, -1, -1])

    # Reshape stack tensor to 2D tensor, and cut form total new spece need for
    # this stacking
    stack_tensor = tf.slice(tf.reshape(stack_tensor, [filter_size, -1]),
                            [0, 0],
                            [-1, total_new_space])

    # Reshpe this 2D tensor to the 3D tensor
    stack_tensor = tf.reshape(stack_tensor, [filter_size, -1, stack_num_filters])

    # Slice the stack tensor from number of previous filter need
    stack_tensor = tf.slice(stack_tensor,
                            [0, 0, 0],
                            [-1, need_prev_num_filters, -1])

    # Get the base tensor to stack on
    base_tensor = tf.slice(input_tensor,
                           [0, 0, 0],
                           [-1, -1, stack_num_filters])

    # This time want to stack all 2D tensors to combined 3D tensor, so axis=1
    return tf.concat((base_tensor, stack_tensor),
                     axis=1,
                     name=name)

def stacking_subpixel(input_tensor,
                      stack_prev_num_filters,
                      stack_num_filters=None,
                      name=None):
    """ Entrance function of stacking subpixel reshuffling layers

    This function identfy how many filters need for stacking.
    If we want to stack 2D tensor and keep same elements, we need change
    number of filters in current layer.

    Args:
        param1 (tensor): input_tensor
        param2 (int): stack_prev_num_filters
        param3 (str): stack_num_filters
        param4 (str): name

    Returns:
        tensor: output stacking subpixel reshuffling layers for 3D tensor

    """
    prev_num_filters = input_tensor.get_shape().as_list()[1]
    num_filters = input_tensor.get_shape().as_list()[2]
    need_prev_num_filters = stack_prev_num_filters - prev_num_filters

    if stack_num_filters is None:

        # Start a loop keep reduce number of filter to search number of stack
        # filter we need
        for i in range(1, num_filters):
            reduce_num_filters = i
            stack_num_filters = num_filters - reduce_num_filters
            size_change = reduce_num_filters * prev_num_filters
            size_goal = stack_num_filters * need_prev_num_filters

            # If find the elements fits all final set, stop it
            if size_change >= size_goal:
                break
    return stacking_subpixel_helper(input_tensor,
                                    stack_prev_num_filters,
                                    stack_num_filters,
                                    name=name)

def batch_normalization(input_tensor, is_training, scope):
    """ Build general batch normalizaion function

    This function is use for add batch normalization for deep neural netwok.
    In the neural network training, when the convergence speed is very slow, or
    when a gradient explosion or other untrainable condition is encountered,
    batch normalizaion can be tried to solve. In addition, batch normalizaion
    can also be added under normal usage to speed up training and
    improve model accuracy.

    Args:
        param1 (tensor): input_tensor
        param2 (bool): is_training
        param3 (str): scope

    Returns:
        tensor: output layer add the batch normaliazation function

    """

    # Selet batch nomalization is use for training or not traning
    return tf.cond(is_training,
                   lambda: tf.contrib.layers.batch_norm(input_tensor,
                                                        decay=0.99,
                                                        is_training=is_training,
                                                        center=True,
                                                        scale=True,
                                                        updates_collections=None,
                                                        scope=scope,
                                                        reuse=False),
                   lambda: tf.contrib.layers.batch_norm(input_tensor,
                                                        decay=0.99,
                                                        is_training=is_training,
                                                        center=True,
                                                        scale=True,
                                                        updates_collections=None,
                                                        scope=scope,
                                                        reuse=True))

def weight_variables(shape, name=None):
    """ Build normal distributions generator

    Output a random value from the truncated normal distribution.
    The resulting value follows a normal distribution with a specified mean
    and standard deviation, and discards the reselection if the resulting value
    is greater than the value of the mean 2 standard deviations.

    Args:
        param1 (list): shape
        param2 (str): name

    Returns:
        list: the tensorfolw veriable

    """

    # Use truncated_normal or random_normal to build
    initial = tf.truncated_normal(shape, mean=0.0, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variables(shape, name=None):
    """ Build constant generator

    This function add the initial bias 0.1 for each depth of tensor
    Output a constant variable

    Args:
        param1 (list): shape
        param2 (str): name

    Returns:
        list: the tensorfolw veriable

    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

########################
# SINGLE LAYER FUNCTIONS
########################

def convolution_1d_act(input_tensor,
                       prev_num_filters,
                       filter_size,
                       num_filters,
                       layer_number,
                       active_function,
                       stride=1,
                       padding='VALID',
                       tensorboard_output=False,
                       name=None):
    """ Build a single convolution layer with active function

    The function build a single convolution layer with intial weight and bias.
    Also add the active function for this single covolution layer

    Args:
        param1 (tensor): pre_tensor
        param2 (int): prev_num_filters
        param3 (int): filter_size
        param4 (int): num_filters
        param5 (funtion): active_function
        param6 (int): layer_number
        param7 (int): stride
        param8 (str): padding
        param9 (bool): tensorboard_output
        param10 (str): name

    Returns:
        tensor: representing the output of the operation

    """

    # Define the filter
    with tf.name_scope('{}_layer_conv_weights'.format(layer_number)):
        w = weight_variables(
            [filter_size, prev_num_filters, num_filters])

        if tensorboard_output:
            histogram_variable_summaries(w)

    # Define the bias
    with tf.name_scope('{}_layer_conv_biases'.format(layer_number)):
        b = bias_variables([num_filters])

        if tensorboard_output:
            histogram_variable_summaries(b)

    # Create the single convolution laryer
    with tf.name_scope('{}_layer_conv_preactivation'.format(layer_number)):
        conv = tf.nn.conv1d(input_tensor, w, stride=stride, padding=padding) + b

        if tensorboard_output:
            histogram_variable_summaries(conv)

    # Add the active function
    with tf.name_scope('{}_layer_conv_activation'.format(layer_number)):
        conv_act = active_function(conv, name=name)

        if tensorboard_output:
            histogram_variable_summaries(conv_act)
    return conv_act

def downsampling_d1_batch_norm_act(input_tensor,
                                   filter_size,                         
                                   layer_number,
                                   stride,
                                   active_function=tf.nn.relu,
                                   is_training=True,
                                   padding='VALID',
                                   tensorboard_output=False,
                                   num_filters=None, 
                                   name=None):
    """ Build a single downsampling layer

    The function build a single convolution layer with intial weight and bias.
    Also add  the batch normalization for this single convolution layer.
    Also add the active function for this single covolution layer.

    Args:
        param1 (tensor): input_tensor
        param2 (int): filter_size
        param3 (int): stride
        param4 (int): layer_number
        param5 (funtion): active_function
        param6 (bool): is_training
        param7 (int): num_filters
        param8 (str): padding
        param9 (bool): tensorboard_output
        param10 (str): name

    Returns:
        tensor: representing the output of the operation

    """

    # Assume this layer is twice the depth of the previous layer if no depth
    # information is given
    if num_filters is None:
        num_filters = 2 * input_tensor.get_shape().as_list()[-1]

    # Define the filter
    with tf.name_scope('{}_layer_conv_weights'.format(layer_number)):

        # input_tensor.get_shape().as_list()[-1] is pre_num_fitlters
        w = weight_variables(
            [filter_size, input_tensor.get_shape().as_list()[-1], num_filters])

        if tensorboard_output:
            histogram_variable_summaries(w)

    # Define the bias
    with tf.name_scope('{}_layer_conv_biases'.format(layer_number)):
        b = bias_variables([num_filters])

        if tensorboard_output:
            histogram_variable_summaries(b)

    # Create the single convolution laryer
    with tf.name_scope('{}_layer_conv_preactivation'.format(layer_number)):
        conv = tf.nn.conv1d(input_tensor, w, stride=stride, padding=padding) + b

        if tensorboard_output:
            histogram_variable_summaries(conv)

    # Add the batch nomalization at output of conlution laryer
    with tf.name_scope('{}_layer_batch_norm'.format(layer_number)) as scope:
        conv_batch_norm = batch_normalization(conv, is_training, scope)

    # Add the active function
    with tf.name_scope('{}_layer_conv_activation'.format(layer_number)):
        conv_batch_norm_act = active_function(conv_batch_norm, name=name)

        if tensorboard_output:
            histogram_variable_summaries(conv_batch_norm_act)
    return conv_batch_norm_act

def upsampling_d1_batch_normal_act_subpixel(input_tensor,
                                            residual_tensor,
                                            filter_size,              
                                            layer_number,
                                            active_function=tf.nn.relu,
                                            stride=1,
                                            is_training=True,
                                            padding='VALID',
                                            tensorboard_output=False,
                                            num_filters=None,
                                            name=None):
    """ Build a single upsampling layer

    The function build a single convolution layer with intial weight and bias.
    Also add  the batch normalization for this single convolution layer.
    Also add the active function for this single covolution layer.
    Also a subpixel convolution that reorders information along one
    dimension to expand the other dimensions.
    Also final convolutional layer with restacking and reordering operations
    is residually added to the original input to yield the upsampled waveform.

    Args:
        param1 (tensor): input_tensor
        param2 (tensor): residual_tensor
        param3 (int): filter_size
        param4 (int): stride
        param5 (int): layer_number
        param6 (funtion): active_function
        param7 (bool): is_training
        param8 (int): num_filters
        param9 (str): padding
        param10 (bool): tensorboard_output
        param11 (str): name

    Returns:
        tensor: representing the output of the operation

    """

    # assume this layer is half the depth of the previous layer if no depth
    # information is given
    if num_filters is None:
        num_filters = int(input_tensor.get_shape().as_list()[-1] / 2)

    # Define the filter
    with tf.name_scope('{}_layer_conv_weights'.format(layer_number)):

        # input_tensor.get_shape().as_list()[-1] is pre_num_fitlters
        w = weight_variables(
            [filter_size, input_tensor.get_shape().as_list()[-1], num_filters])

        if tensorboard_output:
            histogram_variable_summaries(w)

    # Define the bias
    with tf.name_scope('{}_layer_conv_biases'.format(layer_number)):
        b = bias_variables([num_filters])

        if tensorboard_output:
            histogram_variable_summaries(b)

    # Create the single convolution laryer
    with tf.name_scope('{}_layer_conv_preactivation'.format(layer_number)):
        conv = tf.nn.conv1d(input_tensor, w, stride=stride, padding=padding) + b

        if tensorboard_output:
            histogram_variable_summaries(conv)

    # Add the batch nomalization at output of conlution laryer
    with tf.name_scope('{}_layer_batch_norm'.format(layer_number)) as scope:
        conv_batch_norm = batch_normalization(conv, is_training, scope)

    # Add the active function
    with tf.name_scope('{}_layer_conv_activation'.format(layer_number)):
        conv_batch_norm_act = active_function(conv_batch_norm, name=name)

        if tensorboard_output:
            histogram_variable_summaries(conv_batch_norm_act)

    # Build a subpixel shuffling layer
    with tf.name_scope('{}_layer_subpixel_reshuffle'.format(layer_number)):
        subpixel_conv = subpixel_shuffling(
            conv_batch_norm_act,
            residual_tensor.get_shape().as_list()[-1],
            name=name)

        if tensorboard_output:
            histogram_variable_summaries(subpixel_conv)

    # In order to combined final stacking residual connections
    with tf.name_scope('{}_layer_stacking'.format(layer_number)):
        sliced = tf.slice(residual_tensor,
                          begin=[0, 0, 0],
                          size=[-1, subpixel_conv.get_shape().as_list()[1], -1])

        # Stack number of filters (the channels)
        stack_subpixel_conv = tf.concat((subpixel_conv, sliced),
                                         axis=2,
                                         name=name)

        if tensorboard_output:
            histogram_variable_summaries(stack_subpixel_conv)
    return stack_subpixel_conv

#################################
# AUDIO U-NET DEEP NEURAL NETWORK
#################################

def audio_u_net_dnn(input_type,
                    input_shape,
                    num_downsample_layers=8,
                    channel_multiple=8,
                    initial_filter_size=5,
                    initial_stride=2,
                    downsample_filter_size=3,
                    downsample_stride=2,
                    bottleneck_filter_size=4,
                    bottleneck_stride=2,
                    upsample_filter_size=3,
                    tensorboard_output=True,
                    scope_name='audio_u_net_dnn'):
    """ Construct the deep neural network (U-Net)

    The audio U-Net is based on the paper
    https://arxiv.org/abs/1708.00853

    Args:
        param1 (data type): input_type
        param2 (list): input_shape
        param3 (int): num_downsample_layers
        param4 (int): channel_multiple
        param5 (int): initial_filter_size
        param6 (int): initial_stride
        param7 (int): downsample_filter_size
        param8 (int): downsample_stride
        param9 (int): bottleneck_filter_size
        param10 (int): bottleneck_stride
        param11 (int): upsample_filter_size
        param10 (bool): tensorboard_output
        param11 (str): scope_name

    Returns:
        tensor: representing the output of the operation

    """
    
    print('The network summary for {}'.format(scope_name))
    
    # Create list to store layers
    downsample_layers = []
    upsample_layers = []
    layer_count = 1

    with tf.name_scope(scope_name):

        # is_training variable
        train_flag = tf.placeholder(tf.bool)

        # The first dimension of the placeholder is None, 
        # meaning we can have any number of rows.
        audio = [None]
        
        for n in input_shape:
            audio.append(n)

        x = tf.placeholder(input_type, shape=audio)

        # The last second value in list is input audio size
        input_size = audio[-2]

        # The last one vaule in list in input audio channels
        num_channels = audio[-1]

        print('    input: {}'.format(x.get_shape().as_list()[1:]))

        # Build the first downsampling layer
        d1 = downsampling_d1_batch_norm_act(
            x,
            filter_size=initial_filter_size,
            layer_number=layer_count,
            stride=initial_stride,
            is_training=train_flag,
            tensorboard_output=tensorboard_output,
            num_filters=channel_multiple * num_channels)
        downsample_layers.append(d1)

        print('    downsample layer: {}'.format(d1.get_shape().as_list()[1:]))

        layer_count += 1

        # Build next 7 downsampling layer
        for i in range(num_downsample_layers - 1):
            d = downsampling_d1_batch_norm_act(
                downsample_layers[-1],
                filter_size=downsample_filter_size,
                layer_number=layer_count,
                stride=downsample_stride,
                is_training=train_flag,
                tensorboard_output=tensorboard_output)
            downsample_layers.append(d)

            print('    downsample layer: {}'.format(d.get_shape().as_list()[1:]))

            layer_count += 1

        # Build one bottleneck layer
        b = downsampling_d1_batch_norm_act(
            downsample_layers[-1],
            filter_size=bottleneck_filter_size,
            layer_number=layer_count,
            stride=bottleneck_stride,
            is_training=train_flag,
            tensorboard_output=tensorboard_output)

        print('    bottleneck layer: {}'.format(b.get_shape().as_list()[1:]))

        layer_count += 1

        # Build the first upsampling layer
        u1 = upsampling_d1_batch_normal_act_subpixel(
            b,
            downsample_layers[-1],
            filter_size=upsample_filter_size,
            layer_number=layer_count,
            is_training=train_flag,
            tensorboard_output=tensorboard_output,
            num_filters=b.get_shape().as_list()[-1])
        upsample_layers.append(u1)

        print('    upsample layer: {}'.format(u1.get_shape().as_list()[1:]))

        layer_count += 1

        # Build the next 6 layer upsampling layer
        for i in range(num_downsample_layers - 2, -1, -1):
            u = upsampling_d1_batch_normal_act_subpixel(
                upsample_layers[-1],
                downsample_layers[i],
                filter_size=upsample_filter_size,
                layer_number=layer_count,
                is_training=train_flag,
                tensorboard_output=tensorboard_output)
            upsample_layers.append(u)

            print('    upsample layer: {}'.format(u.get_shape().as_list()[1:]))

            layer_count += 1

        # Build last restack layer to map downsampling layer
        target_size = int(input_size / initial_stride)
        restack = stacking_subpixel(
            upsample_layers[-1], target_size + (upsample_filter_size - 1))

        print('    restack layer: {}'.format(restack.get_shape().as_list()[1:]))

        # Add the convolution layer with restack layer
        conv_act = convolution_1d_act(
            restack,
            prev_num_filters=restack.get_shape().as_list()[-1],
            filter_size=upsample_filter_size,
            num_filters=initial_stride,
            layer_number=layer_count,
            active_function=tf.nn.elu,
            tensorboard_output=tensorboard_output)

        print('    final convolution layer: {}'.format(
            conv_act.get_shape().as_list()[1:]))

        # NOTE this effectively is a linear activation on the last conv layer
        subpixel_conv = subpixel_shuffling(conv_act, num_channels)
        y = tf.add(subpixel_conv, x, name=scope_name)

        print('    output: {}'.format(y.get_shape().as_list()[1:]))
        print('--------------------Finished model building--------------------')

    return train_flag, x, y
