# -*- coding: utf-8 -*-
""" Optimizer

This module can use for training as the libary. You don't need modify any
functions in the libary. Please see the description in front of each function
if you don't undersand it. Please import this module if you want use this libary
in anthor project. The mould are assumed to already be preprocessed and spliced
into clips and split into a training, testing, and validation set.
These sets are stored as csv files which are later read in in batches with
functions in this module.

################################################################################
# Author: Weikun Han <weikunhan@gmail.com>
# Crate Date: 03/10/2018
# Update:
# Reference: https://github.com/jhetherly/EnglishSpeechUpsampler
################################################################################
"""

import tensorflow as tf

def learing_rate_scheduling(init_learing_rate,
                            decay_steps,
                            decay_factor,
                            staircase=False,
                            exp_decay_flag=False):
    """ Build a exponential scheduling for the learning rate

    This function can output difference learning rate base on difference input
    settings.

    Args:
        param1 (int): init_learing_rate
        param2 (int): decay_steps
        param3 (int): decay_factor
        param4 (bool): staircase
        param5 (bool): exp_decay_flag

    Returns:
        tensor: the tensor represent special changed learning rate

    """

    # If trainable is True the variable is also added to the graph
    # collection GraphKeys.TRAINABLE_VARIABLES.
    global_step = tf.Variable(0, trainable=False)

    with tf.name_scope('learning_rate'):
        if exp_decay_flag:
            learing_rate = tf.train.exponential_decay(init_learing_rate,
                                                      global_step,
                                                      decay_steps,
                                                      decay_factor,
                                                      staircase=staircase)
        else:
            learing_rate = tf.train.inverse_time_decay(init_learing_rate,
                                                       global_step,
                                                       decay_steps,
                                                       decay_factor,
                                                       staircase=staircase)
    tf.summary.scalar('learning_rate', learing_rate)
    return learing_rate, global_step

def setup_optimizer(learing_rate,
                    loss_funciton,
                    optimizer,
                    batch_norm_flag=True,
                    optimizer_args={},
                    minimize_args={}):
    """ The functon helper to setup the optimizer

    The function can output two optimizer one for batch normalizaion and another
    is not use batch normalization. To disable batch normalization, just set
    batch_norm_flag=False

    Args:
        param1 (tensor): learing_rate
        param2 (tensor): loss_funciton
        param3 (tensor): optimizer
        param4 (bool): batch_norm_flag
        param5 (directory): opttimizer_args
        param6 (directory): minimize_args

    Returns:
        tesor: return the tensor as the customized optimizer define by
               difference input setting

    """
    if batch_norm_flag:

        # Ensures that we execute the update_ops before performing
        # the train_step (for batch normalization)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.name_scope('train'):
                train_step = optimizer(
                    learing_rate, **optimizer_args).minimize(loss_funciton,
                                                             **minimize_args)
    else:
        with tf.name_scope('train'):
            train_step = optimizer(
                learing_rate, **optimizer_args).minimize(loss_funciton,
                                                         **minimize_args)
    return train_step
