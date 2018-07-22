import threading
import multiprocessing
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
from utils import *
from tf_utils import *
from collections import OrderedDict

from random import choice
from time import sleep
from time import time
from basenet.unet import create_conv_net


VALUE_BETA = 0.5
ENTROPY_BETA = 0.01
OP_BOUND = [0.0, 1.0]
LODIFF_BOUND = [0.01, 0.1]
UPDIFF_BOUND = [0.01, 0.1]


class AC_Network():
    def __init__(self, img_size, scope, trainer, img_channels=1, debug=True):
        self.debug = debug
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # Input and visual encoding layers
            with tf.variable_scope('input'):
                self.input_h = img_size[0]
                self.input_w = img_size[1]
                self.image = tf.placeholder(shape=[None,img_size[0], img_size[1]],
                                            dtype=tf.float32, name='input_image')  # [none, h, w]
                self.mask = tf.placeholder(shape=[None,img_size[0]+2, img_size[1]+2],
                                           dtype=tf.float32, name='input_mask')  # [none, h+2, w+2]
                self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')  # for dropout

                # remove edges in mask
                # tmp = tf.image.resize_image_with_crop_or_pad(
                    # tf.expand_dims(self.mask, axis=-1), target_height=img_size[0], target_width=img_size[1])  # [none, h, w]
                tmp = tf.slice(self.mask, [0, 1, 1], [-1, img_size[0], img_size[1]])
                self.imageIn = tf.stack([self.image, tmp], axis=-1, name='input')  # [none, h, w, 2]

            # unet with rnn for feature extraction and inference
            mid_feature, len, self.confidence, self.prediction = \
                self.__unet(self.imageIn, img_channels+1, keep_prob=self.keep_prob)

            # rnn to use mid_feature for value prediction
            mid_feature = self.__rnn(mid_feature, len)
            mid_feature = tf.identity(mid_feature, name='mid_feature')

            # Output layers for policy and value estimations
            '''
            policy is two maps (confidence, prediction) and contains the following things

            1. confidence, per point confidence to take action
            2. operation, add(1) or remove(0) a region
            3. lo_diff, param for Floodfill, [0, 1] representing [0, 255]
            4. up_diff, param for Floodfill, [0, 1] representing [0, 255]
            '''
            with tf.variable_scope("point_infer"):
                self.confidence_flatten = tf.layers.flatten(self.confidence, name='conf_flatten')

                self.prediction_flatten = tf.reshape(
                    self.prediction, shape=[-1, self.map_size[0]*self.map_size[1], 5], name='pred_flatten')

                self.value = slim.fully_connected(
                    slim.flatten(mid_feature),
                    1,
                    activation_fn=None,
                    weights_initializer=normalized_columns_initializer(1.0),
                    biases_initializer=None)

                self.point_samples = tf.multinomial(self.confidence_flatten, 1)[:, 0]
                point_pred = select(self.prediction_flatten, self.point_samples, self.map_size[0] * self.map_size[1])
                op_pred = point_pred[:, 0]
                op_prob = tf.stack([op_pred, 1-op_pred], axis=1)

                self.inf_op = tf.multinomial(op_prob, 1)[:, 0]
                self.lo_dist = tf.distributions.Normal(point_pred[:, 1], point_pred[:, 2])
                self.inf_lo = tf.clip_by_value(self.lo_dist.sample(1)[:, 0], LODIFF_BOUND[0], LODIFF_BOUND[1])
                self.up_dist = tf.distributions.Normal(point_pred[:, 3], point_pred[:, 4])
                self.inf_up = tf.clip_by_value(self.up_dist.sample(1)[:, 0], UPDIFF_BOUND[0], UPDIFF_BOUND[1])

            self.__get_loss(trainer, scope)

    def __get_loss(self, trainer, scope):
        if scope == 'global':
            return
        # for worker scope
        with tf.variable_scope('loss'):
            self.actions = tf.placeholder(shape=[None, 4], dtype=tf.float32, name='actions')

            self.target_v = tf.placeholder(shape=[None], dtype=tf.float32, name='targe_v')
            self.advantages = tf.placeholder(shape=[None], dtype=tf.float32, name='advantages')

            # log(pi)
            # print(self.actions[:, 0])
            prob_point = tf.log(select(self.confidence_flatten,self.actions[:, 0], self.map_size[0]*self.map_size[1]))
            pred = select(self.prediction_flatten, self.actions[:, 0], self.map_size[0]*self.map_size[1])
            prob_op = tf.log(tf.where(self.actions[:, 0]==1, pred[:, 0], 1-pred[:, 0]))
            prob_lo = self.lo_dist.log_prob(self.actions[:, 2])
            prob_up = self.up_dist.log_prob(self.actions[:, 3])
            self.responsible_outputs = prob_point + prob_op + prob_lo + prob_up

            # Loss functions
            self.value_loss = tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1])), name='value_loss')
            self.entropy = - tf.reduce_sum(self.confidence * tf.log(self.confidence), name='entropy_loss')
            self.policy_loss = -tf.reduce_sum(self.responsible_outputs * self.advantages, name='policy_loss')
            self.loss = tf.add_n((VALUE_BETA * self.value_loss, self.policy_loss , -ENTROPY_BETA * self.entropy), name='loss')

            # Get gradients from local network using local losses
            local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            self.gradients = tf.gradients(self.loss, local_vars)
            self.var_norms = tf.global_norm(local_vars)
            grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)

            # Apply local gradients to global network
            global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
            self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))

    def __unet(self, x, channels, keep_prob, layers=3, features_root=16, filter_size=3,
               pool_size=2, summaries=True):
        with tf.name_scope("unet"):
            output, mid_feature, vars, size, mid_shape = create_conv_net(x, keep_prob, channels, 6, layers,
                                                                         features_root, filter_size, pool_size,
                                                                         summaries, debug=self.debug)
        self.unet_var_dict = dict()
        for v in vars:
            self.unet_var_dict[v.name] = v
        self.map_size = (size[0], size[1])
        with tf.name_scope("map_infer"):
            conf, pred = tf.split(output, [1, 5], -1)
            conf = tf.exp(conf)
            conf = conf/tf.reduce_sum(conf, axis=[1,2], keep_dims=True)
            pred = tf.nn.sigmoid(pred)
            mid_feature = tf.layers.flatten(mid_feature)
            len = int(mid_shape[0] * mid_shape[1] * mid_shape[2])
            return mid_feature, len, conf, pred

    def __rnn(self, hidden, len, size=64):
        with tf.variable_scope('rnn'):
            shape = tf.shape(hidden)
            hidden = slim.fully_connected(hidden, size, activation_fn=tf.nn.elu)
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(size , state_is_tuple=True)
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = (c_in, h_in)
            rnn_in = tf.expand_dims(hidden, [0])
            step_size = tf.shape(self.imageIn)[:1]
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
                time_major=False)
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_out = tf.reshape(lstm_outputs, [-1, size])
            rnn_out = slim.fully_connected(rnn_out, len, activation_fn=tf.nn.elu)
            # rnn_out = tf.reshape(rnn_out, shape=shape)
            return rnn_out

