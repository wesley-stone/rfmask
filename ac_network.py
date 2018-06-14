import threading
import multiprocessing
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
from helper import *
from collections import OrderedDict

from random import choice
from time import sleep
from time import time


VALUE_BETA = 0.5
ENTROPY_BETA = 0.01

class AC_Network():
    def __init__(self, img_size, scope,trainer):
        with tf.variable_scope(scope):
            # Input and visual encoding layers
            self.input_h = img_size[0]
            self.input_w = img_size[1]
            self.image = tf.placeholder(shape=[None,img_size[0], img_size[1], 3],dtype=tf.float32)  # [none, h, w, 3]
            self.mask = tf.placeholder(shape=[None,img_size[0]+2, img_size[1]+2],dtype=tf.float32)  # [none, h+2, w+2]
            tmp = tf.expand_dims(self.mask, -1)

            self.keep_prob = tf.placeholder(tf.float32)  # for dropout
            # remove edges in mask
            tmp = tf.image.resize_image_with_crop_or_pad(
                tmp, target_height=img_size[0], target_width=img_size[1])  # [none, h, w]
            self.imageIn = tf.concat([self.image, tmp], axis=-1)  # [none, h, w, 4]

            # Output layers for policy and value estimations
            mid_feature, self.confidence, self.prediction, self.offset = \
                self.__unet(self.imageIn, 4, keep_prob=self.keep_prob)
            self.confidence_flatten = slim.flatten(self.confidence)
            self.prediction_flatten = tf.reshape(self.prediction, shape=[-1, self.map_size[0]*self.map_size[1], 5])

            '''
            policy is a multi-channel map and contains the following things
            
            1. confidence, per point confidence to take action
            2. operation, add(1) or remove(0) a region
            3. lo_diff, param for Floodfill, [0, 1] representing [0, 255]
            4. up_diff, param for Floodfill, [0, 1] representing [0, 255]
            '''
            self.value = slim.fully_connected(
                mid_feature,
                1,
                activation_fn=None,
                weights_initializer=normalized_columns_initializer(1.0),
                biases_initializer=None)

            self.__get_loss(trainer, scope)

    def __get_loss(self, trainer, scope):
        if scope != 'global':
            self.actions = tf.placeholder(shape=[None, 4], dtype=tf.float32, name='actions')

            self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
            self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

            # log(pi)
            prob_point = tf.log(select(self.confidence_flatten,self.actions[:, 0], self.map_size[0]*self.map_size[1]))
            pred = select(self.prediction_flatten, self.actions[:, 0], self.map_size[0]*self.map_size[1])
            print(pred)
            prob_op = tf.log(tf.where(self.actions[:, 0]==1, pred[:, 2], pred[:, 3]))
            prob_lo = norm_probability(pred[:,1], pred[:,2], self.actions[:, 2])
            prob_up = norm_probability(pred[:,3], pred[:,4], self.actions[:, 3])
            self.responsible_outputs = prob_point + prob_op + prob_lo + prob_up

            # Loss functions
            self.value_loss = tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1])))
            self.entropy = - tf.reduce_sum(self.confidence * tf.log(self.confidence))
            self.policy_loss = -tf.reduce_sum(self.responsible_outputs * self.advantages)
            self.loss = VALUE_BETA * self.value_loss + self.policy_loss - ENTROPY_BETA * self.entropy

            # Get gradients from local network using local losses
            local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            self.gradients = tf.gradients(self.loss, local_vars)
            self.var_norms = tf.global_norm(local_vars)
            grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)

            # Apply local gradients to global network
            global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
            self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))

            '''
            # Only the worker network need ops for loss functions and gradient updating.
            self.actions = tf.placeholder(shape=[None],dtype=tf.float32)
            self.target_v = tf.placeholder(shape=[None],dtype=tf.float32)
            self.advantages = tf.placeholder(shape=[None],dtype=tf.float32)

            # Loss functions
            td_loss = self.target_v - tf.reshape(self.value, [-1])
            self.value_loss = tf.reduce_mean(tf.square(td_loss))
            # self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy))
            log_prob = self.normal_dist.log_prob(self.actions)
            self.policy_loss = -tf.reduce_mean(log_prob * self.advantages)
            self.entropy = tf.reduce_mean(self.normal_dist.entropy()) # encourage exploration
            self.loss = VALUE_BETA*self.value_loss + self.policy_loss - ENTROPY_BETA*self.entropy
            # self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs)*self.advantages)

            # Get gradients from local network using local losses
            local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            self.gradients = tf.gradients(self.loss,local_vars)
            self.var_norms = tf.global_norm(local_vars)
            grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,40.0)

            # Apply local gradients to global network
            global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
            self.apply_grads = trainer.apply_gradients(zip(grads,global_vars))
            '''

    def __unet(self, x, channels, keep_prob, layers=3, features_root=16, filter_size=3, pool_size=2):
        # Construct Unet for prediction

        unet_weights = []
        unet_biases = []
        convs = []
        pools = OrderedDict()
        deconv = OrderedDict()
        dw_h_convs = OrderedDict()
        up_h_convs = OrderedDict()
        in_node = x
        in_size = self.input_w
        size = in_size
        # down layers
        for layer in range(0, layers):
            features = 2 ** layer * features_root
            stddev = np.sqrt(2 / (filter_size ** 2 * features))
            if layer == 0:
                w1 = weight_variable([filter_size, filter_size, channels, features], stddev)
            else:
                w1 = weight_variable([filter_size, filter_size, features // 2, features], stddev)

            w2 = weight_variable([filter_size, filter_size, features, features], stddev)
            b1 = bias_variable([features])
            b2 = bias_variable([features])

            conv1 = conv2d(in_node, w1, keep_prob)
            tmp_h_conv = tf.nn.relu(conv1 + b1)
            conv2 = conv2d(tmp_h_conv, w2, keep_prob)
            dw_h_convs[layer] = tf.nn.relu(conv2 + b2)

            unet_weights.append((w1, w2))
            unet_biases.append((b1, b2))
            convs.append((conv1, conv2))

            size -= 4
            if layer < layers - 1:
                pools[layer] = max_pool(dw_h_convs[layer], pool_size)
                in_node = pools[layer]
                size /= 2

        mid_feature = dw_h_convs[layers - 1]
        # insert rnn step for time dependency
        len = size**2 * features
        mid_feature = self.__rnn(mid_feature, len)
        in_node = mid_feature

        # up layers
        for layer in range(layers - 2, -1, -1):
            features = 2 ** (layer + 1) * features_root
            stddev = np.sqrt(2 / (filter_size ** 2 * features))

            wd = weight_variable_devonc([pool_size, pool_size, features // 2, features], stddev)
            bd = bias_variable([features // 2])
            h_deconv = tf.nn.relu(deconv2d(in_node, wd, pool_size) + bd)
            h_deconv_concat = crop_and_concat(dw_h_convs[layer], h_deconv)
            deconv[layer] = h_deconv_concat

            w1 = weight_variable([filter_size, filter_size, features, features // 2], stddev)
            w2 = weight_variable([filter_size, filter_size, features // 2, features // 2], stddev)
            b1 = bias_variable([features // 2])
            b2 = bias_variable([features // 2])

            conv1 = conv2d(h_deconv_concat, w1, keep_prob)
            h_conv = tf.nn.relu(conv1 + b1)
            conv2 = conv2d(h_conv, w2, keep_prob)
            in_node = tf.nn.relu(conv2 + b2)
            up_h_convs[layer] = in_node

            unet_weights.append((w1, w2))
            unet_biases.append((b1, b2))
            convs.append((conv1, conv2))

            size *= 2
            size -= 4

        # Output Map

        weight = weight_variable([1, 1, features_root, 6], stddev)
        bias = bias_variable([6])
        conv = conv2d(in_node, weight, tf.constant(1.0))
        confidence, prediction = tf.split(conv + bias, [1, 5], -1)
        confidence = tf.exp(confidence)
        confidence = confidence/tf.reduce_sum(confidence, axis=[1,2])
        prediction = tf.nn.relu(prediction)
        offset = int(in_size - size)
        self.map_size = (self.input_h - offset, size)
        return mid_feature, confidence, prediction, offset

    def __rnn(self, hidden, len, size=256):
        shape = tf.shape(hidden)
        hidden = slim.flatten(hidden)
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
        rnn_out = tf.reshape(lstm_outputs, [-1, 256])
        rnn_out = slim.fully_connected(rnn_out, len, activation_fn=tf.nn.elu)
        rnn_out = tf.reshape(rnn_out, shape=shape)
        return rnn_out






















