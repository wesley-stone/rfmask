import os
import tensorflow as tf
from ga3c.Config import Config

from utils import *
from tf_utils import *
from basenet.unet import create_conv_net
import re
import basenet.transfer as transfer

class NetworkVP:
    def __init__(self, device, model_name, debug=False):
        self.device = device
        self.model_name = model_name

        self.img_width = Config.IMAGE_WIDTH
        self.img_height = Config.IMAGE_HEIGHT
        self.img_channels = Config.IMAGE_CHANNEL

        self.learning_rate = Config.LEARNING_RATE_START
        self.beta = Config.BETA_START
        self.log_epsilon = Config.LOG_EPSILON

        self.debug = debug
        self.counts = 0

        self.graph = tf.Graph()
        with self.graph.as_default() as g:
            with tf.device(self.device):
                self._create_graph()

            self.sess = tf.Session(
                graph=self.graph,
                config=tf.ConfigProto(
                    allow_soft_placement=True,
                    log_device_placement=False,
                    gpu_options=tf.GPUOptions(allow_growth=True)))
            self.sess.run(tf.global_variables_initializer())

            if Config.TENSORBOARD: self._create_tensor_board()
            if Config.LOAD_CHECKPOINT or Config.SAVE_MODELS:
                vars = tf.global_variables()
                self.saver = tf.train.Saver({var.name: var for var in vars}, max_to_keep=0)

    def _create_graph(self):
        with tf.variable_scope('global', reuse=tf.AUTO_REUSE):
            self.var_learning_rate = tf.placeholder(tf.float32, name='lr', shape=[])
            self.global_step = tf.Variable(0, trainable=False, name='step')
            # Input and visual encoding layers
            with tf.variable_scope('input'):
                self.x = tf.placeholder(shape=[None,self.img_height, self.img_width, self.img_channels],
                                            dtype=tf.float32, name='input_image')  # [none, h, w]
                self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')  # for dropout


            # unet with rnn for feature extraction and inference
            mid_feature, len, self.confidence = \
                self.__unet(self.x, 2, keep_prob=self.keep_prob)

            # rnn to use mid_feature for value prediction
            '''
            mid_feature = self.__rnn(mid_feature, len)
            mid_feature = tf.identity(mid_feature, name='mid_feature')
            '''

            '''
            policy is one map representing the points' possibility of adding to mask
            '''
            with tf.variable_scope("point_infer"):
                # random sample from map
                self.confidence_flatten = tf.layers.flatten(self.confidence, name='conf_flatten')
                map_shape = tf.shape(self.confidence_flatten)
                ones = tf.ones(map_shape, dtype=tf.uint8)
                zeros = tf.zeros(map_shape, dtype=tf.uint8)
                uniform_sample = tf.random_uniform(map_shape, dtype=tf.float32)
                mask_tmp = tf.where(uniform_sample < self.confidence_flatten, ones, zeros)
                self.mask_sample = tf.reshape(mask_tmp, shape=[-1, self.map_size[0], self.map_size[1]])

                v = slim.fully_connected(
                    slim.flatten(mid_feature),
                    1,
                    activation_fn=None,
                    weights_initializer=normalized_columns_initializer(1.0),
                    biases_initializer=None)
                self.value = tf.squeeze(v, axis=1)

            self.__get_loss()
        return

    def __train_ops(self):
        if Config.DUAL_RMSPROP:
            self.opt_p = tf.train.RMSPropOptimizer(
                learning_rate=self.var_learning_rate,
                decay=Config.RMSPROP_DECAY,
                momentum=Config.RMSPROP_MOMENTUM,
                epsilon=Config.RMSPROP_EPSILON)

            self.opt_v = tf.train.RMSPropOptimizer(
                learning_rate=self.var_learning_rate,
                decay=Config.RMSPROP_DECAY,
                momentum=Config.RMSPROP_MOMENTUM,
                epsilon=Config.RMSPROP_EPSILON)
        else:
            self.cost_all = self.cost_p + self.cost_v
            self.opt = tf.train.RMSPropOptimizer(
                learning_rate=self.var_learning_rate,
                decay=Config.RMSPROP_DECAY,
                momentum=Config.RMSPROP_MOMENTUM,
                epsilon=Config.RMSPROP_EPSILON)

        if Config.USE_GRAD_CLIP:
            if Config.DUAL_RMSPROP:
                self.opt_grad_v = self.opt_v.compute_gradients(self.cost_v)
                self.opt_grad_v_clipped = [(tf.clip_by_norm(g, Config.GRAD_CLIP_NORM), v)
                                           for g, v in self.opt_grad_v if not g is None]
                self.train_op_v = self.opt_v.apply_gradients(self.opt_grad_v_clipped)

                self.opt_grad_p = self.opt_p.compute_gradients(self.cost_p)
                self.opt_grad_p_clipped = [(tf.clip_by_norm(g, Config.GRAD_CLIP_NORM), v)
                                           for g, v in self.opt_grad_p if not g is None]
                self.train_op_p = self.opt_p.apply_gradients(self.opt_grad_p_clipped)
                self.train_op = [self.train_op_p, self.train_op_v]
            else:
                self.opt_grad = self.opt.compute_gradients(self.cost_all)
                self.opt_grad_clipped = [(tf.clip_by_average_norm(g, Config.GRAD_CLIP_NORM), v) for g, v in
                                         self.opt_grad]
                self.train_op = self.opt.apply_gradients(self.opt_grad_clipped)
        else:
            if Config.DUAL_RMSPROP:
                self.train_op_v = self.opt_p.minimize(self.cost_v, global_step=self.global_step)
                self.train_op_p = self.opt_v.minimize(self.cost_p, global_step=self.global_step)
                self.train_op = [self.train_op_p, self.train_op_v]
            else:
                self.train_op = self.opt.minimize(self.cost_all, global_step=self.global_step)

    def __get_loss(self):
        with tf.variable_scope('loss'):
            self.actions = tf.placeholder(shape=[None, self.map_size[0], self.map_size[1]], dtype=tf.uint8,
                                          name='actions')

            self.y_r = tf.placeholder(shape=[None,], dtype=tf.float32, name='target_v')

            negate_confidence_flatten = 1 - self.confidence_flatten
            self.response = tf.where(self.actions == 1, self.confidence_flatten, negate_confidence_flatten)
            self.responsible_outputs = tf.reduce_sum(tf.log(self.response))

            # Loss functions
            self.cost_v = tf.reduce_sum(tf.square(self.y_r - self.value), name='value_loss')

            self.entropy = tf.reduce_sum(self.confidence * tf.log(self.confidence), name='entropy_loss')
            self.policy_loss = - tf.reduce_sum(self.responsible_outputs * (self.y_r - tf.stop_gradient(self.value)),
                                               name='policy_loss')
            self.cost_p = tf.add(self.policy_loss, -Config.ENTROPY_BETA*self.entropy)

            self.__train_ops()

    def __unet(self, x, channels, keep_prob, layers=3, features_root=16, filter_size=3,
               pool_size=2, summaries=True):
        with tf.name_scope("unet"):
            output, mid_feature, vars, size, mid_shape = create_conv_net(x, keep_prob, channels, 2, layers,
                                                                         features_root, filter_size, pool_size,
                                                                         summaries, x_size=(self.img_height, self.img_width),
                                                                         debug=self.debug)
            mid_feature = tf.layers.flatten(mid_feature)
            self.origin_output = output
            pred_map = pixel_wise_softmax(output)[..., 1]

        len = int(mid_shape[0] * mid_shape[1] * mid_shape[2])
        self.unet_var_dict = dict()
        for v in vars:
            self.unet_var_dict[v.name] = v
        self.map_size = (size[0], size[1])
        return mid_feature, len, pred_map

    def __rnn(self, hidden, len, size=64):
        with tf.variable_scope('rnn'):
            hidden = slim.fully_connected(hidden, size, activation_fn=tf.nn.elu)
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(size , state_is_tuple=True)
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = (c_in, h_in)
            rnn_in = tf.expand_dims(hidden, [0])
            step_size = tf.shape(self.x)[:1]
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

    def predict_p_and_v(self, x):
        feed_dict = self.__get_base_feed_dict(is_train=False)
        feed_dict.update({self.x: x})
        conf, sample, value = self.sess.run([self.confidence, self.mask_sample, self.value], feed_dict=feed_dict)
        # save_image(conf, 'img_tmp/conf/confidence_%d.jpg'%self.counts)
        # save_image(sample, 'img_tmp/sample/sample_%d.jpg'%self.counts)
        self.counts += 1
        return sample, value

    def __get_base_feed_dict(self, is_train=False):
        prob = Config.KEEP_PROB if is_train else 1.0
        return {self.var_learning_rate: self.learning_rate,
                self.keep_prob: prob,}

    def train(self, x, y_r, a, trainer_id):
        feed_dict = self.__get_base_feed_dict()
        # print(y_r)
        feed_dict.update({self.y_r:y_r,
                          self.x:x,
                          self.actions: a})
        self.sess.run(self.train_op, feed_dict=feed_dict)
        return

    def _get_episode_from_filename(self, filename):
        return int(re.split('/|_|\.', filename)[2])

    def _checkpoint_filename(self, episode):
        return 'checkpoints/%s_%08d' % (self.model_name, episode)

    def save(self, episode):
        self.saver.save(self.sess, self._checkpoint_filename(episode))

    def load(self):
        # load from pre_trained model or previous model
        filename = tf.train.latest_checkpoint(os.path.dirname(self._checkpoint_filename(episode=0)))
        if Config.LOAD_EPISODE > 0:
            filename = self._checkpoint_filename(Config.LOAD_EPISODE)
        self.saver.restore(self.sess, filename)
        return self._get_episode_from_filename(filename)

    def pre_train(self):
        inited = transfer.param_transfer(self.sess, Config.BASE_MODEL_PATH,self.unet_var_dict)
        transfer.guarantee_initialized_variables(self.sess, inited)

    def _create_tensor_board(self):
        # write summary
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        summaries.append(tf.summary.scalar('Pcost_advantage', self.policy_loss))
        summaries.append(tf.summary.scalar('Pcost_entropy', self.entropy))
        summaries.append(tf.summary.scalar('Pcost', self.cost_p))
        summaries.append(tf.summary.scalar('Vcost', self.cost_v))
        summaries.append(tf.summary.scalar('LearningRate', self.var_learning_rate))
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram('weights_%s'%var.name, var))

        self.summary_op = tf.summary.merge(summaries)
        self.log_writer = tf.summary.FileWriter('logs/%s'%self.model_name, self.sess.graph)


