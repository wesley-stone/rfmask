import threading
import multiprocessing
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
from ac_network import AC_Network

from random import choice
from time import sleep
from time import time
from helper import *
from painter import *

OP_BOUND = [0.0, 1.0]
LODIFF_BOUND = [0.01, 0.1]
UPDIFF_BOUND = [0.01, 0.1]


class Worker(object):
    def __init__(self, name,s_size,trainer,model_path,global_episodes, data_type, data_path, max_episode_num = 10):
        self.name = "worker_" + str(name)
        self.number = name        
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter("train_"+str(self.number))
        self.input_size = s_size
        self.max_episode_num = max_episode_num
        self.img_stack = None

        #Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = AC_Network(s_size,self.name,trainer)
        self.update_local_ops = update_target_graph('global',self.name)        
        
        #The Below code is related to setting up the environment
        # for coco
        # self.env = Painter("painter" + str(name), data_type,
                           #'%s/annotations/instances_val2017.json'%data_path,
                            #'%s/val2017/'%data_path)
        self.env = Painter('painter %d'%name, data_type, data_path)
        #End set-up

    def train(self,rollout,sess,gamma,bootstrap_value):
        rollout = np.array(rollout)
        img = self.env.get_image()
        observations = rollout[:,0]
        actions = rollout[:,1]
        print()
        rewards = rollout[:,2]
        next_observations = rollout[:,3]
        values = rollout[:,5]

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus,gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages,gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        image_tmp = np.repeat(self.env.get_image()[np.newaxis, :, :], len(advantages), axis=0)
        feed_dict = {
            self.local_AC.target_v:discounted_rewards,
            self.local_AC.image: image_tmp,
            self.local_AC.mask: np.stack(observations, axis=0),
            self.local_AC.keep_prob: 0.9,
            self.local_AC.actions: np.vstack(actions),
            self.local_AC.advantages:advantages,
            self.local_AC.state_in[0]:self.batch_rnn_state[0],
            self.local_AC.state_in[1]:self.batch_rnn_state[1]}
        v_l,p_l,e_l,g_n,v_n, self.batch_rnn_state,_ = sess.run(
            [self.local_AC.value_loss,
                self.local_AC.policy_loss,
                self.local_AC.entropy,
                self.local_AC.grad_norms,
                self.local_AC.var_norms,
                self.local_AC.state_out,
                self.local_AC.apply_grads],
            feed_dict=feed_dict)
        return v_l ,p_l, e_l, g_n, v_n
        
    def work(self,max_episode_length,gamma,sess,coord,saver):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print ("Starting worker " + str(self.number))
        v_l, p_l, e_l, g_n, v_n = 0,0,0,0,0
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_frames = []
                episode_reward = 0
                episode_step_count = 0
                d = False
                
                self.env.new_episode(self.input_size)
                # self.env.show()

                s = self.env.get_state()
                episode_frames.append(s)
                # s = process_frame(s)
                rnn_state = self.local_AC.state_init
                self.batch_rnn_state = rnn_state
                while self.env.is_episode_finished() == False:
                    # Take an action using probabilities from policy network output.
                    print('another step!')
                    pos, v, op, lo, up, rnn_state = sess.run(
                        [self.local_AC.point_samples,
                         self.local_AC.value,
                         self.local_AC.inf_op,
                         self.local_AC.inf_lo,
                         self.local_AC.inf_up,
                         self.local_AC.state_out,],
                        feed_dict={self.local_AC.mask: [s],
                                   self.local_AC.image: [self.env.get_image()],
                                   self.local_AC.keep_prob: 1.0,
                                   self.local_AC.state_in[0]: rnn_state[0],
                                   self.local_AC.state_in[1]: rnn_state[1]})
                    # y, x, pos, operation, loDiff, upDiff = self.__get_action(conf_f, pred)
                    _,_,y,x = self.__fold(pos)
                    a = [pos[0], op[0], lo[0], up[0]]
                    r, d = self.env.make_action(op[0], (y[0], x[0]), lo[0], up[0])
                    print('reward %f'%r)

                    if not d:
                        s1 = self.env.get_state()
                        episode_frames.append(s1)
                        # s1 = process_frame(s1)
                    else:
                        s1 = s

                    episode_buffer.append([s,a,r,s1,d,v[0,0]])
                    episode_values.append(v[0,0])

                    episode_reward += r
                    s = s1                    
                    total_steps += 1
                    episode_step_count += 1
                    
                    # If the episode hasn't ended, but the experience buffer is full, then we
                    # make an update step using that experience rollout.
                    if len(episode_buffer) == self.max_episode_num \
                            and d != True and episode_step_count != max_episode_length - 1:
                        # Since we don't know what the true final return is, we "bootstrap" from our current
                        # value estimation.
                        v1 = sess.run(
                            self.local_AC.value,
                            feed_dict={
                                self.local_AC.mask:[s],
                                self.local_AC.image:[self.env.get_image()],
                                self.local_AC.keep_prob: 1.0,
                                self.local_AC.state_in[0]:rnn_state[0],
                                self.local_AC.state_in[1]:rnn_state[1]})[0,0]
                        v_l,p_l,e_l,g_n,v_n = self.train(episode_buffer,sess,gamma,v1)
                        episode_buffer = []
                        sess.run(self.update_local_ops)
                    if d:
                        break
                if len(episode_values) == 0:
                    continue
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))
                
                # Update the network using the episode buffer at the end of the episode.
                if len(episode_buffer) != 0:
                    v_l,p_l,e_l,g_n,v_n = self.train(episode_buffer,sess,gamma,0.0)
                                
                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if episode_count % 5 == 0 and episode_count != 0:
                    if self.name == 'worker_0' and episode_count % 25 == 0:
                        time_per_step = 0.05
                        images = np.array(episode_frames)
                        # make_gif(images,'./frames/image'+str(episode_count)+'.gif',
                           # duration=len(images)*time_per_step,true_image=True,salience=False)
                    if episode_count % 5 == 0 and self.name == 'worker_0':
                        saver.save(sess,self.model_path+'/model-'+str(episode_count)+'.cptk')
                        print ("Saved Model")

                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                    summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                    summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                    summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                    summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                    self.summary_writer.add_summary(summary, episode_count)
                    self.summary_writer.add_graph(tf.get_default_graph())

                    self.summary_writer.flush()
                if self.name == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1

    def __get_action(self, conf_f, pred):
        print(conf_f[0])
        pos = np.random.choice(len(conf_f[0]), p=conf_f[0])
        print(pos)
        y, x, yn, xn = self.__fold(pos)
        pred = pred[0][y][x]
        operation = np.clip(np.random.choice(2, p=(pred[0], 1 - pred[0])), OP_BOUND[0], OP_BOUND[1])
        loDiff = np.clip(np.random.normal(pred[1], pred[2]), LODIFF_BOUND[0], LODIFF_BOUND[1])
        upDiff = np.clip(np.random.normal(pred[3], pred[4]), UPDIFF_BOUND[0], UPDIFF_BOUND[1])
        return yn, xn, pos, operation, loDiff, upDiff

    def __fold(self, pos):
        h, w = self.env.h-self.local_AC.offset, self.env.w - self.local_AC.offset
        y, x = pos//w, pos%w
        return y, x, y*1.0/h, x*1.0/w
