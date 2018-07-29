import numpy as np
import cv2
import matplotlib;matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue
from reward import *
from cycle.cycle_reword import cycleReward
from dataset import *
from utils import *


class Painter(object):
    def __init__(self, name, data_type='coco', *args):
        self.name = name
        self.data_type = data_type
        self._display = False
        self._state_change = False
        if self.data_type == 'coco':
            '''
            args[0] is the location of annotation json file
            args[1] is the directory of image files
            '''
            self.reward = cocoReward(args[0])
            self.dataset = cocoDataSet(args[1])
        elif self.data_type == 'cycle':
            self.reward = cycleReward(args[0])
            self.reward.size = (256, 256)
        self.counts = 0
        self.queue = Queue()

    def load_resize_img(self, rimg):
        self.size = rimg.shape
        self.observation = np.zeros((2, self.size[0], self.size[1]), np.float32)
        self.observation[0] = rimg/255.0

    def step(self, operation):
        if operation is not None:
            h1, w1 = operation.shape
            h_offset, w_offset = (self.size[0]-h1)//2, (self.size[1]-w1)//2
            # copy action
            self.observation[1, h_offset:-h_offset, w_offset:-w_offset] = operation
        r_, done = self.__reward()
        return self.observation, r_, done, 'info'

    def reset(self):
        if self.data_type == 'cycle':
            img, _ = self.reward.next_episode()
            self.load_resize_img(img)
            return self.observation
        return None

    def __reward(self):
        if self.data_type == 'cycle':
            return self.reward.get_reward(self.observation[1])
        return 0.1, False

    def display(self):
        # start another process for rendering
        if self._display:
            # if already start a process
            return
        self._display = True
        self._state_change = True
        self.rend_process = Process(target=_render_func, args=(self.queue,))
        self.rend_process.start()

    def render(self):
        if self.queue.full():
            self.queue.get_nowait()
        self.queue.put((self.observation, self.reward.get_gt()))


def _render_func(queue):
    '''
    one process for rendering image
    :param queue: GPainter's data queue
    :return:
    '''
    plt.ion()
    fig = plt.figure()
    observation, gt = queue.get()
    com_pred = combine_to_display(observation[0], gt, observation[1])
    im = plt.imshow(com_pred, cmap=plt.get_cmap('gray'))

    while True:
        observation, gt = queue.get()
        com_pred = combine_to_display(observation[0], gt, observation[1])
        im.set_data(com_pred)
        fig.canvas.draw()
        plt.pause(0.1)

