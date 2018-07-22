import numpy as np
import cv2
from reward import *
from cycle.cycle_reword import cycleReward
from dataset import *
import time
from utils import *

class Painter(object):
    def __init__(self, name, data_type='coco', *args):
        self.name = name
        self.data_type = data_type
        self._display = False
        if self.data_type == 'coco':
            '''
            args[0] is the location of annotation json file
            args[1] is the directory of image files
            '''
            self.reward = cocoReward(args[0])
            self.dataset = cocoDataSet(args[1])
        elif self.data_type == 'cycle':
            self.reward = cycleReward(args[0])

    def load(self, path):
        # self.img = np.zeros(shape=(self.size[0], self.size[1], 3), dtype=np.uint8)
        tmp = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        self.load_img(tmp)

    def load_img(self, img):
        self.origin = img
        img = np.expand_dims(cv2.resize(self.origin, dsize=(self.size[0], self.size[1])), -1)
        self.load_resize_img(img)

    def load_resize_img(self, rimg):
        self.img = rimg
        self.h, self.w = self.size[0], self.size[1]
        self.regularized_img = rimg.astype(dtype=np.float32)/255
        self.mask = np.zeros(self.size, dtype=np.uint8)  # 0 means unmask, 255 means mask

    def make_action(self, operation):
        h1, w1 = operation.shape
        h_offset, w_offset = (self.size[0]-h1)//2, (self.size[1]-w1)//2
        # copy action
        self.mask[h_offset:-h_offset, w_offset:-w_offset] = operation

        if self._display:
            self.refresh()
        return self.__reward()

    def new_episode(self, size):
        if self.data_type == 'coco':
            self.size = size
            print('another episode!')
            # self.img_path, self.img_id = self.dataset.next()
            self.img_path, self.img_id = 'D:/Datasets/coco/val2017/000000532493.jpg', 532493
            self.load(self.img_path)
            self.reward.next_episode(self.img_id)
            return
        elif self.data_type == 'cycle':
            self.size = size
            self.reward.size = size
            img, _ = self.reward.next_episode()
            self.load_resize_img(img)

    def get_state(self):
        return self.mask

    def get_image(self):
        return self.regularized_img

    def __reward(self):
        if self.data_type == 'coco':
            return self.reward.get_reward(self.mask)
        elif self.data_type == 'cycle':
            return self.reward.get_reward(self.mask)
        return 0.1, False

    def onMouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            print((x,y))
            # self.make_action(1, (0.222, 0.077), 0.1, 0.1)

            print(self.mask)

    def display(self):
        self._display = True
        cv2.destroyWindow('prediction')
        cv2.namedWindow('prediction')
        self.refresh()

    def refresh(self):
        com_pred = combine_to_display(self.regularized_img, self.reward.get_gt(), self.mask)
        cv2.imshow('prediction', com_pred)
        cv2.waitKey(10)

    def show(self):
        cv2.destroyWindow('demo')
        cv2.namedWindow('demo')
        cv2.setMouseCallback('demo', self.onMouse)
        while (1):
            cv2.imshow('demo', self.regularized_img)
            cv2.imshow('mask', self.mask*255)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

if __name__ == '__main__':
    p = Painter('painter0', 'cycle', 'D:\\Datasets\\cycle')
    p.size = (512, 512)
    p.new_episode(p.size)
    print(p.regularized_img)
    print(p.reward.get_gt())
    print(p.mask)
    p.display()
    '''
    data_path= 'D:/Datasets/coco'
    p = Painter("painter0", 'coco',
                           '%s/annotations/instances_val2017.json'%data_path,
                            '%s/val2017/'%data_path)
    p.new_episode((512, 512))
    print(p.regularized_img*255)
    p.show()

    '''

