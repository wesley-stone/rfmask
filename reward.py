from __future__ import print_function
from pycocotools.coco import COCO
from pycocotools.mask import *
import os, sys, zipfile
import urllib.request
import shutil
import numpy as np
import skimage.io as io
from skimage.transform import resize
import pylab

class Reward(object):
    def get_reward(self, *args):
        raise NotImplementedError

    def next_episode(self, *args):
        raise NotImplementedError


class cocoReward(Reward):
    def __init__(self, ann_path):
        self.ann_path = ann_path
        self.coco = COCO(ann_path)
        self.cur_ins_index = -1
        self.cur_score = 0

    def next_episode(self, *args):
        img_id = args[0]
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        tmp = self.coco.loadImgs(img_id)[0]
        self.shape = (tmp['height'], tmp['width'])
        self.anns = self.coco.loadAnns(ann_ids)
        print('%d instance in mask gt of %d' % (len(self.anns), img_id))
        self.rles = [self.coco.annToRLE(ann) for ann in self.anns]
        # self.rle_states = np.zeros(len(self.anns))
        # denote current detected mask
        self.cur_ins_index = -1
        self.cur_score = 0

    def get_reward(self, *args):
        '''

        :param args: args[0] is current mask
        :return: score increment and possibility that whether the
        '''
        # just return the iou change
        mask = args[0]
        mask = resize(mask, self.shape).astype(np.uint8)
        rle = encode(np.asfortranarray(mask))
        if self.cur_ins_index == -1:
            # choose the instance with maximum iou
            scores = iou(self.rles, rle, np.zeros(len(self.rles)))
            self.cur_ins_index = np.argmax(scores)
            self.cur_score = scores[self.cur_ins_index][0]
            return self.cur_score, False
        scores = iou([self.rles[self.cur_ins_index]], rle, np.zeros(len(self.rles)))
        incre = scores[0][0]-self.cur_score
        self.cur_score = scores[0][0]
        print('score %f'%self.cur_score)

        instance_stop = self.cur_score>=0.8  # sufficiently good
        if instance_stop:
            # do not detect this again
            del self.rles[self.cur_ins_index]
            self.cur_ins_index = -1

        return incre, self.cur_score>=0.8


    def get_reward2(self, *args):
        mask = args[0]



