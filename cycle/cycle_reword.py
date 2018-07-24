import reward
import os
import numpy as np
import cv2
from cycle import cycle_util


class cycleReward(reward.Reward):

    def __init__(self, root_path, size=None, shuffle = 1000):
        self.path = root_path
        self.img_path = os.path.join(self.path, 'img')
        self.ann_path = os.path.join(self.path, 'ann')
        self.id_path = os.path.join(self.path, 'img/data.txt')
        with open(self.id_path, 'r') as f:
            ids = f.readlines()
            self.ids = np.array([int(id) for id in ids])
        self.num = len(self.ids)
        self.pool = np.random.choice(self.ids, shuffle)
        self.shuffle = shuffle
        self.cur_ins_index = -1
        self.cur_ins_iou = 0.0
        self.cur = 0
        self.size = size

    def next_episode(self, *args):
        if self.cur == self.shuffle:
            self.cur = 0
            self.pool = np.random.choice(self.ids, self.shuffle)
        img_id = self.pool[self.cur]
        img_path = os.path.join(self.img_path, '%s.jpg'%(str(img_id).zfill(6)))
        self.img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        ann_path = os.path.join(self.ann_path, '%s.npy'%(str(img_id).zfill(6)))
        self.anns = np.load(ann_path)
        self.ann_cnt = self.anns.shape[0]
        self.ann_visit = np.zeros(self.ann_cnt, dtype=np.int8)

        if self.size != None:
            self.size_zeros = np.zeros(self.size, dtype=np.float32)
            self.size_ones = np.ones(self.size, dtype=np.float32)
            self._resize()

        return self.img, self.anns

    def _resize(self):
        self.anns = self.anns.astype(np.uint8)
        tmp = np.zeros(shape=(self.ann_cnt, self.size[0], self.size[1]),
                       dtype=np.uint8)
        for i in range(self.ann_cnt):
            cv2.resize(self.anns[i], dsize=self.size, dst=tmp[i],
                       interpolation=cv2.INTER_CUBIC)
        self.anns = tmp
        tmp = np.zeros(shape=self.size, dtype=np.uint8)
        cv2.resize(self.img, dsize=self.size, dst=tmp,
                   interpolation=cv2.INTER_CUBIC)
        self.img = tmp

    def get_reward(self, *args):
        mask = args[0]

        if self.cur_ins_index == -1:
            scores = []
            for i in range(self.ann_cnt):
                if self.ann_visit[i]:
                    # have visited
                    continue
                scores.append(self.iou(mask, self.anns[i]))
            target = np.argmax(scores)
            self.cur_ins_iou = scores[target]
            if scores[target] == 0:
                # do not need to try any more
                return 0.0, True
            else:
                self.cur_ins_index = target
                return scores[target], False
        score = self.iou(mask, self.anns[self.cur_ins_index])
        reward = score - self.cur_ins_iou
        self.cur_ins_iou = score
        if score < 0.9 or reward > 0:
            # if iou is low or reward still good
            return reward, False
        else:
            # if iou is good enough and reward starts decline, stop segmentation
            self.cur_ins_index = -1
            self.ann_visit[self.cur_ins_index] = 1
            return reward, True

    def get_gt(self):
        if self.cur_ins_index == -1:
            return np.zeros(shape=self.size)
        return self.anns[self.cur_ins_index]

    def iou(self, mask1, mask2):
        m1 = np.sum(np.where((mask1 == 1) & (mask2 == 1), self.size_ones, self.size_zeros))
        m2 = np.sum(np.where((mask1 == 0) & (mask2 == 0), self.size_zeros, self.size_ones))
        return m1 * 1.0 / m2


if __name__ == '__main__':
    rw = cycleReward('D:\\Datasets\\cycle')
    rw.size = (768, 768)
    rw.next_episode()

    cycle_util.tool_show_data(rw.img, rw.anns, rw.ann_cnt)


