import os
import numpy as np

class DataSet(object):
    def next(self):
        raise NotImplementedError

class cocoDataSet(DataSet):
    def __init__(self, data_path, pool_size = 1000):
        self.data_path = data_path
        tmp = os.listdir(data_path)
        self.img_paths = ['%s/%s' % (data_path, s) for s in tmp]
        self.img_ids = [int(s.split('.')[0]) for s in tmp]
        self.size = len(self.img_paths)
        self.log = np.zeros(self.size, dtype=np.uint32)
        self.pool_size = pool_size
        self.pool = self.__next_pool()
        self.cur_index = 0

    def __next_pool(self):
        tmp = np.random.choice(self.size, size=self.pool_size)
        return tmp

    def next(self):
        if self.cur_index == self.pool_size:
            self.pool = self.__next_pool()
            self.cur_index = 0
        r = self.pool[self.cur_index]
        self.cur_index += 1
        return self.img_paths[r], self.img_ids[r]


if __name__ == '__main__':
    # test
    ds = cocoDataSet('D:/Datasets/coco/train2017')
    print(ds.size)
    for i in range(2345):
        print(ds.next())