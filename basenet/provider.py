import cv2
import glob
import numpy as np
from cycle import cycle_util
from PIL import Image
import matplotlib.pyplot as plt
import os

class BaseDataProvider:
    """
    Abstract base class for DataProvider implementation. Subclasses have to
    overwrite the `_next_data` method that load the next data and label array.
    This implementation automatically clips the data with the given min/max and
    normalizes the values to (0,1]. To change this behavoir the `_process_data`
    method can be overwritten. To enable some post processing such as data
    augmentation the `_post_process` method can be overwritten.

    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping

    """

    channels = 1
    n_class = 2


    def __init__(self, a_min=None, a_max=None):
        self.a_min = a_min if a_min is not None else -np.inf
        self.a_max = a_max if a_min is not None else np.inf

    def _load_data_and_label(self):
        data, label = self._next_data()

        train_data = self._process_data(data)
        labels = self._process_labels(label)

        train_data, labels = self._post_process(train_data, labels)

        nx = train_data.shape[1]
        ny = train_data.shape[0]

        return train_data.reshape(1, ny, nx, self.channels), labels.reshape(1, ny, nx, self.n_class),

    def _process_labels(self, label):
        if self.n_class == 2:
            nx = label.shape[1]
            ny = label.shape[0]
            labels = np.zeros((ny, nx, self.n_class), dtype=np.float32)
            labels[..., 1] = label
            labels[..., 0] = 1 - label
            return labels

        return label

    def _process_data(self, data):
        # normalization
        data = np.clip(np.fabs(data), self.a_min, self.a_max)
        data -= np.amin(data)
        data /= np.amax(data)
        return data

    def _post_process(self, data, labels):
        """
        Post processing hook that can be used for data augmentation

        :param data: the data array
        :param labels: the label array
        """
        return data, labels

    def __call__(self, n):
        '''

        train_data, labels = self._load_data_and_label()
        nx = train_data.shape[1]
        ny = train_data.shape[2]

        X = np.zeros((n, nx, ny, self.channels))
        Y = np.zeros((n, nx, ny, self.n_class))

        X[0] = train_data
        Y[0] = labels
        for i in range(1, n):
            train_data, labels = self._load_data_and_label()
            X[i] = train_data
            Y[i] = labels

        return X, Y
        '''
        return self._next_batch(n)


    def _next_data(self):
        raise NotImplementedError()

    def _next_batch(self, batch_size):
        raise NotImplementedError()


class cycleProvider(BaseDataProvider):
    channels = 2
    n_class = 2

    def __init__(self, nx, ny, path=None, load_to_mem=False):
        super(cycleProvider, self).__init__()
        self.nx = nx
        self.ny = ny
        self.path = path
        self.size = -1
        if path is not None:
            ls_pth = os.path.join(path, 'img\\data.txt')
            img_pth = os.path.join(path, 'img')
            ann_pth = os.path.join(path, 'ann')
            with open(ls_pth, 'r') as f:
                ls = f.read().splitlines()
                self.img_path = [os.path.join(img_pth, '%s.jpg'%l) for l in ls]
                self.ann_path = [os.path.join(ann_pth, '%s.npy'%l) for l in ls]
            self.size = len(self.img_path)
            self.size = 500
            if load_to_mem:
                # load all data to the mem
                print('data loading starting...')
                self.imgs = np.zeros((self.size, self.ny, self.nx, 2), dtype=np.float32)
                self.anns = np.zeros((self.size, self.ny, self.nx, 2), dtype=np.float32)
                for n in range(self.size):
                    tmp_img = cv2.imread(self.img_path[n], cv2.IMREAD_GRAYSCALE)
                    tmp_anns = np.load(self.ann_path[n])
                    cnt = tmp_anns.shape[0]
                    size = tmp_anns.shape[1:]

                    s_label = np.zeros(size, dtype=np.bool)
                    for i in range(cnt):
                        s_label = np.logical_or(s_label, tmp_anns[i])
                    rimg, s_label = self.resize(tmp_img, s_label)
                    # replicate to 2 channels
                    self.imgs[n,:,:,0] = rimg
                    # set second channel to 0
                    self.imgs[n,:,:,1] = 0
                    self.anns[n,:,:,1] = s_label
                    self.anns[n,:,:,0] = 1 - s_label
                    if n % 1000 == 0:
                        print('%d.th image has been loaded...'%n)
                # img normalization
                print('image normalization...')
                mins = np.amin(self.imgs, axis=(1,2,3), keepdims=True)
                maxs = np.amax(self.imgs, axis=(1,2,3), keepdims=True)
                self.imgs = (self.imgs - mins)/maxs
                print('data loading finished...')

    def _next_data(self):
        if self.path is None:
            img, label = cycle_util.create_image_and_label(self.nx, self.ny)
        else:
            img, label = self._random_select_from_file()

        cnt = label.shape[0]
        size = label.shape[1:]
        s_label = np.zeros(size, dtype=np.bool)
        for i in range(cnt):
            s_label = np.logical_or(s_label, label[i])
        img, s_label = self.resize(img, s_label)
        # replicate to 2 channels
        dimg = np.zeros((self.ny, self.nx, 2), np.uint8)
        dimg[..., 0] = img
        dimg[..., 1] = np.zeros(img.shape, dtype=np.uint8)
        return dimg, s_label

    def _next_batch(self, batch_size):
        idx = np.random.choice(self.size, size=batch_size)
        return self.imgs[idx], self.anns[idx]

    def _random_select_from_file(self):
        idx = np.random.choice(self.size)
        img = cv2.imread(self.img_path[idx], cv2.IMREAD_GRAYSCALE)
        anns = np.load(self.ann_path[idx])
        return img, anns

    def resize(self, img, label):
        if label.shape == (self.ny, self.nx):
            return img, label
        label = label.astype(np.uint8)
        tmp = np.zeros(shape=(self.ny, self.nx), dtype=np.uint8)
        cv2.resize(label, dsize=(self.ny, self.nx), dst=tmp,
                       interpolation=cv2.INTER_CUBIC)
        itmp = np.zeros(shape=(self.ny, self.nx), dtype=np.uint8)
        cv2.resize(img, dsize=(self.ny, self.nx), dst=itmp,
                   interpolation=cv2.INTER_CUBIC)
        return itmp, tmp


def render(imgs, labels, num):
    # imgs = np.squeeze(imgs[..., 0:1])
    imgs = imgs[..., 0]
    for i in range(num):
        ish = plt.imshow(imgs[i])
        ish.set_cmap('gray')
        plt.show()
        ish = plt.imshow(labels[i, :, :, 0])
        ish.set_cmap('gray')
        plt.show()
        ish = plt.imshow(labels[i, :, :, 1])
        ish.set_cmap('gray')
        plt.show()


def render_labels(labels, num):
    for i in range(num):
        if labels.ndim == 4:
            ish = plt.imshow(labels[i, :, :, 1])
            ish.set_cmap('gray')
            plt.show()
            ish = plt.imshow(labels[i, :, :, 0])
            ish.set_cmap('gray')
            plt.show()
        else:
            ish = plt.imshow(labels[i, :, :])
            ish.set_cmap('gray')
            plt.show()


def render_gray(img):
    ish = plt.imshow(img)
    ish.set_cmap('gray')
    plt.show()



if __name__ == '__main__':
    prd = cycleProvider(512, 512, path='D:\\Datasets\\cycle')
    imgs, labels = prd(10)
    imgs = np.squeeze(imgs[...,0])
    labels = np.squeeze(labels)
    print(imgs.shape)
    print(labels.shape)
    for i in range(10):
        ish = plt.imshow(imgs[i])
        ish.set_cmap('gray')
        plt.show()
        ish = plt.imshow(labels[i,:,:,0])
        ish.set_cmap('gray')
        plt.show()
        ish = plt.imshow(labels[i,:,:,1])
        ish.set_cmap('gray')
        plt.show()
