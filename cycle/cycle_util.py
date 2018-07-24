from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

import threading
from time import sleep


def create_image_and_label(nx, ny, cnt=10, r_min=5, r_max=80,
                           border=92, sigma=20):
    image = np.ones((nx, ny, 1))
    label = np.zeros((cnt, nx, ny), dtype=np.bool)
    colors = range(80, 241, (160//cnt))
    mask = np.zeros((nx, ny), dtype=np.bool)
    for i in range(cnt):
        a = np.random.randint(border, nx - border)
        b = np.random.randint(border, ny - border)
        r = np.random.randint(r_min, r_max)
        h = colors[i]

        y, x = np.ogrid[-a:nx - a, -b:ny - b]
        label[i] = x * x + y * y <= r * r
        label[i] = np.logical_xor(np.logical_and(mask, label[i]), label[i])
        mask = np.logical_or(mask, label[i])

        image[label[i]] = h

    image += np.random.normal(scale=sigma, size=image.shape)
    image -= np.amin(image)
    image /= np.amax(image)
    image *= 255
    return image.astype(np.uint8), label


def to_rgb(img):
    img = img.reshape(img.shape[0], img.shape[1])
    img[np.isnan(img)] = 0
    img -= np.amin(img)
    img /= np.amax(img)
    blue = np.clip(4 * (0.75 - img), 0, 1)
    red = np.clip(4 * (img - 0.25), 0, 1)
    green = np.clip(44 * np.fabs(img - 0.5) - 1., 0, 1)
    rgb = np.stack((red, green, blue), axis=2)
    return rgb


def generate_data(root_path, start=0, num=1000, min_cnt=1, max_cnt=5):
    nxs = np.array([512, 600, 768, 1024])
    nys = np.array([512, 600, 768, 1024])
    nx_samples = np.random.choice(len(nxs), num)
    ny_samples = np.random.choice(len(nys), num)
    nx_samples = nxs[nx_samples]
    ny_samples = nys[ny_samples]
    cnt_samples = np.random.choice(max_cnt-min_cnt, num) + min_cnt
    pic_pth = os.path.join(root_path, 'img')
    ann_pth = os.path.join(root_path, 'ann')

    for t in range(num):
        name = str(t+start).zfill(6) + '.jpg'
        ann_name = str(t+start).zfill(6) + '.npy'
        print('%s %s'%(threading.current_thread(),name))
        tpath = os.path.join(pic_pth, name)
        tapath = os.path.join(ann_pth, ann_name)
        img, labels = create_image_and_label(nx_samples[t], ny_samples[t], cnt_samples[t])
        cv2.imwrite(tpath, img)
        np.save(tapath, labels)


def tool_generate_data():
    start = 0
    num = 500
    threads = []
    for i in range(20):
        trd = threading.Thread(target=generate_data, args=('D:\\Datasets\\cycle', start, num))
        threads.append(trd)
        trd.setDaemon(True)
        print(trd.getName())
        trd.start()
        start+= num
    for d in threads:
        d.join()


def tool_generate_list(pth):
    img_path = os.path.join(pth, 'img')
    ls = os.listdir(img_path)
    dst = os.path.join(img_path, 'data.txt')
    with open(dst, 'w') as f:
        for l in ls:
            l = l.split('.')[0]
            f.write(l + '\n')


def tool_show_data(img, anns, cnt):
    ish = plt.imshow(img)
    ish.set_cmap('gray')
    plt.show()

    for i in range(cnt):
        ish = plt.imshow(anns[i])
        ish.set_cmap('gray')
        plt.show()


if __name__== '__main__':
    # tool_generate_list('D:/Datasets/cycle')
    tool_generate_data()
    '''read data'''
    '''
    img = cv2.imread('/Users/redrock/cycle_data/img/000001.jpg', flags=cv2.IMREAD_GRAYSCALE)
    label = np.load('/Users/redrock/cycle_data/annotations/000001.npy')
    '''
    '''test image generate'''
    '''
    # img, label = create_image_and_label(512, 512,cnt=3)
    # img = np.squeeze(img)
    # img = to_rgb(img)
    ish = plt.imshow(img)
    ish.set_cmap('gray')
    plt.show()
    cnt = label.shape[-1]
    for i in range(cnt):
        ish = plt.imshow(label[..., i])
        ish.set_cmap('gray')
        plt.show()
        print(iou(label[..., 0], label[..., i]))
    '''
