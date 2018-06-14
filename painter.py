import numpy as np
import cv2


class Painter:
    def __init__(self, name):
        self.name = name

    def load(self, path):
        self.img = np.zeros(shape=self.size, dtype=np.uint8)
        tmp = cv2.imread(path)
        print(self.size)
        cv2.resize(tmp, dsize=(self.size[0], self.size[1]), dst=self.img)
        self.h, self.w = self.img.shape[:2]
        self.mask = np.ones((self.h+2, self.w+2), np.uint8)  # 1 means unmask, 255 means mask
        self.pre_mask = self.mask.copy()

    def make_action(self, operation, point, loDiff, upDiff):
        '''

        :param operation: <0.5 for region removal, >0.5 for addition
        :param point: point position value (x, y) in [0, 1]
        :param loDiff: floodfill param in [0, 1]
        :param upDiff: floodfill param in [0, 1]
        :return: operation reward
        '''
        if operation == 0:  # remove a region
            flags = 4 | (1 << 8) | cv2.FLOODFILL_MASK_ONLY
        else: # add a region
            flags = 4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY
        # transform to image position
        point = (int(point[0]*self.w), int(point[1]*self.h))
        # apply floodfill on mask
        print(point)
        print(loDiff)
        print(upDiff)
        cv2.floodFill(self.img, self.pre_mask, point, newVal=0,
                      loDiff=loDiff*255, upDiff=upDiff*255,
                      flags=flags)
        # swap pre_mask and mask
        tmp = self.pre_mask
        self.pre_mask = self.mask
        self.mask = tmp
        return self.__reward()

    def new_episode(self, size):
        self.size = size
        print('another episode!')
        self.load(self.__next())
        return


    def get_state(self):
        return self.mask

    def get_image(self):
        return self.img

    def is_episode_finished(self):
        return False


    def __reward(self):
        # compute improvement from pre_mask to mask
        self.reward = 0.01
        return self.reward

    def __next(self):
        return 'demo/000000000092.jpg'

    def onMouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            print((x,y))
            self.make_action(1, (x,y), (10,)*3, (10,)*3)


    def show(self):
        cv2.destroyWindow('demo')
        cv2.namedWindow('demo')
        cv2.setMouseCallback('demo', self.onMouse)
        while (1):
            cv2.imshow('demo', self.img)
            cv2.imshow('mask', self.mask)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

if __name__ == '__main__':
    p = Painter('demo/000000000471.jpg')


