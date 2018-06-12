import numpy as np
import cv2

class Painter():
    def __init__(self, name):
        self.name = name

    def load(self, path):
        self.img = cv2.imread(path)
        self.h, self.w = self.img.shape[:2]
        self.mask = np.zeros((self.h+2, self.w+2), np.uint8)


    def make_action(self, op, point, loDiff, upDiff, flags=4|(255<<8)|cv2.FLOODFILL_MASK_ONLY):
        cv2.floodFill(self.img, self.mask, point, newVal= 0,
                      loDiff=loDiff, upDiff=upDiff,
                      flags=flags)


    def new_episode(self):
        return


    def get_state(self):
        return self.img, self.mask


    def is_episode_finished(self):
        return False


    def reward(self):
        return 0.01


    def onMouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            print((x,y))
            self.make_action((x,y), (10,)*3, (10,)*3)



if __name__ == '__main__':
    p = Painter('demo/000000000471.jpg')
    cv2.namedWindow('demo')
    cv2.setMouseCallback('demo', p.onMouse)
    while (1):
        cv2.imshow('demo', p.img)
        cv2.imshow('mask', p.mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q键退出
            break
    cv2.destroyAllWindows()


