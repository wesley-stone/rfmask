"""
A simple example of an animated plot
"""
import numpy as np
import matplotlib;matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

'''
fig, ax = plt.subplots()

img = np.random.rand(128, 128)
im = plt.imshow(img, cmap=plt.get_cmap('gray'))


def animate(i):
    im.set_array(np.random.rand(128, 128))
    return im,


# Init only required for blitting to give a clean slate.
def init():
    im.set_array(np.random.rand(128, 128))
    return im,

ani = animation.FuncAnimation(fig, animate, np.arange(1, 200),
                              interval=25, blit=True)
plt.show(block=False)
print('jfiejaifejiafeafae')
plt.show()
'''

plt.ion()
fig = plt.figure()
img = np.random.rand(128, 128)
im = plt.imshow(img, cmap=plt.get_cmap('gray'))

while True:
    time.sleep(1)
    im.set_data(np.random.rand(128, 128))
    fig.canvas.draw()
    plt.pause(0.5)
