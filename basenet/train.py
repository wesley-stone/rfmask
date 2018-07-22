from basenet.unet import *
from basenet.provider import *

nx, ny = 512, 512
prd = cycleProvider(nx=nx, ny=ny, path='D:\\Datasets\\cycle', load_to_mem=True)
net = Unet(channels=2, n_class=2, layers=3, features_root=16)
trainer = Trainer(net, optimizer="momentum", opt_kwargs=dict(momentum=0.2))

path = trainer.train(prd, "./unet_trained", training_iters=128, epochs=20, display_step=2)