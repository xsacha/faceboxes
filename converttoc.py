import os
import torch
from torch.nn import init
from torch.utils.data import DataLoader
from torchvision import transforms, models

from networks import FaceBox
from multibox_loss import MultiBoxLoss
from dataset import ListDataset
import numpy as np

def train():

    learning_rate = 0.001
    num_epochs = 300
    batch_size = 32

    net = FaceBox()
    print('load model...')
    net.load_state_dict(torch.load('weight/faceboxes.pt'))
    traced_script_module = torch.jit.trace(net, torch.rand(1, 3, 1024, 1024))
    traced_script_module.save("faceboxes.pt")

if __name__ == '__main__':
    train()

