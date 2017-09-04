#train.py
from __future__ import print_function
from math import log10
import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from unet import UNet
from BSDDataLoader import get_training_set,get_test_set
import torchvision


# Training settings
class option:
    def __init__(self):
        self.cuda = True #use cuda?
        self.batchSize = 4 #training batch size
        self.testBatchSize = 4 #testing batch size
        self.nEpochs = 140 #umber of epochs to train for
        self.lr = 0.001 #Learning Rate. Default=0.01
        self.threads = 4 #number of threads for data loader to use
        self.seed = 123 #random seed to use. Default=123
        self.size = 428
        self.remsize = 20
        self.colordim = 1
        self.target_mode = 'bon'
        self.pretrain_net = "/home/wcd/PytorchProject/Unet/unetdata/checkpoint/model_epoch_140.pth"

def map01(tensor,eps=1e-5):
    #input/output:tensor
    max = np.max(tensor.numpy(), axis=(1,2,3), keepdims=True)
    min = np.min(tensor.numpy(), axis=(1,2,3), keepdims=True)
    if (max-min).any():
        return torch.from_numpy( (tensor.numpy() - min) / (max-min + eps) )
    else:
        return torch.from_numpy( (tensor.numpy() - min) / (max-min) )


def sizeIsValid(size):
    for i in range(4):
        size -= 4
        if size%2:
            return 0
        else:
            size /= 2
    for i in range(4):
        size -= 4
        size *= 2
    return size-4



opt = option()
target_size = sizeIsValid(opt.size)
print("outputsize is: "+str(target_size))
if not target_size:
    raise  Exception("input size invalid")
target_gap = (opt.size - target_size)//2
cuda = opt.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
train_set = get_training_set(opt.size + opt.remsize, target_mode=opt.target_mode, colordim=opt.colordim)
test_set = get_test_set(opt.size, target_mode=opt.target_mode, colordim=opt.colordim)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

print('===> Building unet')
unet = UNet(opt.colordim)


criterion = nn.MSELoss()
if cuda:
    unet = unet.cuda()
    criterion = criterion.cuda()

pretrained = True
if pretrained:
    unet.load_state_dict(torch.load(opt.pretrain_net))

optimizer = optim.SGD(unet.parameters(), lr=opt.lr)
print('===> Training unet')

def train(epoch):
    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        randH = random.randint(0, opt.remsize)
        randW = random.randint(0, opt.remsize)
        input = Variable(batch[0][:, :, randH:randH + opt.size, randW:randW + opt.size])
        target = Variable(batch[1][:, :,
                         randH + target_gap:randH + target_gap + target_size,
                         randW + target_gap:randW + target_gap + target_size])
        #target =target.squeeze(1)
        #print(target.data.size())
        if cuda:
            input = input.cuda()
            target = target.cuda()
        input = unet(input)
        #print(input.data.size())
        loss = criterion( input, target)
        epoch_loss += loss.data[0]
        loss.backward()
        optimizer.step()
        if iteration%10 is 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.data[0]))
    imgout = input.data/2 +1
    torchvision.utils.save_image(imgout,"/home/wcd/PytorchProject/Unet/unetdata/checkpoint/epch_"+str(epoch)+'.jpg')
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))


def test():
    totalloss = 0
    for batch in testing_data_loader:
        input = Variable(batch[0],volatile=True)
        target = Variable(batch[1][:, :,
                          target_gap:target_gap + target_size,
                          target_gap:target_gap + target_size],
                          volatile=True)
        #target =target.long().squeeze(1)
        if cuda:
            input = input.cuda()
            target = target.cuda()
        optimizer.zero_grad()
        prediction = unet(input)
        loss = criterion(prediction, target)
        totalloss += loss.data[0]
    print("===> Avg. test loss: {:.4f} dB".format(totalloss / len(testing_data_loader)))


def checkpoint(epoch):
    model_out_path = "/home/wcd/PytorchProject/Unet/unetdata/checkpoint/model_epoch_{}.pth".format(epoch)
    torch.save(unet.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

for epoch in range(141, 141+opt.nEpochs + 1):
    train(epoch)
    if epoch%10 is 0:
        checkpoint(epoch)
    test()
checkpoint(epoch)