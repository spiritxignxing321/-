from make_cards import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
import cv2

class mnistClass():

    train_dataset = datasets.MNIST(root="data/sample",
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   download=True)    #下载训练集

    test_dataset = datasets.MNIST(root="data/fandy_new",
                                  train=False,
                                  transform=transforms.ToTensor(),
                                  download=True)    #下载测试集

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)   # 装载训练集

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=True)   # 装载测试集
'''
    def __init__(self):
        self.index = 0
        self.cards, self.label = load_from_file("data/sample")
        self.cards_test, self.label_test = load_from_file("data/fandy_new")
        self.num=self.cards.shape[0]

    def getNum(self):
        return self.num

    def getTest(self):
        return self.cards_test,self.label_test

    def getTrain(self):
        return self.cards,self.label

    def get_next_batch(self,batch_size):
        if self.index+batch_size<self.num:
            end=self.index+batch_size
            data=self.cards[self.index:end:1, :]
            label = self.label[self.index:end:1, :]
            self.index=self.index+batch_size
            return data,label
        else:
            end=(self.index+batch_size)%self.num
            data = self.cards[self.index:end:-1, :]
            label = self.label[self.index:end:-1, :]
            self.index = end
            return data,label
'''