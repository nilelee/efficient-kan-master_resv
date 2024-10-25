import torch
import torch.nn as nn

import torch.nn.functional as F

class Alexnet(nn.Module):
    # 构造函数
    def __init__(self):
        super(Alexnet,self).__init__()
        # 对由多个输入平面组成的输入信号进行二维卷积
        # in_channels: int,输入通道
    #    out_channels: int,输出通道
        self.ReLU = nn.ReLU(inplace=True)
        self.c1 = torch.nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1)
        
        # 最大池化的目的在于保留原特征的同时减少神经网络训练的参数，
        # 使得训练时间减少。相当于1080p的视频变为了720p
        # MaxPool2d对输入向量做二维最大池化操作,能减少计算量
        self.s1 = torch.nn.MaxPool2d(kernel_size=2,stride=2)
        # kernel_size: Union[int, Tuple[int, ...]],
        # stride: Union[int, Tuple[int, ...], NoneType] = None,
        self.c2 = torch.nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.s2 = torch.nn.MaxPool2d(kernel_size=2,stride=2)
        
        
        self.c3 = torch.nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        
        
        self.c4 = torch.nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=1)
        
        self.c5 = torch.nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding=1)
        self.s5 = nn.MaxPool2d(kernel_size=3,stride=2)
        # # Flatten()：将张量（多维数组）平坦化处理，神经网络中第0维表示的是batch_size，
        # 所以Flatten()默认从第二维开始平坦化
        self.flatten = nn.Flatten()
        # 全连接层&softmax
        # 负责将卷积输出的二维特征图转化成一维的一个向量，由此实现了端到端的学习过程（即：输入一张图像或一段语音，输出一个向量或信息）
        self.fc1 = nn.Linear(3*3*256,1024)
        self.fc2 = nn.Linear(1024,512) 
        self.fc3 = nn.Linear(512,10)
        
    def forward(self,x):# 正向传递的函数 池化激活 卷积激活
        x = x.view(x.size(0), 1, 28, 28)
        x = self.ReLU(self.c1(x))
        x = self.s1(x)
        x = self.ReLU(self.c2(x))
        x = self.s2(x)
        x = self.ReLU(self.c3(x))
        x = self.ReLU(self.c4(x))
        x = self.ReLU(self.c5(x))
        x = self.s5(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.dropout(x,p=0.5)
            
        x = self.fc2(x)
        x = F.dropout(x,p=0.5)
            
        x = self.fc3(x)
            
        output = F.log_softmax(x,dim=1)# 计算分类后，每个数字的概率值
            
        return output
