import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self,input,hidden,output):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input, hidden)  # 第一层：输入维度28*28，输出维度64
        self.fc2 = nn.Linear(hidden,output)     # 第二层：输入维度64，输出维度10

    def forward(self, x):
        x = x.view(-1, 28*28)            # 将输入张量展平为向量
        x = torch.relu(self.fc1(x))      # 第一层的激活函数使用ReLU
        x = self.fc2(x)                  # 第二层没有激活函数，输出层
        return x
