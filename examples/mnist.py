# Train on MNIST

# CIFAR-10 32x32 3 channels


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from cnn import Alexnet
from mlp import MLP
from cnn_kan import Alexnet_KAN
import matplotlib.pyplot as plt
import csv
## 路径修改
import sys
sys.path.append('../src')
from src.efficient_kan import KAN

##################################################################################
#### 数据处理                       数据处理                        数据处理   #####
##################################################################################

##转tensor并且进行标准化
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

##################################################################################
#### 数据集构建                     数据集构建                      数据集构建  #####
##################################################################################

##torchvision加载mnist手写识别数据集
trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
valset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)

##################################################################################
#### 数据加载器                  数据加载器                      数据集加载器   #####
##################################################################################

##设定batch_size为64且利用shuffle打乱训练数据
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
valloader = DataLoader(valset, batch_size=64, shuffle=False)

##################################################################################
#### 模型实例化                  模型实例化                     模型实例化      #####
##################################################################################

##设定KAN为3层
##第一层（输入层）为28x28，传输数据的shape为(batch_size,channel,width,heigh)=>(64，1，28，28)
##第二层（隐藏层）为64，自己设定
##第三层（输出层）为10，对应得下游任务是图片10分类
# model = KAN([28 * 28, 64, 10])
# model_name='KAN'

##mlp设定
##传入参数（w*h，hidden，分类数量）
# model=MLP(28 * 28,64,10)
# model_name='MLP'

##cnn设定
# model=Alexnet()
# model_name='Alexnet'

##cnn+KAN设定
model=Alexnet_KAN()
model_name='AlexnetKAN'

##################################################################################
####  设备设定                    设备设定                       设备设定      #####
##################################################################################

##device 指定 cuda or cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(model)

##################################################################################
####  优化器设定&学习策略    优化器设定&学习策略              优化器设定&学习策略  ####
##################################################################################

##优化器设定
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
##指数型下降的学习率调节器
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

##################################################################################
####  交叉熵损失                   交叉熵损失                  交叉熵损失      #####
##################################################################################
criterion = nn.CrossEntropyLoss()



trian_loss_list=[]
train_acc_list=[]
val_loss_list=[]
val_acc_list=[]



for epoch in range(20):
##################################################################################
####  训练过程设定               训练过程设定                 训练过程设定      #####
##################################################################################
    model.train()
    with tqdm(trainloader) as pbar:
        for i, (images, labels) in enumerate(pbar):
            print(images.shape)
    
        ## 将(64，1，28，28)的图片数据转化为（64，28*28）的二维数据
            # images = images.view(-1, 28 * 28).to(device)
            images = images.view(-1, 3, 32, 32).to(device) #三通道彩色图片

        ##梯度归0
            optimizer.zero_grad()
        ##将（64，28*28）的二维数据放入模型得到（64，10）的预测值
            # output = model(images)
            output = model(images)

        ##将预测值和真实值进行损失计算
            loss = criterion(output, labels.to(device))
            
        ##反向传播
            loss.backward()
        ##优化
            optimizer.step()
        ##精度计算
            accuracy = (output.argmax(dim=1) == labels.to(device)).float().mean()
            pbar.set_postfix(loss=loss.item(), accuracy=accuracy.item(), lr=optimizer.param_groups[0]['lr'])
        ##绘图数据统计
        trian_loss_list.append(loss.item())
        train_acc_list.append(accuracy.item())

##################################################################################
####  验证过程设定               验证过程设定                 验证过程设定      #####
##################################################################################

    model.eval()
    val_loss = 0
    val_accuracy = 0
    ##梯度不更新
    with torch.no_grad():
        for images, labels in valloader:
            # images = images.view(-1, 28 * 28).to(device)
            images = images.view(-1, 3,32 * 32).to(device)#彩色

            output=model(images)
            val_loss += criterion(output, labels.to(device)).item()
            val_accuracy += (
                (output.argmax(dim=1) == labels.to(device)).float().mean().item()
            )
    val_loss /= len(valloader)
    val_accuracy /= len(valloader)

    # 更新学习率
    scheduler.step()

    print(
        f"Epoch {epoch + 1}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}"
    )
    val_loss_list.append(val_loss)
    val_acc_list.append(val_accuracy)


##################################################################################
####  指标展示                   指标展示                 指标展示             #####
##################################################################################

epochs = range(1, len(trian_loss_list) + 1)  # 横轴表示 epochs

# 绘制训练损失和验证损失的折线图
plt.figure(figsize=(10, 5))
plt.plot(epochs, trian_loss_list, 'b', label='Training loss')
plt.plot(epochs, val_loss_list, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f"pictures/{model_name}_loss_curve{epoch+1}.png")
plt.show()

# 绘制训练准确率和验证准确率的折线图
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_acc_list, 'b', label='Training accuracy')
plt.plot(epochs, val_acc_list, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(f"pictures/{model_name}_accuracy_curve{epoch+1}.png")
plt.show()

##################################################################################
####  指标保存                  指标保存                指标保存               #####
##################################################################################
with open(f"result/{model_name}_epoch{epoch+1}.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"])
        for i in range(len(trian_loss_list)):
            writer.writerow([i+1, trian_loss_list[i], train_acc_list[i], val_loss_list[i], val_acc_list[i]])