import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import time
# 定义超参数
batch_size = 32 #分类面，loss函数， batchsize太小会增加坏样本的影响
learning_rate = 2e-3
num_epoches = 200

# 下载训练集 MNIST 手写数字训练集
train_dataset = datasets.MNIST(
    root='./data', train=True, transform=transforms.ToTensor(), download=True) #此数据集里面有10个标签

test_dataset = datasets.MNIST(
    root='./data', train=False, transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 定义 Logistic Regression 模型
class Logistic_Regression(nn.Module):
    def __init__(self, in_dim, n_class):
        super(Logistic_Regression, self).__init__()
        self.logistic = nn.Linear(in_dim, n_class)#全连接层起到分类的作用, 输入一张图片,输出分属10个类别的概率

    def forward(self, x):
        out = self.logistic(x)
        return out  #32*10,batchsize*num_class


model = Logistic_Regression(28 * 28, 10)  # 图片大小是28x28
use_gpu = torch.cuda.is_available()  # 判断是否有GPU加速
#print(use_gpu) False
if torch.cuda.is_available() :
    model = model.cuda()
# 定义loss和optimizer
criterion = nn.CrossEntropyLoss() #交叉熵，代价函数，分类专用
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 开始训练
for epoch in range(num_epoches):
    print('*' * 10)
    print('epoch {}'.format(epoch + 1))
    since = time.time()
    running_loss = 0.0
    running_acc = 0.0
    for i, data in enumerate(train_loader, 1): #第i批， 共有i * 32张图片
        img, label = data #img.size() [32,1,28,28], batchsize = 32, 一次传入32张照片
        img = img.view(img.size(0), -1)  # 将图片展开成 28x28，即为32*764

        if torch.cuda.is_available():
            img = Variable(img).cuda()
            label = Variable(label).cuda()
        else:
            img = Variable(img)
            label = Variable(label)

        # 向前传播
        out = model(img)
        loss = criterion(out, label)
        '''
        print('out:{}'.format(out.size()))
        print('label:{}'.format(label.size()))
        print('img:{}'.format(img.size()))
        print(loss)
            out:torch.Size([32, 10]) 32*10,32张图片分属10个类别的概率
            label:torch.Size([32])
            img:torch.Size([32, 784])
            tensor(1.7395)
            '''
        running_loss += loss.item() * label.size(0)
        _, pred = torch.max(out, 1) # _ 起到占位的作用， pred 32X1， 预测的类别
        num_correct =            (pred == label).sum()  #预测正确的个数， 但是num_coreect 仍然是一个Variable变量
        running_acc += num_correct.item()
        # 向后传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 300 == 0:
            print('[{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(
                epoch + 1, num_epoches, running_loss / (batch_size * i),
                running_acc / (batch_size * i)))

    print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
        epoch + 1, running_loss / (len(train_dataset)), running_acc / (len(
            train_dataset))))
    print(len(train_dataset))


    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    for data in test_loader:
        img, label = data
        img = img.view(img.size(0), -1)
        if torch.cuda.is_available() :
            with torch.no_grad():
                img = Variable(img).cuda()
                label = Variable(label).cuda()
        else:
            with torch.no_grad():
                img = Variable(img)
                label = Variable(label) #volatile=True
        out = model(img)
        loss = criterion(out, label)
        eval_loss += loss.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        eval_acc += num_correct.item()
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
        test_dataset)), eval_acc / (len(test_dataset))))
    print('Time:{:.1f} s'.format(time.time() - since))

# 保存模型
torch.save(model.state_dict(), './logstic.pth')