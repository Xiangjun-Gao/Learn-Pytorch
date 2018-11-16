import torch as t
from torch import nn
from torch import optim

class Linear(nn.Moudle):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.w = nn.Parameter(t.randn(in_features, out_features))
        self.b = nn.Parameter(t.randn(out_features))

    def forward(self, x):
        x = x.mm(self.w)
        return x + self.b.expand_as(x)


class Perceptron(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(Perceptron, self).__init__()
        self.layer1 = Linear(in_features, hidden_features)
        # 此处的Linear是前面自定义的全连接层
        self.layer2 = Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.layer1(x)
        x = t.sigmoid(x)
        return self.layer2(x)


perceptron = Perceptron(3, 4, 1)
for name, para in perceptron.named_parameters():
    print(name, para.size())


# 首先定义一个LeNet网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 5), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120), nn.ReLU(), nn.Linear(120, 84),
            nn.ReLU(), nn.Linear(84, 10))

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.classifier(x)
        return x


net = Net()

'''
对于如何调整学习率，主要有两种做法。
一种是修改optimizer.param_groups中对应的学习率，
另一种是更简单也是较为推荐的做法——新建优化器，由于optimizer十分轻量级，构建开销很小，故而可以构建新的optimizer。
但是后者对于使用动量的优化器（如Adam），会丢失动量等状态信息，可能会造成损失函数的收敛出现震荡等情况。
'''

# 方法1: 调整学习率，新建一个optimizer
old_lr = 0.1
optimizer = optim.SGD([{
    'params': net.features.parameters()
}, {
    'params': net.classifier.parameters(),
    'lr': old_lr * 0.1
}],
                       lr=1e-5)
# 方法2: 调整学习率, 手动decay, 保存动量
for param_group in optimizer.param_groups:
    param_group['lr'] *= 0.1  # 学习率为之前的0.1倍

