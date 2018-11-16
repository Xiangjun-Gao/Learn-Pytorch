from __future__ import print_function
import torch as t
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

t.__version__
# 构建 5x3 矩阵，只是分配了空间，未初始化
x = t.Tensor(5, 3)
x

x = t.Tensor([[1, 2], [3, 4]])
x
# 使用[0,1]均匀分布随机初始化二维数组
x = t.rand(5, 3)
print(x.size(), x.size()[0], x.size()[1])

y = t.rand(5, 3)
x + y

# 加法的第三种写法：指定加法结果的输出目标为result
result = t.Tensor(5, 3)  # 预先分配空间
t.add(x, y, out=result)  # 输入到result
result

a = t.ones(5)
a
b = a.numpy()
b

a = np.ones(5)
b = t.from_numpy(a)  # Numpy -> Tensor
print(a)
print(b)  # Tensor和Numpy共享内存,转换很快，同时改变

b.add_(1)  # 以`_`结尾的函数会修改自身
print(a)
print(b)

# print(b.item())
# ValueError: only one element tensors can be converted to Python scalars

scalar = b[0]
print(scalar)

print(scalar.size())  # 0-dim
print(scalar.item())  # 使用scalar.item()能从中取出python对象的数值
'''此外在pytorch中还有一个和np.array 很类似的接口: torch.tensor, 二者的使用十分类似。'''
tensor = t.tensor([2])  # 注意和scalar的区别
print(tensor, scalar)

# t.tensor(), np.array总是会进行数据拷贝，新tensor和原来的数据不再共享内存。
numpy = np.array([3, 4])
print(numpy)
old_numpy = numpy
# new_numpy = np.array(old_numpy)# 数据拷贝
new_numpy = old_numpy  # 数据引用
new_numpy[0] = 1111
print(old_numpy, new_numpy)

tensor = t.tensor([3, 4])
old_tensor = tensor
new_tensor = t.tensor(old_tensor)
new_tensor[0] = 1111
print(old_tensor, new_tensor)

# 如果你想共享内存的话，建议使用torch.from_numpy()或者tensor.detach()来新建一个tensor, 二者共享内存。
new_tensor = old_tensor.detach()
new_tensor[0] = 1111
print(old_tensor, new_tensor)

# Tensor可通过.cuda 方法转为GPU的Tensor，从而享受GPU带来的加速运算。
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
x = x.to(device)
y = y.to(device)
z = x + y
# 还可以使用tensor.cuda() 的方式将tensor拷贝到gpu上

# 为tensor设置 requires_grad 标识，代表着需要求导数
# pytorch 会自动调用autograd 记录操作
x = t.ones(2, 2, requires_grad=True)
y = x.sum()  # tensor(4.)

print(y.grad_fn)

y.backward()
print(x.grad)
x.grad.data.zero_()


class Net(nn.Module):
    def __init__(self):
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        # 下式等价于nn.Module.__init__(self)
        super(Net, self).__init__()
        # 卷积层 '1'表示输入图片为单通道, '6'表示输出通道数， 也就是卷积核的数量，'5'表示卷积核为5*5
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 仿射层/全连接层，y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        #  reshape，‘-1’表示自适应                                              ? ? ?
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


net = Net()
print(net)
'''
params = list(net.parameters())
print(len(params))
# print(params)

# forward函数的输入和输出都是Tensor。
for name, parameters in net.named_parameters():
    print(name, ':', parameters.size())

input = t.randn(1, 1, 32, 32)
out = net(input)
print(out.size())

net.zero_grad()
out.backward(t.ones(1, 10))  # ???
'''
output = net(input)
target = t.arrange(0, 10).view(1, 10)
criterion = nn.MSELoss()
loss = criterion(output, target)
print(loss)

net.zero_grad()
print('反向传播之前 conv1.bias的梯度')
print(net.conv1.bias.grad)
loss.backward()
print('反向传播之后 conv1.bias的梯度')
print(net.conv1.bias.grad)

optimizer = optim.SGD(net.parameters(), lr=0.01)
optimizer.zero_grad()
# 计算损失
output = net(input)
loss = criterion(output, target)

# 反向传播
loss.backward()

# 更新参数
optimizer.step()
