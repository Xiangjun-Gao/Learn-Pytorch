import torch as t
from matplotlib import pyplot as plt
from IPython import display
'''
tensor可以很随意的在gpu/cpu上传输。
使用tensor.cuda(device_id)或者tensor.cpu()。另外一个更通用的方法是tensor.to(device)。
'''
device = t.device('cpu')  # 如果你想用gpu，改成t.device('cuda:0')

# 设置随机数种子，保证在不同电脑上运行时下面的输出一致
t.manual_seed(1000)


def get_fake_data(batch_size=8):
    ''' 产生随机数据：y=x*2+3，加上了一些噪声'''
    x = t.rand(batch_size, 1, device=device) * 5
    y = x * 2 + 3 + t.randn(batch_size, 1, device=device)
    return x, y


print(t.randn(8, 1))

w = t.rand(1, 1).to(device)
b = t.zeros(1, 1).to(device)

lr = 0.02  # 学习率

for ii in range(500):
    x, y = get_fake_data(batch_size=4)

    # forward：计算loss
    y_pred = x.mm(w) + b.expand_as(y)  # x@W等价于x.mm(w);for python3 only
    loss = 0.5 * (y_pred - y)**2  # 均方误差
    loss = loss.mean()

    # backward：手动计算梯度
    dloss = 1
    dy_pred = dloss * (y_pred - y)

    dw = x.t().mm(dy_pred)  # 转置后矩阵相乘
    db = dy_pred.sum()

    # 更新参数
    w.sub_(lr * dw)
    b.sub_(lr * db)

    if ii % 100 == 0:

        # 画图
        display.clear_output(wait=True)
        x = t.arange(0, 6).view(-1, 1)
        y = x.mm(w) + b.expand_as(x)
        plt.plot(x.cpu().numpy(), y.cpu().numpy())  # predicted

        x2, y2 = get_fake_data(batch_size=32)
        plt.scatter(x2.numpy(), y2.numpy())  # true data

        plt.xlim(0, 5)
        plt.ylim(0, 13)
        plt.show()
        plt.pause(0.5)

print('w: ', w.item(), 'b: ', b.item())
