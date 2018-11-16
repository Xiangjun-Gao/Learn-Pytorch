import torch as t
import numpy as np
# Tensor函数新建tensor,1 既可以接收一个list 2 根据指定的形状新建tensor 3 还能传入其他的tensor
a = t.Tensor(2, 3)
print(a)
b = t.Tensor([[1, 2, 3], [4, 5, 6]])
print(b, b.tolist)

# a = t.tensor(2, 3) 使用错误，只能接受一个参数
b = t.tensor([[1, 2, 3], [4, 5, 6]])
print(b)

b_size = b.size()  # tensor.shape等价于tensor.size()
print(b_size)  # tensor.size()返回torch.Size对象，它是tuple的子类，但其使用方式与tuple略有区别

# 创建一个和b形状一样的tensor
c = t.Tensor(b_size)
print(type(c), type(b))
# 创建一个元素为2和3的tensor, 下面两条语句说明，即可以接受list 也可以接受tuple
d = t.Tensor((2, 3))
d = t.tensor((2, 3))
c.numel()  # b中元素总个数，2*3，等价于b.nelement()
# print(c)
'''
Error:overflow
'''
print(d)

# torch.tensor是在0.4版本新增加的一个新版本的创建tensor方法，使用的方法，和参数几乎和np.array完全一致
t.ones(2, 3)
t.zeros(2, 3)
t.arange(1, 6, 2)
t.linspace(1, 10, 3)
t.randn(2, 3, device=t.device('cpu'))
t.eye(2, 3, dtype=t.int)  # 对角线为1, 不要求行列数一致

tensor = t.Tensor(1, 2)  # 注意和t.tensor([1, 2])的区别
tensor.shape

# tensor.view方法可以调整tensor的形状, 共享内存。
# tensor.squeeze() 在删除所有维度=1的维度， 若传入参数a， 如果=1，删除第a维度。 共享主存。
a = t.arange(0, 6)
a.view(2, 3)
print(a)

b = a.view(-1, 3)  # 当某一维为-1的时候，会自动计算它的大小
print(b, b.shape)
b[0][0] = 1111
print(a)
c = b.unsqueeze(1)
b[0][0] = 0
print(a)
print(b)
print(c)
c = b.squeeze(1)
print(a)
print(b)
print(c)

# resize是另一种可用来调整size的方法，view 必须使得大小前后相同
# 如果新大小超过了原大小，会自动分配新的内存空间，而如果新大小小于原大小，则之前的数据依旧会被保存，看一个例子。

t.set_default_tensor_type('torch.FloatTensor')
a = t.tensor([[1, 2], [3, 4]], dtype=t.double)
print(a, a.dtype)
b = a  # 内存共享
b[0][0] = 1111
print(b, b.dtype)
print(a, a.dtype)

a = a.float()
print(a, a.dtype)


# 注意： 当numpy的数据类型和Tensor的类型不一样的时候，使用torch.Tensor数据会被复制，不会共享内存。
# 不论输入的类型是什么，t.tensor都会进行数据拷贝，不会共享内存
a = np.ones([2, 3])
print(a.dtype)
b = t.Tensor(a)
print(b.dtype)
