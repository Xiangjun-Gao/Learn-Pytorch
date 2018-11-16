from __future__ import print_function
import torch as t


def f(x):
    '''计算y'''
    y = x**2 * t.exp(x)
    return y


def gradf(x):
    '''手动求导函数'''
    dx = 2 * x * t.exp(x) + x**2 * t.exp(x)
    return dx


x = t.randn(3, 4, requires_grad=True)
y = f(x)
y.backward(t.ones(y.shape))
print(x.grad)
print(gradf(x))

x = t.ones(1)
b = t.rand(1, requires_grad=True)
w = t.rand(1, requires_grad=True)
y = w * x  # 等价于y=w.mul(x)
z = y + b  # 等价于z=y.add(b)
print(x.requires_grad, b.requires_grad, w.requires_grad)
print(x.is_leaf, w.is_leaf, b.is_leaf)  # 颠覆
print(y.is_leaf, z.is_leaf)

print(z.grad_fn)
print(z.grad_fn.next_functions)
print(y.grad_fn.next_functions)
print(w.grad_fn, x.grad_fn)

with t.no_grad():
    x = t.ones(1)
    w = t.rand(1, requires_grad=True)
    y = x * w
# y依赖于w和x，虽然w.requires_grad = True，但是y的requires_grad依旧为False
x.requires_grad, w.requires_grad, y.requires_grad
# (False, True, False)

