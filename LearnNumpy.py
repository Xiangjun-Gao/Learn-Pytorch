import sys
from math import sqrt
import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt

# dict
nums = [0, 1, 2, 3, 4, 8, 9]
num_to_square = {x: x**2 for x in nums if x % 2 != 0}
print(num_to_square)

# set
nums = {int(sqrt(x)) for x in range(30)}
print(nums)

d = {(x, x + 1): x for x in range(6)}
print(d)
t = (5, 6)
print(type(t))
print(d[t])
print(d[(1, 2)])


# "python.pythonPath": "E://Anaconda3//python.exe"
def sign(x):
    if x > 0:
        return 'positive'
    elif x < 0:
        return 'negative'
    else:
        return 'zero'


for x in [-1, 0, 1]:
    print(sign(x))


def hello(name, loud=False):
    if loud:
        print('Hello, %s' % name.upper())
    else:
        print('Hello, %s' % name)


hello('Bod')
hello('Fred', loud=True)


class Greeter(object):
    # Constructor
    def __init__(self, name):
        self.name = name  # Create an instance variable

    # Instance method
    def greet(self, loud=False):
        if loud:
            print('HELLO, %s!' % self.name.upper())
        else:
            print('Hello, %s!' % self.name)


g = Greeter('Fred')  # Construct an instance of the Greeter class
g.greet()  # Call an instance method; prints "Hello, Fred"
g.greet(loud=True)  # Call an instance method; prints "HELLO, FRED!"

print(np.arange(10))
print(range(4))
for i in range(4):
    print(i)

a = np.array([[1, 2], [3, 4], [5, 6]])
bool_idx = (a > 2)
print('bool_idx\n', bool_idx)
print('print(a[bool_idx])\n', a[bool_idx])
a[bool_idx] = 2
print('a[bool_idx] = 2\n', a)

x = np.array([1, 2])
print(x.dtype)
x = np.array([1.0, 2.0])
print(x.dtype)

x = np.array([[1, 2], [3, 4]], dtype=np.float64)
y = np.array([[5, 6], [7, 8]], dtype=np.float64)
print(x + y)
print(x - y)
print(x * y)
print(x / y)
print(np.sqrt(x))

x = np.array([[1, 2], [3, 4]])
y = np.array([[5, 6], [7, 8]])
v = np.array([9, 10])
w = np.array([11, 12])

x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = np.empty_like(x)
print(y)
for i in range(x.shape[0]):
    y[i, :] = x[i, :] + v
print(y)

x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = x + v
print(y)
''''
img = imread(
    r'E:\F disk\GitHub\0-All-My-Code\Python\code for learning\Basket.jpg')
print(img.dtype, img.shape)

x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

plt.plot(x, y_sin)
plt.plot(x, y_cos)
plt.title('Sine and Cosine')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.legend(['sine', 'cosine'])
plt.show()

plt.subplot(2, 1, 1)
plt.plot(x, y_sin)
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Sine')
plt.legend(['sine'])

plt.subplot(2, 1, 2)
plt.plot(x, y_cos)
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Cosine')
plt.legend(['cosine'])

plt.show()

img_tinted = img * [1, 0.8, 0.9]
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.subplot(1, 2, 2)
plt.imshow(np.uint8(img_tinted))

plt.show()

print(sys.version)
'''

a = np.array([[1, 2], [3, 4]])
print(a.shape)
b = np.zeros(a.shape)
print(b)
