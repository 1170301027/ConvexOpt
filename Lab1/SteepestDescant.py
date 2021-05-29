#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: lenovo
@file: SteepestDescant.py
@time: 2021/5/22 9:49
"""
import random
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker


def f(x, y):
    return (1 - x) ** 2 + 100 * (y - x * x) ** 2


def H(x, y):
    return np.matrix([[1200 * x * x - 400 * y + 2, -400 * x],
                      [-400 * x, 200]])


def grad(x, y):
    return np.matrix([[2 * x - 2 + 400 * x * (x * x - y)],
                      [200 * (y - x * x)]])



def goldsteinsearch(d, x,y, alpham, rho, t):
    '''
    线性搜索子函数
    当前迭代点x,y和当前搜索方向d
    '''
    flag = 0

    a = 0
    b = alpham
    d = np.squeeze(np.asarray(d))
    fk = np.squeeze(np.asarray(f(x,y)))
    gk = np.squeeze(np.asarray(grad(x,y)))

    phi0 = fk
    dphi0 = np.dot(gk, d)
    # print(dphi0)
    alpha = b * random.uniform(0, 1)

    while (flag == 0):
        newfk = f(x + alpha * d[0],y + alpha * d[1])
        phi = newfk
        # print(phi,phi0,rho,alpha ,dphi0)
        if (phi - phi0) <= (rho * alpha * dphi0):
            if (phi - phi0) >= ((1 - rho) * alpha * dphi0):
                flag = 1
            else:
                a = alpha
                b = b
                if (b < alpham):
                    alpha = (a + b) / 2
                else:
                    alpha = t * alpha
        else:
            a = a
            b = alpha
            alpha = (a + b) / 2
    return alpha


def calLambda(x, y, g):
    return goldsteinsearch(-g,x,y,1,0.1,2)

def delta_grad(x, y):
    g = grad(x, y)
    lambda_ = calLambda(x,y,g)#0.002
    print(lambda_)
    delta = lambda_ * g
    return delta


# ----- 绘制等高线 -----
# 数据数目
n = 256
# 定义x, y
x = np.linspace(-1, 1.1, n)
y = np.linspace(-0.1, 1.1, n)

# 生成网格数据
X, Y = np.meshgrid(x, y)

plt.figure()
# 填充等高线的颜色
plt.contourf(X, Y, f(X, Y), 5, alpha=0, cmap=plt.cm.hot)
# 绘制等高线, 8是等高线分为几部分
C = plt.contour(X, Y, f(X, Y), 8, locator=ticker.LogLocator(), colors='black', linewidth=0.01)
# 绘制等高线数据
plt.clabel(C, inline=True, fontsize=10)
# ---------------------

x = np.matrix([[-0.2],
               [0.4]])

tol = 0.00001
xv = [x[0, 0]]
yv = [x[1, 0]]

plt.text(x[0, 0], x[1, 0],"start")
start = time.time()
for t in range(6000):
    delta = delta_grad(x[0, 0], x[1, 0])
    if abs(delta[0, 0]) < tol and abs(delta[1, 0]) < tol:
        break
    x = x - delta
    xv.append(x[0, 0])
    yv.append(x[1, 0])
end = time.time()
print("iteration:" + str(t))
print(xv[-1])
print(yv[-1])
print("耗时："+str(end-start))
plt.plot(xv, yv, label='track')
# plt.plot(xv, yv, label='track', marker='o')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Steepest Descent for Rosenbrock Function')
plt.legend()
plt.show()
