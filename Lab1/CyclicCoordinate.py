#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: lenovo
@file: CyclicCoordinate.py
@time: 2021/5/22 10:26
"""
import math
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker


def f(x, y):
    return (1 - x) ** 2 + 100 * (y - x * x) ** 2


def H(x, y):
    return np.matrix([[1200 * x * x - 400 * y + 2, -400 * x],
                      [-400 * x, 200]])


def grad(x, y):
    return np.matrix([[2 * x - 2 + 400 * x * (x * x - y)],
                      [200 * (y - x * x)]])

def s(n):
    if n%2 == 0:
        return np.matrix([[1],[0]])
    else:
        return np.matrix([[0],[1]])

def advance_retreat_method( x, y, direction: list, step=0, delta=0.1) -> tuple:
    """
    find the initial section of step
    """
    point0 = (x,y)
    alpha0= step

    alpha1 = alpha0 + delta
    point1 = (point0[0]+direction[0]*delta, point0[1]+direction[1]*delta)
    if f(point0[0],point0[1]) < f(point1[0],point1[1]):
        while True:
            delta *= 2
            alpha2 = alpha0 - delta
            point2 = (point0[0] - direction[0] * delta, point0[1] - direction[1] * delta)
            if f(point2[0],point2[1]) < f(point0[0],point0[1]):
                alpha1, alpha0 = alpha0, alpha2
                point1, point0 = point0, point2
            else:
                return alpha2, alpha1
    else:
        while True:
            delta *= 2
            alpha2 = alpha1 + delta
            point2 = (point1[0] + direction[0] * delta, point1[1] + direction[1] * delta)
            if f(point2[0],point2[1]) < f(point1[0],point1[1]):
                alpha0, alpha1 = alpha1, alpha2
                point0, point1 = point1, point2
            else:
                return alpha0, alpha2

def goldsteinsearch(d,x,y, rho):
    '''
    线性搜索子函数
    当前迭代点x,y和当前搜索方向d
    '''
    d = np.squeeze(np.asarray(d))
    a,b = advance_retreat_method(x,y,d)

    golden_num = (math.sqrt(5) - 1) / 2
    p,q= a + (1 - golden_num) * (b - a), a + golden_num * (b - a)
    while abs(b - a) > rho:
        fp = f(x + p * d[0],y + p * d[1])
        fq = f(x + q * d[0],y + q * d[1])
        if fp<fq:
            b,q = q,p
            p = a + (1 - golden_num) * (b - a)
        else:
            a,p = p,q
            q = a + golden_num * (b - a)
    alpha = (a+b)/2
    return alpha


def calLambda(x, y, d):
    return goldsteinsearch(d,x,y,0.1)

def cyc_coo(x,y,n):
    lambda_ = calLambda(x,y,s(n))
    print(lambda_)
    delta = lambda_ * s(n)
    return delta


# ----- 绘制等高线 -----
# 数据数目
n = 256
# 定义x, y
x = np.linspace(-1, 1.1, n)
y = np.linspace(-1, 1.1, n)

# 生成网格数据
X, Y = np.meshgrid(x, y)

plt.figure()
# 填充等高线的颜色, 8是等高线分为几部分
plt.contourf(X, Y, f(X, Y), 5, alpha=0, cmap=plt.cm.hot)
# 绘制等高线
C = plt.contour(X, Y, f(X, Y), 8, locator=ticker.LogLocator(), colors='black', linewidth=0.01)
# 绘制等高线数据
plt.clabel(C, inline=True, fontsize=10)
# ---------------------

x = np.matrix([[-0.3],
               [-0.4]])

tol = 0.01
xv = [x[0, 0]]
yv = [x[1, 0]]

plt.plot(x[0, 0], x[1, 0], marker='o')
start = time.time()
for t in range(10000):
    x = np.matrix([[xv[-1]],
                  [yv[-1]]])
    delta = cyc_coo(x[0, 0], x[1, 0],0)
    x1 = x + delta
    if np.linalg.norm(grad(x1[0,0],x1[1,0])) < tol:
        break
    xv.append(x1[0, 0])
    yv.append(x1[1, 0])
    delta = cyc_coo(x1[0, 0], x1[1, 0],1)
    x2 = x1 + delta
    if np.linalg.norm(grad(x2[0,0],x2[1,0])) < tol:
        break
    xv.append(x2[0, 0])
    yv.append(x2[1, 0])
end = time.time()
print("iteration:" + str(t))
print(xv)
print(yv)
print(xv[-1])
print(yv[-1])
print("耗时："+str(end-start))
plt.plot(xv, yv, label='track')
# plt.plot(xv, yv, label='track', marker='o')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Cyclic Coordinate\'s Method for Rosenbrock Function')
plt.legend()
plt.show()

