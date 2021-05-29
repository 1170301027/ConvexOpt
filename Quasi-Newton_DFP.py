#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: lenovo
@file: Quasi-Newton_DFP.py
@time: 2021/4/25 13:44
"""
from numpy.linalg import linalg

'''拟牛顿法：DFP， BFGS'''
# coding:UTF-8

from numpy import *
from numpy.ma import shape
import matplotlib.pyplot as plt

# fun
def fun(x):
    # return 100 * (x[0, 0] ** 2 - x[1, 0]) ** 2 + (x[0, 0] - 1) ** 2
    x1 = x[0,0]
    x2 = x[1,0]
    return 10 * (x1 ** 2) + x2 ** 2

# gfun
def gfun(x):
    result = zeros((2, 1))
    # result[0, 0] = 400 * x[0, 0] * (x[0, 0] ** 2 - x[1, 0]) + 2 * (x[0, 0] - 1)
    # result[1, 0] = -200 * (x[0, 0] ** 2 - x[1, 0])
    x1 = x[0, 0]
    x2 = x[1, 0]
    result[0, 0] = 20 * x1
    result[1, 0] = 2 * x2
    return result


def dfp(fun, gfun, x0, precision):
    '''
    DFP算法
    :param fun: 目标函数
    :param gfun: 导数
    :param x0:初始值
    :return:
    '''
    result = []
    max_k = 500 # 最大迭代次数
    rho = 0.55
    sigma = 0.4
    m = shape(x0)[0]
    Hessian_k = eye(m)
    k = 0
    while (k < max_k):
        gk = mat(gfun(x0))  # 计算梯度
        print(gk)
        if gk[1][0]**2 + gk[0][0]**2 < precision**2: break
        dk = -mat(Hessian_k) * gk
        m = 0
        mk = 0
        while (m < 20):
            newf = fun(x0 + rho ** m * dk)
            oldf = fun(x0)
            if (newf < oldf + sigma * (rho ** m) * (gk.T * dk)[0, 0]):
                mk = m
                break
            m = m + 1

        # DFP校正
        x = x0 + rho ** mk * dk
        sk = x - x0
        yk = gfun(x) - gk
        if (sk.T * yk > 0):
            Hessian_k = Hessian_k - (Hessian_k * yk * yk.T * Hessian_k) / (yk.T * Hessian_k * yk) + (sk * sk.T) / (sk.T * yk)

        k = k + 1
        x0 = x
        result.append(fun(x0))

    return result

if __name__ == '__main__':

    x0 = mat([[0.1], [1]]) # 矩阵
    result = dfp(fun, gfun, x0, precision=0.01)

    n = len(result)
    x = arange(0, n, 1)
    y = result
    print(y)
    # for i,j in x,y:
    #     plt.plot(i,j,"r*")
    plt.plot(x,y)
    plt.title("DFP")
    plt.show()


