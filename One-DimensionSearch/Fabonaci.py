#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: lenovo
@file: Fabonaci.py
@time: 2021/4/24 20:47
"""

from math import *
import matplotlib.pyplot as plt  # 绘图模块
from pylab import *  # 绘图辅助模块



# 通用函数f(x)靠用户录入
def function(x):
    fx = str_fx.replace("x", "%(x)f")  # 所有的"x"换为"%(x)function"
    return eval(fx % {"x": x})  # 字典类型的格式化字符串，将所有的"x"替换为变量x


# 绘图函数：给定闭区间（绘图间隔），绘图间隔默认为0.05，若区间较小，请自行修改
def drawf(a, b, interp=0.05):
    x = [a + ele * interp for ele in range(0, int((b - a) / interp))]
    y = [function(ele) for ele in x]
    plt.figure(1)
    plt.plot(x, y)
    xlim(a, b)
    title(init_str + "    " + str(esp), color="b")  # 标注函数名
    plt.show()
    return "绘图完成！"


# 在本案例中，使用一次性生成斐波拉契列表的方法算法复杂度更低
# 生成的斐波拉契数列F(0)=0,F(1)=F(2)=1,剩余项和普通斐波拉契数列类似
def Fab_list(Fmax):
    a,b=0,1
    Fablist = [a,b]  # 返回一个列表
    while Fablist[-1]< Fmax:
        a,b = b,a+b
        Fablist.append(b)
    Fablist.pop()  # 舍掉最后一个元素
    return Fablist


def Fabonaci_search(a, b, Fab):
    n = len(Fab)-1  # 获取斐波那契数列的长度
    if n < 3:
        return "精度过低，无法进行斐波那契一维搜索"
    data = list()
    data.append([a, b])
    x1 = a + Fab[n - 2]/Fab[n] * (b - a)
    x2 = a + Fab[n - 1]/Fab[n] * (b - a)
    t = n
    while (t > 3):
        if function(x1) > function(x2):  # 如果f(x1)>f(x2)，则在区间(x1,b)内搜索
            a = x1
            x1 = x2
            x2 = a + Fab[t - 1]/Fab[t] * (b - a)
            plt.plot(x2, function(x2), 'r*')
            data.append([x1, b])
        elif function(x1) < function(x2):  # 如果f(x1)<(x2),则在区间(a,x2)内搜索
            b = x2
            x2 = x1
            x1 = a + Fab[t - 2]/Fab[t]* (b - a)
            plt.plot(x1, function(x1), 'r*')
            data.append([a, x2])
        else:  # 如果f(x1)=function(x2)，则在区间(x1,x2)内搜索
            a = x1
            b = x2
            x1 = a + Fab[t - 2]/Fab[t] * (b - a)
            x2 = a + Fab[t - 1]/Fab[t] * (b - a)
            plt.plot(x1, function(x1), 'r*', x2, function(x2), 'r*')
        data.append([x1, x2])
        t -= 1
    x1 = a + 0.5 * (b - a)  # 斐波那契数列第3项和第2项的比
    x2 = x1 + 0.1 * (b - a)  # 偏离一定距离，人工构造的点
    if function(x1) > function(x2):  # 如果f(x1)>function(x2)，则在区间(x1,b)内搜索
        plt.plot(x2, function(x2), 'r*')
        data.append([x1, b])
        a = x1
    elif function(x1) < function(x2):  # 如果f(x1)<function(x2),则在区间(a,x2)内搜索
        plt.plot(x1, function(x1), 'r*')
        data.append([a, x2])
        b = x2
    else:  # 如果f(x1)=function(x2)，则在区间(x1,x2)内搜索
        plt.plot(x1, function(x1), 'r*', x2, function(x2), 'r*')
        data.append([x1, x2])
    with open(r"一维搜索（斐波那契法）.txt", mode="w", encoding="utf-8")as a_file:
        for i in range(0, len(data)):
            a_file.write("%d：\t" % (i + 1))
            for j in range(0, 2):
                a_file.write("f(%.3f)=%.7f\t" % (data[i][j], function(data[i][j])))
            a_file.write("\n")
    print("写入文件成功！")
    return [a, b]


# init_str = input("请输入一个函数，默认变量为x：\n")  # 输入的最初字符串
# para = input("请依次输入一维搜索的区间a,b和最终区间的精确值（用空格分隔）").split()  # 导入区间
init_str = "3*x^2-21.6*x-1"
para = ["0","25","0.08"]
# init_str = "2*x^2-x-1"
# para = ["-1","1","0.06"]
para = [float(ele) for ele in para]  # 将输入的字符串转换为浮点数
low, high, esp = para  # 输入参数列表（最小值、最大值和最终精度）
str_fx = init_str.replace("^", "**")  # 将所有的“^"替换为python的幂形式"**"
print(Fabonaci_search(low, high, Fab_list(int(ceil((high-low)/esp)))))  # 传入区间和斐波拉契列表
drawf(low, high, (high - low) / 2000)  # 默认精度是2000个点


