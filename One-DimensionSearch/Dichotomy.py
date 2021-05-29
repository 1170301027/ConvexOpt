#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: lenovo
@file: Dichotomy.py
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

def Dichotomy_search(a, b, esp,sigma):
    data = list()
    data.append([a, b])
    c = (a + b) / 2
    # 预先设置sigma
    x1 = c - sigma/2
    x2 = c + sigma/2
    count = 0
    while b-a>esp:
        print(a,b,x1,x2,function(x1),function(x2))
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
            count += 1
            if count > 3:break
            a = x1
            b = x2
        c = (a + b) / 2
        # 预先设置sigma
        x1 = c - sigma / 2
        x2 = c + sigma / 2
    with open(r"一维搜索（二分法）.txt", mode="w", encoding="utf-8")as a_file:
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
print(Dichotomy_search(low, high, esp, sigma=esp/10))  # 传入区间和列表
drawf(low, high, (high - low) / 2000)  # 默认精度是2000个点


