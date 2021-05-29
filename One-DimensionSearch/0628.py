#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: lenovo
@file: 0628.py
@time: 2021/4/24 20:20
"""

from pylab import *
# 通用函数f(x)靠用户录入
def function(x):
    fx = str_fx.replace("x", "%(x)f")  # 所有的"x"换为"%(x)function"
    return eval(fx % {"x": x})  # 字典类型的格式化字符串，将所有的"x"替换为变量x


# 绘图函数：给定闭区间（绘图间隔），绘图间隔默认为0.05，若区间较小，请自行修改
def drawf(a,b,interp=0.05):
    x = [a+ele*interp for ele in range(0, int((b-a)/interp))]
    y = [function(ele) for ele in x]
    plt.figure(1)
    plt.plot(x, y)
    xlim(a, b)
    title(init_str + "    " + str(esp), color="b")
    plt.show()

# 黄金分割法进行一维搜索的函数
def gold_div_search(a,b,esp):
    data=list()
    x1=a+rou*(b-a)
    x2=b-rou*(b-a)
    data.append([a,x1,x2,b])
    while(b-a>esp):
        if function(x1)>function(x2):  #如果f(x1)>function(x2)，则在区间(x1,b)内搜索
            a=x1
            x1=x2
            x2=b-rou*(b-a)
            plt.plot(x2,function(x2),'r*')
        elif function(x1)<function(x2):  #如果f(x1)<function(x2),则在区间(a,x2)内搜索
            b=x2
            x2=x1
            x1=a+rou*(b-a)
            plt.plot(x1,function(x1),'r*')
        else:  #如果f(x1)=function(x2)，则在区间(x1,x2)内搜索
            a=x1
            b=x2
            x1=a+rou*(b-a)
            x2=b-rou*(b-a)
            plt.plot(x1,function(x1),'r*',x2,function(x2),'r*')
        data.append([a,x1,x2,b])
    with open("一维搜索（黄金分割法）.txt",mode="w",encoding="utf-8")as a_file:
    # 保存的txt文件在程序的同目录下
        for i in range(0,len(data)):
            a_file.write("%d：\t"%(i+1))
            for j in range(0,4):
                a_file.write("function(%.3f)=%.3f\t"%(data[i][j],function(data[i][j])))
            a_file.write("\n")
    print("写入文件成功！")
    return [a,b]

rou = 1-(sqrt(5)-1)/2  # 1-rou为黄金分割比
# init_str = input("请输入一个函数，默认变量为x：\n")  # 输入的最初字符串
# para=input("请依次输入一维搜索的区间a,b和最终区间的精确值（用空格分隔）").split() # 导入区间
# init_str = "3*x^2-21.6*x-1"
# para = ["0","25","0.08"]
init_str = "2*x^2-x-1"
para = ["-1","1","0.06"]
para=[float(ele) for ele in para]
a,b,esp=para
str_fx = init_str.replace("^", "**")  # 将所有的“^"替换为python的幂形式"**"
gold_div_search(a,b,esp)  # 调用黄金分割法并保存文件
drawf(a,b,(b-a)/2000)  # 绘制函数图形

