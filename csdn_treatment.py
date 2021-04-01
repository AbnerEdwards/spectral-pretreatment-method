# -*- coding: utf-8 -*-
# @Time    : 2020/12/12 11:41
# @Author  : my_name_is_BUG
# @FileName: csdn_treatment.py
# @Software: PyCharm
# @Cnblogs ：https://blog.csdn.net/qq2512446791
# https://blog.csdn.net/weixin_44116061/article/details/107242494
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import pywt
import matplotlib.pyplot as plt

def snv(data_x):  #标准正态变换
    m = data_x.shape[0]
    n = data_x.shape[1]
    data_std = np.std(data_x, axis=1)
    data_average = np.mean(data_x, axis=1)
    data_snv = [[((data_x[i][j] - data_average[i]) / data_std[i]) for j in range(n)] for i in range(m)]
    return data_snv

def MSC(data_x):  #多元散射校正
    mean = np.mean(data_x, axis=0)
    n, p = data_x.shape
    msc_x = np.ones((n, p))
    for i in range(n):
        y = data_x[i, :]
        lin = LinearRegression()
        lin.fit(mean.reshape(-1, 1), y.reshape(-1, 1))
        k = lin.coef_
        b = lin.intercept_
        msc_x[i, :] = (y - b) / k
    return msc_x

def create_x(size, rank):
    x = []
    for i in range(2 * size + 1):
        m = i - size
        row = [m**j for j in range(rank)]
        x.append(row)
    x = np.mat(x)
    return x

def sg(data_x, window_size, rank):
    """
     * Savitzky-Golay平滑滤波函数
     * data - list格式的1×n纬数据
     * window_size - 拟合的窗口大小
     * rank - 拟合多项式阶次
     * ndata - 修正后的值
    """
    m = int((window_size - 1) / 2)
    odata = data_x[:]
    for i in range(m):
        odata.insert(0, odata[0])
        odata.insert(len(odata), odata[len(odata) - 1])
    x = create_x(m, rank)
    b = (x * (x.T * x).I) * x.T
    a0 = b[m]
    a0 = a0.T
    ndata = []
    for i in range(len(data_x)):
        y = [odata[i + j] for j in range(window_size)]
        y1 = np.mat(y) * a0
        y1 = float(y1)
        ndata.append(y1)
    return ndata

def SG(data_x, window_size, rank):
    ans = []
    for i in range(data_x.shape[0]):
        ans.append(sg(list(data_x[i, :]), window_size, rank))
    return np.array(ans)

def move_avg(data_x,n,mode="valid"):
    #滑动平均滤波
    tmp = None
    for i in range(data_x.shape[0]):
        if(i == 0):
            tmp = np.convolve(data_x[i, :], np.ones((n,))/n, mode=mode)
        else:
            tmp = np.vstack((tmp, np.convolve(data_x[i, :], np.ones((n,))/n, mode=mode)))
    return tmp

def D1(data_x): #一阶差分
    temp1 = pd.DataFrame(data_x)
    temp2 = temp1.diff(axis=1)
    temp3 = temp2.values
    return np.delete(temp3, 0, axis=1)

def D2(data_x): #二阶差分
    temp2 = (pd.DataFrame(data_x)).diff(axis=1)
    temp3 = np.delete(temp2.values, 0, axis=1)
    temp4 = (pd.DataFrame(temp3)).diff(axis=1)
    spec_D2 = np.delete(temp4.values, 0, axis=1)
    return spec_D2

def wave(data_x): #小波变换
    def wave_(data_x):
        w = pywt.Wavelet('db8')  # 选用Daubechies8小波
        maxlev = pywt.dwt_max_level(len(data_x), w.dec_len)
        coeffs = pywt.wavedec(data_x, 'db8', level=maxlev)
        threshold = 0.04
        for i in range(1, len(coeffs)):
            coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))
        datarec = pywt.waverec(coeffs, 'db8')
        return datarec
    tmp = None
    for i in range(data_x.shape[0]):
        if (i == 0):
            tmp = wave_(data_x[i])
        else:
            tmp = np.vstack((tmp, wave_(data_x[i])))
    return tmp


