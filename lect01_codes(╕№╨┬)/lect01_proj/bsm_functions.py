# -*- coding: utf-8 -*-

"""
    作者:     梁斌
    版本:     1.0
    日期:     2017/05/01
    项目名称：使用Python实现蒙特卡洛模拟的期权估值
    项目参考：《Python金融大数据分析》第3章
"""

from __future__ import division, print_function
from math import log, sqrt, exp
from scipy import stats


def bsm_call_value(S0, K, T, r, sigma):
    """
        根据BSM公式计算期权估值
        
        参数
        ======
        S0:     初始标的物价格，即t=0
        K:      期权行权价格
        T:      期权到期日
        r:      固定无风险短期利率
        sigma:  标的物固定波动率
        
        返回值
        ======
        value:  当前期权定价
    """
    S0 = float(S0)
    d1 = (log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = (log(S0 / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    value = (S0 * stats.norm.cdf(d1, 0.0, 1.0) - K * exp(-r * T) * stats.norm.cdf(d2, 0.0, 1.0))
    return value
