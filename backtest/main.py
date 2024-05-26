# coding=utf-8
from __future__ import print_function, absolute_import
from gm.api import *
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# The reason of setting as this is GPT only give "正面" and "中性", which needs "中性" to be negative.
SIGNAL_MAP = {"极度正面": 5,
              "正面": 1,
              "中性": -1,
              "负面": -3,
              "极度负面": -5}


def init(context):
    # run algo every day, and only can pass context variables
    schedule(schedule_func=algo, date_rule='1d', time_rule='09:30:00')
    context.data = pd.read_csv("result/Quant-GPT2.csv", index_col=0)
    context.pointer = 0

def tushare_to_juejin(code: str):
    component = code.split('.')
    if component[1] == "SH":
        return "SHSE." + component[0]
    elif component[1] == "SZ":
        return "SZSE." + component[0]
    else:
        # this code should not be ran
        raise ValueError("should not have this case" + code)

def algo(context):
    order_close_all()
    signal_time = context.data.iloc[context.pointer, 1]
    signal_time = datetime.strptime(signal_time, '%Y-%m-%d %H:%M:%S%z')
    signals = {}
    while signal_time < context.now:
        if signals.get(context.data.iloc[context.pointer, -1], 0) == 0:
            signals[context.data.iloc[context.pointer, -1]] = []
        signals[context.data.iloc[context.pointer, -1]].append(SIGNAL_MAP[context.data.iloc[context.pointer, 0]])
        context.pointer += 1
        if context.pointer == len(context.data):
            break
        signal_time = context.data.iloc[context.pointer, 1]
        signal_time = datetime.strptime(signal_time, '%Y-%m-%d %H:%M:%S%z')
    
    sum_exp_avg = 0
    for stock_code, signal in signals.items():
        exp_avg_signal = 0
        sum_weight = 0
        epsilon = 1
        lambda_coef = 1.01
        for s in signal:
            sum_weight += epsilon
            exp_avg_signal += epsilon * s
            epsilon *= lambda_coef
        exp_avg_signal /= sum_weight
        if exp_avg_signal < 0:
            exp_avg_signal = 0
        sum_exp_avg += exp_avg_signal
        signals[stock_code] = exp_avg_signal
    if sum_exp_avg > 0:
        for stock_code, signal in signals.items():
            order_percent(symbol=tushare_to_juejin(stock_code), percent=signal/sum_exp_avg, side=OrderSide_Buy,
                 order_type=OrderType_Market, position_effect=PositionEffect_Open, price=0)

# 查看最终的回测结果
def on_backtest_finished(context, indicator):
    print(indicator)


if __name__ == '__main__':
    '''
        strategy_id策略ID, 由系统生成
        filename文件名, 请与本文件名保持一致
        mode运行模式, 实时模式:MODE_LIVE回测模式:MODE_BACKTEST
        token绑定计算机的ID, 可在系统设置-密钥管理中生成
        backtest_start_time回测开始时间
        backtest_end_time回测结束时间
        backtest_adjust股票复权方式, 不复权:ADJUST_NONE前复权:ADJUST_PREV后复权:ADJUST_POST
        backtest_initial_cash回测初始资金
        backtest_commission_ratio回测佣金比例
        backtest_slippage_ratio回测滑点比例
        backtest_match_mode市价撮合模式，以下一tick/bar开盘价撮合:0，以当前tick/bar收盘价撮合：1
    '''
    run(strategy_id='your id',
        filename='main.py',
        mode=MODE_BACKTEST,
        token='your token',
        backtest_start_time='2023-04-14 08:00:00',
        backtest_end_time='2024-04-11 16:00:00',
        backtest_adjust=ADJUST_POST,
        backtest_initial_cash=20000000,
        backtest_commission_ratio=0,
        backtest_slippage_ratio=0,
        backtest_match_mode=1)

