# coding=utf-8
#%%
from __future__ import print_function, absolute_import
from gm.api import *

def tushare_to_juejin(code: str):
    component = code.split('.')
    if component[1] == "SH":
        return "SHSE." + component[0]
    elif component[1] == "SZ":
        return "SZSE." + component[0]
    else:
        # this code should not be ran
        raise ValueError("should not have this case" + code)

set_token('your token_id')

#%%
import pandas as pd
import numpy as np

SH50 = pd.read_csv("../data/info/000016SH.csv", index_col=0)
KC50 = pd.read_csv("../data/info/000688SH.csv", index_col=0)

all_codes = list(set(list(SH50.con_code.unique()) + list(KC50.con_code.unique())))
all_codes = [tushare_to_juejin(code) for code in all_codes]

#%%

data = history(symbol=all_codes, frequency='1d', start_time='2018-10-08 09:00:00', end_time='2024-04-14 16:00:00',
               fields='open, bob', adjust=ADJUST_POST, df=True)
print(data)