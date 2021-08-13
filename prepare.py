#!/usr/bin/env python
#
#                                   prepare.py
#

import os
import pandas as pd
import numpy as np


def prepare(i_file):

    skip = 14
    #df = pd.read_csv(i_file, index_col='日付', parse_dates=True)
    df = pd.read_csv(i_file)
    df['infected_nr'] = df['国内の感染者数_1日ごとの発表数'] / 10000
    df['infected_log1p_nr'] = np.log1p(df['infected_nr'])
    df['日付'] = pd.to_datetime(df['日付'])
    print (df)
    #df.set_index('日付', inplace=True)
    w, w_log, y = [], [], []
    for i in range(len(df) - skip - 1): 
        #print (i, i+skip-1)
        w.append(df.loc[i:(i+skip-1), 'infected_nr'])
        w_log.append(df.loc[i:(i+skip-1), 'infected_log1p_nr'])
        y.append(df.loc[(i+skip), 'infected_nr'])
    x = np.array(w)
    x_log = np.array(w_log)
    y = np.array(y)
    re_x = x.reshape(x.shape[0], x.shape[1], 1)
    re_x_log = x.reshape(x_log.shape[0], x_log.shape[1], 1)
    re_y = y.reshape(y.shape[0], 1)
    print ('re_x', re_x.shape)
    return re_x, re_x_log, re_y, df['日付']


if __name__ == '__main__':

    #i_file = '%s/data/covid19/nhk_news_covid19_domestic_daily_data.csv' % os.environ['HOME']
    #i_file = '%s/data/covid19/nhk_news_covid19_domestic_daily_data-0809.csv' % os.environ['HOME']
    i_file = '%s/data/covid19/nhk_news_covid19_domestic_daily_data-0811.csv' % os.environ['HOME']
    X, X_log, y, d = prepare(i_file)
    import pickle
    #with open('covid19-7day.pkl', 'wb') as o_handle:
    #with open('covid19-14day.pkl', 'wb') as o_handle:
    #with open('covid19-14day-0809.pkl', 'wb') as o_handle:
    with open('covid19-14day-0811.pkl', 'wb') as o_handle:
        pickle.dump((X, X_log, y, d), o_handle)
