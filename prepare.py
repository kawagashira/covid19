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
    df['infected_nr']   = df['国内の感染者数_1日ごとの発表数'] / 10000
    df['infected_log_nr'] = np.log10(df['infected_nr'] + 1)
    df['dead_nr']       = df['国内の死者数_1日ごとの発表数'] / 1000
    df['dead_log_nr']   = np.log10(df['国内の死者数_1日ごとの発表数'] + 1)
    print (df['dead_nr'])
    df['日付'] = pd.to_datetime(df['日付'])
    print (df)
    w, w_log, y = [], [], []
    for i in range(len(df) - skip - 1): 
        w.append(np.array(df.loc[i:(i+skip-1), ['infected_nr', 'dead_nr']]))
        #print (np.array(df.loc[i:(i+skip-1), ['infected_nr', 'dead_nr']]))
        w_log.append(np.array(df.loc[i:(i+skip-1), ['infected_log_nr', 'dead_log_nr']]))
        #print (i, np.array(df.loc[(i+skip), ['infected_nr', 'dead_nr']]))
        y.append(np.array(df.loc[(i+skip), ['infected_nr', 'dead_nr']]))
    x = np.array(w)
    x_log = np.array(w_log)
    y = np.array(y)
    #print (x_log)
    #re_x = x.reshape(x.shape[0], x.shape[1], 1)
    #re_x = x
    #re_x_log = x.reshape(x_log.shape[0], x_log.shape[1], 1)
    #re_x_log = x_log
    #re_y = y.reshape(y.shape[0], 1)
    #re_y = y
    #print ('re_x', re_x.shape)
    #return re_x, re_x_log, re_y, df['日付']
    return x, x_log, y, df['日付']


if __name__ == '__main__':

    import pickle
    #i_file = '%s/data/covid19/nhk_news_covid19_domestic_daily_data.csv' % os.environ['HOME']
    #i_file = '%s/data/covid19/nhk_news_covid19_domestic_daily_data-0809.csv' % os.environ['HOME']
    i_file = 'data/csv/nhk_news_covid19_domestic_daily_data-0811.csv'
    X, X_log, y, d = prepare(i_file)
    print ('X', X.shape)
    #with open('covid19-7day.pkl', 'wb') as o_handle:
    #with open('covid19-14day.pkl', 'wb') as o_handle:
    #with open('covid19-14day-0809.pkl', 'wb') as o_handle:
    #with open('covid19-14day-0809.pkl', 'wb') as o_handle:
    o_file = 'data/pkl/covid19-14day-0811-id.pkl'
    with open(o_file, 'wb') as o_handle:
        pickle.dump((X, X_log, y, d), o_handle)
