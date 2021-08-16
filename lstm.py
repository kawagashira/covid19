#!/usr/bin/env python
#
#                           lstm.py
#

import numpy as np
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, LSTM, Activation
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from keras.regularizers import l2

in_out_neurons = 2

def train(x, y):

    # モデル構築

    # 1つの学習データのStep数(今回は25)
    length_of_sequence = X.shape[1] 
    #in_out_neurons = 1

    model = Sequential()
    """
    model.add(LSTM(
        n_hidden,
        #dropout=0.5,
        #recurrent_dropout=0.5,
        #kernel_regularizer=l2(0.01),
        #bias_regularizer=l2(0.001),
        batch_input_shape=(None, length_of_sequence, in_out_neurons),
        return_sequences=True))
    """
    model.add(LSTM(
        n_hidden,
        #dropout=0.5,
        #recurrent_dropout=0.5,
        #kernel_regularizer=l2(0.01),
        #bias_regularizer=l2(0.001),
        batch_input_shape=(None, length_of_sequence, in_out_neurons),
        return_sequences=False))
    model.add(Dense(in_out_neurons))
    model.add(Activation("linear"))
    #optimizer = Adam(lr=0.001)
    optimizer = RMSprop(lr=0.0001)
    model.compile(loss="mean_squared_error", optimizer=optimizer)

    # TRAIN
    print ('TRAIN')
    early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=20)
    model.fit(x, y,
          batch_size=100,
          epochs=n_epoch,
          validation_split=0.3,
          callbacks=[early_stopping]
          )
    return model


def predict_future(model, X, length):

    skip = X.shape[1]
    future_test = X[-1]
    #future_result = np.empty((1))
    future_result = np.empty((1, in_out_neurons))
    for i in range(length):
        #print ('future_test', future_test.shape)
        test_data = future_test.reshape(1, skip, in_out_neurons)
        batch_predict = model.predict(test_data)
        #print ('batch_predict', batch_predict.shape)
        future_test = np.delete(future_test, 0, axis=0)
        #print ('future_test del', future_test.shape)
        #future_test = np.append(future_test, batch_predict)
        future_test = np.vstack([future_test, batch_predict])
        #print ('future_test app', future_test.shape)
        #future_result = np.append(future_result, batch_predict)
        #print ('future_result', future_result.shape)
        future_result = np.vstack([future_result, batch_predict])
    #print ('return future_result', future_result.shape)
    future_result = np.delete(future_result, 0, axis=0)
    #print ('return future_result', future_result.shape)
    #return future_result.reshape(future_result.shape[0], in_out_neurons)
    return future_result


def plot(y, pred, d, f):

    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    infected_k  = 10000
    dead_k      = 100

    """
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    #plt.plot(y)

    print (d.values)
    print (y)
    plt.plot(d.values[:len(y)], y.reshape(len(y)))
    dayｓ = mdates.DayLocator()
    daysFmt = mdates.DateFormatter('%m/%d')
    ax.xaxis.set_major_locator(days)
    ax.xaxis.set_major_formatter(daysFmt)
    """

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(y[:, 0] * infected_k,      c='k', label='#infected persons')
    ax1.plot(pred[:, 0] * infected_k,   c='b', label='#predicted infection')
    ax1.set_ylabel('#infected persons')
    ax1.legend()

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(y[:, 1] * dead_k,       c='r', label='#dead persons')
    ax2.plot(pred[:, 1] * dead_k,    c='y', label='#predicted death')
    ax2.set_ylabel('#dead persons')
    ax2.legend()
    fig.show()
    print ('GRAPH', f)
    fig.savefig(f)
    return


if __name__ == '__main__':

    import pickle
    import json
    import datetime
    import os

    n_hidden = 30
    #train_size = 559
    #train_size = 560
    train_size = 563
    #n_epoch = 3
    n_epoch = 300

    now = datetime.datetime.now()
    timecode = now.strftime("%y%m%d-%H%M")     # like 210813-0909
    dir = 'model/' + timecode + '/'
    if not os.path.isdir(dir):
        os.mkdir(dir)

    param = {}
    param['train_size'] = train_size

    #with open('covid19-7day.pkl', 'rb') as i_handle:
    #with open('covid19-14day.pkl', 'rb') as i_handle:
    #with open('covid19-14day-0809.pkl', 'rb') as i_handle:
    #i_file = 'data/pkl/covid19-14day-0811.pkl'
    #i_file = 'data/pkl/covid19-14day-0811-id.pkl'
    i_file = 'data/pkl/covid19-14day-0815-id.pkl'
    with open(i_file, 'rb') as i_handle:
        X, X_log, y, d = pickle.load(i_handle)
    print ('X', X.dtype, X.shape)
    X_train, y_train = X[:train_size], y[:train_size]
    #X_train, y_train = X_log[:train_size], y[:train_size]
    model = train(X_train, y_train)
    m_file = 'model/%s/%s.h5' % (timecode, timecode)
    t_file = 'model/%s/%s-model.txt' % (timecode, timecode)
    param['files'] = {}
    param['files']['resource']      = i_file
    param['files']['model']         = m_file
    param['files']['model_desc']    = t_file

    model.save(m_file)
    print ('SAVED MODEL', m_file)
    with open(t_file, 'w') as t_handle:
        model.summary(print_fn=lambda x: t_handle.write(x + '\n'))
    model = None
    model = load_model(m_file)

    param['optimizer'] = {}
    param['optimizer']['name'] = model.optimizer._name
    param['optimizer']['learning_rate'] = float(model.optimizer._hyper['learning_rate'])
    p_file = dir + timecode + '-param.txt'
    param['files']['parameter'] = p_file

    pred = model.predict(X_train)
    print (y_train.shape, pred.shape)
    future_result = predict_future(model, X_train, X.shape[0] - train_size + 90)
    con_pred = np.vstack([pred, future_result])
    print ('y', y.shape, 'con_pred', con_pred.shape)
    g_file = 'model/%s/%s-graph.png' % (timecode, timecode)
    param['files']['analytical_graph'] = g_file
    plot(y, con_pred, d, g_file)

    print ('PARAM', p_file)
    with open(p_file, 'w') as p_handle:
        p_handle.write(json.dumps(param, indent=2))
