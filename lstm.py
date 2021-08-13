#!/usr/bin/env python
#
#                           lstm.py
#

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from keras.regularizers import l2

def train(x, y):

    # モデル構築

    # 1つの学習データのStep数(今回は25)
    length_of_sequence = X.shape[1] 
    in_out_neurons = 1
    n_hidden = 300
    #n_hidden = 100
    #n_hidden = 30
    #n_hidden = 10

    model = Sequential()
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
          epochs=3,
          validation_split=0.3,
          callbacks=[early_stopping]
          )
    return model


def predict_future(model, X, length):

    skip = X.shape[1]
    future_test = X[-1]
    future_result = np.empty((1))
    for i in range(length):
        test_data = future_test.reshape(1, skip, 1)
        batch_predict = model.predict(test_data)
        future_test = np.delete(future_test, 0)
        future_test = np.append(future_test, batch_predict)
        future_result = np.append(future_result, batch_predict)
    future_result = np.delete(future_result, 0)
    #print (future_result.reshape(future_result.shape[0], 1))
    return future_result.reshape(future_result.shape[0], 1)


def plot(y, pred, d):

    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

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

    plt.plot(y, label='#infected persons')
    plt.plot(pred, label='#predicted')
    plt.ylabel('#persons (x 10000)')
    plt.legend()
    plt.show()


if __name__ == '__main__':

    import pickle
    #train_size = 552
    #train_size = 557
    train_size = 559

    #with open('covid19-7day.pkl', 'rb') as i_handle:
    #with open('covid19-14day.pkl', 'rb') as i_handle:
    #with open('covid19-14day-0809.pkl', 'rb') as i_handle:
    with open('covid19-14day-0811.pkl', 'rb') as i_handle:
        X, X_log, y, d = pickle.load(i_handle)
    print ('X', X.shape)
    X_train, y_train = X[:train_size], y[:train_size]
    #X_train, y_train = X_log[:train_size], y[:train_size]
    model = train(X_train, y_train)
    model.save('model.hdf5')
    #print (model.__dict__)
    #with open('model.pkl', 'wb') as m_handle:
    #    pickle.dump(model, m_handle)
    pred = model.predict(X_train)
    print (y_train.shape, pred.shape)
    #plot(y, pred)
    future_result = predict_future(model, X_train, X.shape[0] - train_size + 90)
    con_pred = np.vstack([pred, future_result])
    print (y.shape, con_pred.shape)
    plot(y, con_pred, d)
