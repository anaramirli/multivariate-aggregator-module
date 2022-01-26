# VAR model based on : https://towardsdatascience.com/anomaly-detection-in-multivariate-time-series-with-var-2130f276e5e9
# LSTM based on :  https://towardsdatascience.com/lstm-autoencoder-for-anomaly-detection-e1f4f2ee7ccf


"""Algorithms for multivariate aggregation."""

from tensorflow.keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import regularizers


from sklearn.preprocessing import MinMaxScaler
import scipy.stats as stats
from scipy.stats import multivariate_normal
import numpy as np
import pandas as pd

from statsmodels.tsa.vector_ar.var_model import VAR


def vector_aggregator(train, test, low=1, high=50):
    AIC = {}
    best_aic, best_order = np.inf, 0

    for i in range(low, high):
        model = VAR(endog=train)
        var_result = model.fit(maxlags=i)
        AIC[i] = var_result.aic

        if AIC[i] < best_aic:
            best_aic = AIC[i]
            best_order = i

    print('BEST ORDER', best_order, 'BEST AIC:', best_aic)
    ### TRAIN BEST MULTIVARIATE MODEL ###

    var = VAR(endog=train)
    var_result = var.fit(maxlags=best_order)

    print(var_result.aic)

    ### COMPUTE TRAIN T2 METRIC ###

    residuals_mean = var_result.resid.values.mean(axis=0)
    residuals_std = var_result.resid.values.std(axis=0)

    residuals = (var_result.resid.values - residuals_mean) / residuals_std
    cov_residuals = np.linalg.inv(np.cov(residuals.T))

    T = np.diag((residuals).dot(cov_residuals).dot(residuals.T))

    ### COMPUTE UCL ###

    m = var_result.nobs
    p = var_result.resid.shape[-1]
    alpha = 0.01

    UCL = stats.f.ppf(1 - alpha, dfn=p, dfd=m - p) * \
        (p * (m + 1) * (m - 1) / (m * m - m * p))
    print(UCL)
    pred = []

    for i in range(best_order, len(test)):
        pred.append(var_result.forecast(
            test.iloc[i - best_order:i].values, steps=1))

    pred = np.vstack(pred)
    print(pred.shape)

    residuals_test = test.iloc[best_order:].values - pred
    residuals_test = (residuals_test - residuals_mean) / residuals_std

    T_test = np.diag((residuals_test).dot(cov_residuals).dot(residuals_test.T))
    y_scale = 100
    t_totall = np.concatenate((T, T_test), axis=None)
    data = pd.concat([train, test])

    df_t = pd.DataFrame(T)
    df_t.columns = ['T2_train']

    df_t_totall = pd.DataFrame(t_totall)
    df_t_totall.columns = ['T2']
    return df_t_totall


def normalizer(train, test):
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(train)
    x_test = scaler.transform(test)

    return x_train, x_test


def reshaper(x_train, x_test):
    new_x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
    new_x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])
    return new_x_train, new_x_test


def lstm_model(x_train, n, activation='relu', optimizer='adam', loss='mae'):

    model = Sequential()
    model.add(LSTM(int(n), activation='relu', input_shape=(
        x_train.shape[1], x_train.shape[2]), return_sequences=True))
    model.add(LSTM(int(n/2), activation='relu', return_sequences=True))
    model.add(LSTM(int(n/4), activation='relu', return_sequences=True))
    model.add(LSTM(int(n/8), activation='relu', return_sequences=False))
    model.add(RepeatVector(x_train.shape[1]))
    model.add(LSTM(int(n/8), activation='relu', return_sequences=True))
    model.add(LSTM(int(n/4), activation='relu', return_sequences=True))
    model.add(LSTM(int(n/2), activation='relu', return_sequences=True))
    model.add(LSTM(int(n), activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(x_train.shape[2])))
    model.compile(optimizer='adam', loss='mae')
    model.summary()

    return model


def lstm_fit_model(model, x_train, nb_epochs = 10, batch_size = 31, validation_split=0.1, patience=None):
    
    if patience==None:
                   patience=int(0.1*nb_epochs)
                                
    callback = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    
    history = model.fit(x_train, x_train, epochs=nb_epochs, batch_size=batch_size, validation_split=validation_split, callbacks=[callback]).history

    return history


def predictor(model, data, x_data):
    x_pred = model.predict(x_data)
    x_pred = x_pred.reshape(x_pred.shape[0], x_pred.shape[2])
    x_pred = pd.DataFrame(x_pred, columns=data.columns)
    x_pred.index = data.index

    return x_pred


def lstm_claculate_loss(x_pred, data, x_data):
    scored = pd.DataFrame(index=data.index)
    xdata = x_data.reshape(x_data.shape[0], x_data.shape[2])
    scored['Loss_mae'] = np.mean(np.abs(x_pred - xdata), axis=1)
    scored.head()

    return scored


def gauss_threshold(data):
    threshold = multivariate_normal.pdf(data)
    #threshold = zscore(data)
    return threshold
