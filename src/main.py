"""Multivariate Aggregator module."""

__version__ = '0.0.3'

from typing import Dict, List
from fastapi import FastAPI
from pydantic import BaseModel

from .core.algorithms import normalizer, reshaper, lstm_model, predictor, lstm_fit_model, lstm_claculate_loss, gauss_threshold, vector_aggregator

from adtk.transformer import PcaReconstructionError
from adtk.data import validate_series
from sklearn.preprocessing import MinMaxScaler
import scipy.stats as stats
from scipy.stats import multivariate_normal
from statsmodels.tsa.vector_ar.var_model import VAR

from tensorflow import keras
import joblib

import time
import numpy as np
import pandas as pd


app = FastAPI(
    title='Multivariate Aggregator module.',
    docs_url='/documentation',
    redoc_url='/redoc',
    description='Multivariate aggregator based on multivariate time series data.',
    version=__version__
)

    
class ModelPath(BaseModel):
    '''Parameters for explanation generation'''
    model: str
    scaler: str
        
    
class MultivariateTimeSeriesData(BaseModel):
    '''Data provided for handling model for MultiVariateTimeseriesData'''
    data: Dict[str, List[float]]

        
class TrainMVTS(BaseModel):
    '''Data provided for traning lstm for MultiVariateTimeseriesData'''

    train_data: MultivariateTimeSeriesData
    paths: ModelPath
    activation: str = 'relu'
    optimizer: str = 'adam'
    loss: str = 'mae'
    nb_epochs: int = 300
    batch_size: int = 64
    validation_split: int = 0.15
    patience = 20
    initial_embeding_dim: int = 128
        
class AggregatedMVTS(BaseModel):
    test_data: MultivariateTimeSeriesData
    paths: ModelPath
    
class BestVAR(BaseModel):
    train_data: MultivariateTimeSeriesData
    low_order: int = 1
    high_order: int = 50
        
class TrainVAR(BaseModel):
    train_data: MultivariateTimeSeriesData
    paths: ModelPath
    order: int = 1
        
class TestVAR(BaseModel):
    test_data: MultivariateTimeSeriesData
    paths: ModelPath
    order: int = 1
    

class AggregatedPCA(BaseModel):
    """Parameters for PCA anomaly detection."""
    test_data: MultivariateTimeSeriesData
    principal_component: int = 1

        
class AggregatedOut(BaseModel):
    '''Aggregated Score'''
    out: List[float]
        

@app.post('/multivariate-lstm-train')
async def aggregate_multivariate_lstm(mvts_data: TrainMVTS):
    """Apply LSTM reconstruction error to aggregate the Multivariate data"""
    
    train_x = pd.DataFrame.from_dict(mvts_data.train_data.data)
    
    # normalise
    scaler = MinMaxScaler()
    scaler = scaler.fit(train_x)
    train_x = scaler.transform(train_x)
    
    # reshape data
    train_x = train_x.reshape(train_x.shape[0], 1, train_x.shape[1])
    
    model = lstm_model(train_x,
                       mvts_data.initial_embeding_dim,
                       mvts_data.loss
                      )
    
    history = lstm_fit_model(model = model,
                             x_train = train_x,
                             nb_epochs = mvts_data.nb_epochs,
                             batch_size = mvts_data.batch_size,
                             validation_split=mvts_data.validation_split,
                             patience=mvts_data.patience
                            )
    try: 
        
        model.save(mvts_data.paths.model)
  
        with open(mvts_data.paths.scaler, 'wb') as fo:
            joblib.dump(scaler, fo)  
            
        return {"dump_status": "model is saved successfully"}
    except Exception as inst:
        return {"dump_status": str(inst)}
    
    
@app.post('/aggregate-multivariate-lstm-score', response_model = AggregatedOut)
async def aggregate_multivariate_lstm(mvts_data: AggregatedMVTS):
    """Apply LSTM reconstruction error to aggregate the Multivariate data"""
    
    
    # load model
    model = keras.models.load_model(mvts_data.paths.model)
    # get scaler
    scaler = joblib.load(mvts_data.paths.scaler)
    
    # get data
    test_x = pd.DataFrame.from_dict(mvts_data.test_data.data)
    
    # normalise
    test_x = scaler.transform(test_x)
    
    # reshape data
    test_x = test_x.reshape(test_x.shape[0], 1, test_x.shape[1])
    
    # predict
    test_x_pred = model.predict(test_x)
    
    # get score
    test_score = list(np.mean(np.abs(test_x - test_x_pred), axis=2)[:,0])
    
    return AggregatedOut(out=test_score)

@app.post('/best-multivariate-var-order')
async def best_multivariate_var_order(mvts_data: BestVAR):
    """Apply VAR to find best lag order"""
    
    # get data
    train_data = pd.DataFrame.from_dict(mvts_data.train_data.data)
    
    # add datetime index to data
    train_data.index=pd.to_datetime(train_data.index, unit='ms')
    
    AIC = {}
    best_aic, best_order = np.inf, 0

    for i in range(mvts_data.low_order, mvts_data.high_order):
        model = VAR(endog=train_data)
        var_result = model.fit(maxlags=i)
        AIC[i] = var_result.aic

        if AIC[i] < best_aic:
            best_aic = AIC[i]
            best_order = i
            
    return {"best_order": best_order}


@app.post('/train-multivariate-var')
async def train_multivariate_var(mvts_data: TrainVAR):
    """Train VAR and return var_result"""
    
    # get data
    train_data = pd.DataFrame.from_dict(mvts_data.train_data.data)
    
    # add datetime index to data
    train_data.index=pd.to_datetime(train_data.index, unit='ms')
    
    # train var
    var = VAR(endog=train_data)
    var_result = var.fit(maxlags=mvts_data.order)
    
    # compute UCL
    m = var_result.nobs
    p = var_result.resid.shape[-1]
    alpha = 0.01

    UCL = stats.f.ppf(1 - alpha, dfn=p, dfd=m - p) * \
        (p * (m + 1) * (m - 1) / (m * m - m * p))
    
    
    # save var results
    try: 
        with open(mvts_data.paths.model, 'wb') as fo:
            joblib.dump(var_result, fo)   
            
        return {"dump_status": "model is saved successfully",
				"UCL": UCL}
    except Exception as inst:
        return {"dump_status": str(inst),
				"UCL": UCL}
		
    
@app.post('/aggregate-multivariate-var', response_model = AggregatedOut)
async def aggregate_multivariate_var(mvts_data: TestVAR):
    """Return Test T2 metric"""
   
    # get data
    test_data = pd.DataFrame.from_dict(mvts_data.test_data.data)
    
    # add datetime index to data
    test_data.index=pd.to_datetime(test_data.index, unit='ms')
    
    # load var_result
    var_result = joblib.load(mvts_data.paths.model)
    
    # compute train t2 metric
    residuals_mean = var_result.resid.values.mean(axis=0)
    residuals_std = var_result.resid.values.std(axis=0)

    residuals = (var_result.resid.values - residuals_mean) / residuals_std
    cov_residuals = np.linalg.inv(np.cov(residuals.T))

    T = np.diag((residuals).dot(cov_residuals).dot(residuals.T))

    # compute UCL for loaded var_result
    m = var_result.nobs
    p = var_result.resid.shape[-1]
    alpha = 0.01

    UCL = stats.f.ppf(1 - alpha, dfn=p, dfd=m - p) * \
        (p * (m + 1) * (m - 1) / (m * m - m * p))
    
    pred = []

    # iterative prediction on test data
    for i in range(mvts_data.order, len(test_data)):
        pred.append(var_result.forecast(
            test_data.iloc[i - mvts_data.order:i].values, steps=1))
    pred = np.vstack(pred)

    # compute test T2 metric
    residuals_test = test_data.iloc[mvts_data.order:].values - pred
    residuals_test = (residuals_test - residuals_mean) / residuals_std

    # get T2 metric scores
    T_test = list(np.diag((residuals_test).dot(cov_residuals).dot(residuals_test.T)))

    return AggregatedOut(out=T_test)



@app.post('/aggregate-multivariate-pca', response_model = AggregatedOut)
async def aggregate_multivariate_pca(mvts_data: AggregatedPCA):
    """Apply PCA reconstruction error to aggregate the Multivariate data"""
    
    # get data
    data = pd.DataFrame.from_dict(mvts_data.test_data.data)
    
    # add datetime index to data
    data.index=pd.to_datetime(data.index, unit='ms')
    
    # validate data
    data = validate_series(data)
    
    # get pca reconstruction error
    pca_reconstruction_error = list(PcaReconstructionError(mvts_data.principal_component).fit_transform(data).values)

    return AggregatedOut(out=pca_reconstruction_error)


# previous ones

# class ParametersLSTM(BaseModel):
#     """Parameters for Multivariate Time Series"""
#     num_features: float = 32
#     num_split: float = 3000
       
    
    
# class ParametersVAR(BaseModel):
#     """Parameters for VAR anomaly detection."""
#     low: float = 1
#     high: float = 50


# class ParametersPCA(BaseModel):
#     """Parameters for PCA anomaly detection."""
#     k: float = 1


# class MultivariateTimeSeriesData(BaseModel):
#     """Data provided for Multivariate aggregation."""
#     train_data: Dict[str, Dict]
#     score_data: Dict[str, Dict]
#     parameters: ParametersLSTM


# class UnivariateTimeSeries(BaseModel):
#     """Aggregated Multivariate Data as an Univariate time series"""
#     univariate_time_series: Dict[str, float]
      


# @app.post('/Ley's ', response_model=UnivariateTimeSeries)
# async def aggregate_multivariate_lstm(time_series_data: MultivariateTimeSeriesData):
#     """Apply LSTM reconstruction error to aggregate the Multivariate data"""
#     # TODO: refactor

#     # print(time_series_data.parameters)
#     train = pd.DataFrame.from_dict(time_series_data.train_data, orient='index')
#     train.index = pd.to_datetime(train.index, unit='ms')

#     test = pd.DataFrame.from_dict(time_series_data.score_data, orient='index')
#     test.index = pd.to_datetime(test.index, unit='ms')

#     x_train, x_test = normalizer(train, test)
#     new_x_train, new_x_test = reshaper(x_train, x_test)
#     model = lstm_model(
#         new_x_train, MultivariateTimeSeriesData.parameters.num_features)
#     history = lstm_fit_model(model, new_x_train)
#     x_pred_train = predictor(model, train, new_x_train)
#     x_pred_test = predictor(model, test, new_x_test)

#     scored_test = lstm_claculate_loss(x_pred_test, test, new_x_test)
#     scored_train = lstm_claculate_loss(x_pred_train, train, new_x_train)

#     scored = pd.concat([scored_train, scored_test])

#     scored = scored.fillna('')

#     return UnivariateTimeSeries(univariate_time_series=scored.to_dict())


# @app.post('/aggregate-multivariate-var', response_model=UnivariateTimeSeries)
# async def aggregate_multivariate_var(time_series_data: MultivariateTimeSeriesData):
#     """Apply VAR to aggregate the Multivariate data"""
#     # TODO: refactor

#     # create pandas Series from dictionary containing the time series
#     train_data = pd.DataFrame.from_dict(
#         time_series_data.train_data, orient='index')
#     train_data.index = pd.to_datetime(train_data.index, unit='ms')

#     test_data = pd.DataFrame.from_dict(
#         time_series_data.score_data, orient='index')
#     test_data.index = pd.to_datetime(test_data.index, unit='ms')

#     autoregression_vector = vector_aggregator(train_data, test_data)

#     return UnivariateTimeSeries(univariate_time_series=autoregression_vector)


# @app.post('/aggregate-multivariate-pca', response_model=UnivariateTimeSeries)
# async def aggregate_multivariate_pca(time_series_data: MultivariateTimeSeriesData):
#     """Apply PCA reconstruction error to aggregate the Multivariate data"""
#     # TODO: refactor

#     # create pandas Series from dictionary containing the time series
#     train_data = pd.DataFrame.from_dict(
#         time_series_data.train_data, orient='index')
#     train_data.index = pd.to_datetime(train_data.index, unit='ms')

#     test_data = pd.DataFrame.from_dict(
#         time_series_data.score_data, orient='index')
#     test_data.index = pd.to_datetime(test_data.index, unit='ms')

#     df_test = validate_series(test_data)
#     pca_reconstruction_error = PcaReconstructionError(
#         ParametersPCA.k).fit_transform(test_data).rename("PCA Reconstruction Error")

#     return UnivariateTimeSeries(univariate_time_series=pca_reconstruction_error)


# # This is not an modularized endpoint but in case of need
# @app.post('/lstm-multivariate-aggregator')
# async def multivariate_aggregator(time_series_data: MultivariateTimeSeriesData):
#     """Apply LSTM reconstruction error to aggregate the Multivariate data and return dictionary of the errors and threshols"""
#     # TODO: refactor

#     print(time_series_data.parameters)
#     train = pd.DataFrame.from_dict(time_series_data.train_data, orient='index')
#     test = pd.DataFrame.from_dict(time_series_data.score_data, orient='index')

#     x_train, x_test = normalizer(train, test)
#     new_x_train, new_x_test = reshaper(x_train, x_test)
#     model = lstm_model(new_x_train)
#     history = lstm_fit_model(model, new_x_train)
#     x_pred_train = predictor(model, train, new_x_train)
#     x_pred_test = predictor(model, test, new_x_test)

#     scored_test = lstm_claculate_loss(x_pred_test, test, new_x_test)
#     scored_train = lstm_claculate_loss(x_pred_train, train, new_x_train)

#     scored = pd.concat([scored_train, scored_test])

#     input = scored_train['Loss_mae']
#     threshold = gauss_threshold(input)
#     scored['Threshold'] = threshold.mean() + (2 * threshold.std())
#     scored['Threshold2'] = np.mean(
#         scored_train['Loss_mae']) + (2 * np.std(scored_train['Loss_mae']))
#     scored['Anomaly'] = scored['Loss_mae'] > scored['Threshold']
#     scored['em_score'] = scored['Loss_mae'].ewm(com=0.5).mean()
#     scored['rolling_threshold'] = scored['Loss_mae'].rolling(window=500).mean()
#     scored['rolling_threshold'] = scored['rolling_threshold'] + \
#         np.std(scored['Loss_mae'])

#     scored = scored.fillna('')

#     return scored.to_dict()
