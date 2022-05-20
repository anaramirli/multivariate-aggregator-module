"""Multivariate Aggregator module."""

__version__ = '2.0.1'

from typing import Dict, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .core.algorithms import lstm_model, lstm_fit_model

from adtk.transformer import PcaReconstructionError
from adtk.data import validate_series
from sklearn.preprocessing import MinMaxScaler
import scipy.stats as stats
from statsmodels.tsa.vector_ar.var_model import VAR

from tensorflow import keras
import joblib

import numpy as np
import pandas as pd
import os
import shutil
from fastapi.staticfiles import StaticFiles

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
async def multivariate_lstm_train(mvts_data: TrainMVTS):
    """Apply LSTM reconstruction error to aggregate the Multivariate data"""

    train_x = pd.DataFrame.from_dict(mvts_data.train_data.data)

    # normalise
    scaler = MinMaxScaler()
    scaler = scaler.fit(train_x)
    train_x = scaler.transform(train_x)

    # reshape data
    train_x = train_x.reshape(train_x.shape[0], 1, train_x.shape[1])

    model = lstm_model(
        train_x,
        mvts_data.initial_embeding_dim,
        mvts_data.loss
        )
	
    model = lstm_fit_model(
        model = model,
        x_train = train_x,
        nb_epochs = mvts_data.nb_epochs,
        batch_size = mvts_data.batch_size,
        validation_split=mvts_data.validation_split,
        patience=mvts_data.patience
        )

    try:
        path_to_model = os.path.join('data', mvts_data.paths.model)
        model.save(path_to_model)

        path_to_scaler = os.path.join('data', mvts_data.paths.scaler)
        with open(path_to_scaler, 'wb') as fo:
            joblib.dump(scaler, fo)

        return {"dump_status": "model is saved successfully"}
    except Exception as inst:
        return {"dump_status": str(inst)}


@app.post('/aggregate-multivariate-lstm-score', response_model=AggregatedOut)
async def aggregate_multivariate_lstm_score(mvts_data: AggregatedMVTS):
    """Apply LSTM reconstruction error to aggregate the Multivariate data"""

    # load model
    path_to_model = os.path.join('data', mvts_data.paths.model)
    model = keras.models.load_model(path_to_model)

    # get scaler
    path_to_scaler = os.path.join('data', mvts_data.paths.scaler)
    scaler = joblib.load(path_to_scaler)

    # get data
    test_x = pd.DataFrame.from_dict(mvts_data.test_data.data)

    # normalise
    test_x = scaler.transform(test_x)

    # reshape data
    test_x = test_x.reshape(test_x.shape[0], 1, test_x.shape[1])

    # predict
    test_x_pred = model.predict(test_x)

    # get score
    test_score = list(np.mean(np.abs(test_x - test_x_pred), axis=2)[:, 0])

    return AggregatedOut(out=test_score)


@app.post('/best-multivariate-var-order')
async def best_multivariate_var_order(mvts_data: BestVAR):
    """Apply VAR to find best lag order"""

    # get data
    train_data = pd.DataFrame.from_dict(mvts_data.train_data.data)

    # add datetime index to data
    train_data.index = pd.to_datetime(train_data.index, unit='ms')

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
    train_data.index = pd.to_datetime(train_data.index, unit='ms')

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
        path_to_model = os.path.join('data', mvts_data.paths.model)
        with open(path_to_model, 'wb') as fo:
            joblib.dump(var_result, fo)

        return {"dump_status": "model is saved successfully",
                "UCL": UCL}
    except Exception as inst:
        return {"dump_status": str(inst),
                "UCL": UCL}


@app.post('/aggregate-multivariate-var', response_model=AggregatedOut)
async def aggregate_multivariate_var(mvts_data: TestVAR):
    """Return Test T2 metric"""

    # get data
    test_data = pd.DataFrame.from_dict(mvts_data.test_data.data)

    # add datetime index to data
    test_data.index = pd.to_datetime(test_data.index, unit='ms')

    # load var_result
    path_to_model = os.path.join('data', mvts_data.paths.model)
    var_result = joblib.load(path_to_model)

    # compute train t2 metric
    residuals_mean = var_result.resid.values.mean(axis=0)
    residuals_std = var_result.resid.values.std(axis=0)

    residuals = (var_result.resid.values - residuals_mean) / residuals_std
    cov_residuals = np.linalg.inv(np.cov(residuals.T))

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
    T_test = list(np.diag((residuals_test).dot(
        cov_residuals).dot(residuals_test.T)))

    return AggregatedOut(out=T_test)


@app.post('/aggregate-multivariate-pca', response_model=AggregatedOut)
async def aggregate_multivariate_pca(mvts_data: AggregatedPCA):
    """Apply PCA reconstruction error to aggregate the Multivariate data"""

    # get data
    data = pd.DataFrame.from_dict(mvts_data.test_data.data)

    # add datetime index to data
    data.index = pd.to_datetime(data.index, unit='ms')

    # validate data
    data = validate_series(data)

    # get pca reconstruction error
    pca_reconstruction_error = list(PcaReconstructionError(
        mvts_data.principal_component).fit_transform(data).values)

    return AggregatedOut(out=pca_reconstruction_error)


@app.post('/remove-model')
async def remove_models(paths_to_models: ModelPath):
    """Remove models locally stored in container"""
    model_path = os.path.join('data', paths_to_models.model)
    scaler_path = os.path.join('data', paths_to_models.scaler)

    try:
        os.remove(scaler_path)
        shutil.rmtree(model_path)
        return {'message': 'ok'}
    except:
        raise HTTPException(500, detail='Error')


# serve static files
app.mount("/data", StaticFiles(directory="data/", html=True), name="model data")



@app.post('/request-model-files')
async def request_model_files(paths_to_models: ModelPath):
    """Returns paths <model> and to <scaler>"""
    model_path = os.path.join('data', paths_to_models.model)
    scaler_path = os.path.join('data', paths_to_models.scaler)

    return {
        'path_to_model_archive': model_path,
        'path_to_scalar': scaler_path
    }


@app.get('/list-model-files')
async def list_model_files():
    """Returns list of files in data/. This list can be used to download served static files (not directories)."""
    ls = os.listdir('data')
    return {'files': ls}


@app.post('/remove-model-files')
async def remove_model_files(list_file_system_entries: List):
    """Remove files and directories in data/. Files or directories which do not exist are ignored"""
    for file_system_entry in list_file_system_entries:
        path_to_file_system_entry = os.path.join('data', file_system_entry)

        if os.path.isfile(path_to_file_system_entry):
            # file
            os.remove(path_to_file_system_entry)
        else:
            # directory
            shutil.rmtree(path_to_file_system_entry, ignore_errors=True)

    return 'ok'
