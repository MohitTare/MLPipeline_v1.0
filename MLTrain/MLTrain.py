from __init__ import main
import os
import sys
import pandas as pd
import numpy as np
from framework.MLConfigReader import MLConfigReaderObj
from DataPreparation.DataPrepare import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,accuracy_score,recall_score,roc_auc_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder,StandardScaler
from MLValidate.MLValidate import modelValidate
from framework.MLModelSave import modelSave

def modelTrain(cfg):
    logger = logging.getLogger(__name__)
    logger.info("Fetching the parameters from the config")
    datapath = cfg.get('DATA', 'path', fallback=None)
    if datapath is None:
        logger.info("No data provided for training the model")
        sys.exit(2)
    logger.info("Fetching input data for training")
    train_df = getData(datapath)
    logger.info("The fetched data has the following dimension - %s", train_df.shape)
    logger.info("Fetching the tuning parameters for training")
    param = dict()
    featurelist = cfg.get('TRAINING', 'features', fallback='all')
    if featurelist == 'all':
        param['featurelist'] = list(train_df.columns)
    param['featurelist'] = list(featurelist.split(','))
    param['target'] = cfg.get('TRAINING', 'target', fallback = None)
    if param['target'] is None:
        logger.info('No Target Variable provided for training the model. Exiting the code')
        sys.exit(2)
    param['penalty'] = cfg.get('TRAINING', 'penalty', fallback='l2')
    param['class_weight'] = cfg.get('TRAINING', 'class_weight', fallback=None)
    param['C'] = cfg.getfloat('TRAINING', 'C', fallback = 1.0)
    param['fit_intercept'] = cfg.getboolean('TRAINING', 'fit_intercept', fallback=True)
    param['intercept_scaling'] = cfg.getint('TRAINING', 'intercept_scaling', fallback=1)
    param['solver'] = cfg.get('TRAINING', 'solver', fallback='liblinear')
    param['max_iter'] = cfg.getint('TRAINING', 'max_iter', fallback=100)
    param['multi_class'] = cfg.get('TRAINING', 'multi_class', fallback='ovr')
    param['warm_start'] = cfg.getboolean('TRAINING', 'warm_start', fallback=False)
    param['n_jobs'] = cfg.get('TRAINING', 'n_jobs', fallback=1)
    logger.info('The tuning parameters are - %s \n', param)
    logger.info('Training model with the above tuning parameters')
    model_lr = LogisticRegression(penalty=param['penalty'],C=param['C'],fit_intercept=param['fit_intercept'],
                                intercept_scaling=param['intercept_scaling'],class_weight=param['class_weight'],
                                solver=param['solver'],max_iter=param['max_iter'],multi_class=param['multi_class'],
                                warm_start=param['warm_start'],n_jobs=param['n_jobs'])
    lb = LabelEncoder()
    y = train_df.loc[:,param['target']]
    X = train_df.loc[:, param['featurelist']]
    lb.fit(y)
    y = lb.transform(y)
    #TODO Check why LabelEndoder cant be included in sklearn pipeline
    pipe = Pipeline([('ss', StandardScaler()), ('lr', model_lr)])
    pipe.fit(X, y)
    cv = cfg.getboolean('CROSSVALIDATION', 'gencv', fallback=True)
    if cv is True:
        cv_cs = cfg.get('CROSSVALIDATION', 'Cs', fallback='0,1,10,100')
        cv_cs = list(map(float, cv_cs.split(',')))
        cv_round = cfg.getint('CROSSVALIDATION', 'cv', fallback=3)
        modelcv = LogisticRegressionCV(Cs=cv_cs, scoring='accuracy', cv=cv_round)
        modelcv.fit(X, y)
        logger.info("Cross validation completed. The max cv score is - \n %s", modelcv.scores_[1].max())
    is_validate = cfg.getboolean('VALIDATION', 'genstats', fallback=False)
    if is_validate is True:
        logger.info("Generating the score for Training data")
        val_metrictype = cfg.get('VALIDATION', 'metrictype', fallback='classification')
        val_results = modelValidate(model_lr.predict(X), y, val_metrictype)
        logger.info("Training set classification results are - \n %s", val_results)
    is_modelsave = cfg.getboolean('MODELSAVING', 'savemodel', fallback=False)
    if is_modelsave is True:
        logger.info("Serializing and Saving the model")
        saveloc = cfg.get('MODELSAVING', 'saveloc', fallback=os.getcwd())
        savename = cfg.get('MODELSAVING', 'savename', fallback='model.pk')
        modelSave(saveloc,savename,pipe)
        logger.info("Model Serialized and Saved successfully")


if __name__ == '__main__':
    main(modelTrain)