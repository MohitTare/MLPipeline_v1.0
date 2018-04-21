from MLTrain.__init__ import main
import os
import sys
import logging
import pandas as pd
import numpy as np
from framework.MLConfigReader import MLConfigReaderObj
from framework.MLModelSave import modelLoad
from DataPreparation.DataPrepare import getData
from sklearn.pipeline import Pipeline


def modelPredict(cfg):
    logger = logging.getLogger(__name__)
    modelpath = cfg.get('MODELLOADING', 'saveloc', fallback=None)
    modelname = cfg.get('MODELLOADING', 'savename', fallback=None)
    if (modelpath is None) or (modelname is None):
        logger.info("Either the model save path or the modelname is not provided. Please recheck and run again")
        sys.exit(2)
    logger.info("Reading the Test/Prediction Data")
    datapath = cfg.get("DATA", "path", fallback = None)
    if datapath is None:
        logger.info("No Test/Prediciton Data Provided. Please check the config and rerun")
        sys.exit(2)
    if os.path.exists(datapath):
        test_df = getData(datapath)
    logger.info("Loading the model")
    model = modelLoad(modelpath, modelname)
    y_pred = model.predict(test_df)
    dict_mapping = {'0': 'Iris-setosa','1': 'Iris-versicolor', '2': 'Iris-virginica'}
    y_pred = list(map(lambda x: dict_mapping[str(x)], y_pred))
    logger.info("The Predicted class is  %s", y_pred)


if __name__ == '__main__':
    main(modelPredict)
