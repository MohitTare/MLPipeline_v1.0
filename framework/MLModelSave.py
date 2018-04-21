import os
import sys
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline

def modelSave(saveloc, savename, model):
    try:
        with open(saveloc + savename, 'wb') as f:
            joblib.dump(model, f)
    except:
        raise Exception("Model Save Unsuccessful. Please check the logs")

def modelLoad(saveloc, savename):
    try:
        with open(saveloc + savename, 'rb') as f:
            model = joblib.load(f)
            return model
    except Exception, e:
        raise Exception("Model Load Unsuccessful. The error is " + str(e) + "Please check the logs. Exiting the code")
        sys.exit(2)