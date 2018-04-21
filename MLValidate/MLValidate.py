import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

def modelValidate(y_pred,y_actual,metric_type):
    if str(metric_type) == 'classification':
        return classification_report(y_actual, y_pred)