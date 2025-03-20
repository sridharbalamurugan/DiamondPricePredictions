import os
import sys
import joblib
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            joblib.dump(obj, file_obj)

        logging.info(f"Object saved successfully at {file_path}")  

    except Exception as e:
        logging.error("Error while saving object", exc_info=True)
        raise CustomException(e, sys)

def evaluate_model(X_train, X_test, y_train, y_test, models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            
            # Train model
            model.fit(X_train, y_train)    
            
            # Predicting on test data
            y_test_pred = model.predict(X_test)
            
            # Get R2 score
            test_model_score = r2_score(y_test, y_test_pred)
            
            report[list(models.keys())[i]] = test_model_score 
        
        return report
    
    except Exception as e:
        logging.info("Exception occurred during model evaluation")
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return joblib.load(file_obj)
    except Exception as e:
        logging.info('Except Ocurred in load_object funtion utils')
        raise CustomException(e,sys)
        

  