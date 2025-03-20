from sklearn.impute import SimpleImputer  # Handling missing values
from sklearn.preprocessing import OrdinalEncoder # Handling feature Scaling
from sklearn.preprocessing import StandardScaler # Ordinal Encoding
# pipelines

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import sys,os
from dataclasses import dataclass 

import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


## Data Transformation Config

@dataclass
class DataTransformationconfig:
    preprocessor_ob_file_path=os.path.join('artifacts','preprocessor.pkl')

## Data Transformationclass
class DataTransformation:
    def __init__(self,config=None):
        if config is None:
          self.data_transformation_config= DataTransformationconfig()
        else:
          self.data_transformation_config=config 

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')

            # Define which columns should be ordinal_enclosed and which should be scaled

            categorical_cols = ['cut','color','clarity']
            numerical_cols = ['carat','depth','table','x','y','z']

            # Define the custom ranking for each ordinal variable

            cut_categories = ['Fair','Good','Very Good','Premium','Ideal']
            color_categories = ['J','I','H','G','F','E','D']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']
        
            logging.info('pipeline Initiated')

            # Numerical pipeline
            num_pipeline = Pipeline(
               steps = [
                   ('imputer',SimpleImputer(strategy='median')),
                   ('scalar', StandardScaler())
            
               ]
            ) 
            
            # Categorical pipeline
            cat_pipeline =Pipeline(
                steps = [
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('ordinalencoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                    ('scalar',StandardScaler())
                    ]
            )
            preprocessor = ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_cols),
            ('cat_pipeline',cat_pipeline,categorical_cols)
            ])
          
            return preprocessor
            
            logging.info('Pipeline completed')
    
        except Exception as e:

            logging.error('Error occured in Data Transformation')
            raise CustomException(e,sys)
            
    def initiate_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head : \n{test_df.head().to_string()}')


            logging.info('obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'price'

            drop_columns = [target_column_name,'Unnamed: 0']

            # features into independent and dependent variables
        
            input_features_train_df = train_df.drop(columns=drop_columns,axis = 1)
            target_feature_train_df = train_df[target_column_name]

            input_features_test_df = test_df.drop(columns=drop_columns,axis = 1)
            target_feature_test_df = test_df[target_column_name]
            # apply the transformation
            input_features_train_arr=preprocessing_obj.fit_transform(input_features_train_df)
            input_features_test_arr=preprocessing_obj.transform(input_features_test_df)

            logging.info('Applying preprocessing object on training and testing datasets.')

            train_arr =  np.c_[input_features_train_arr,np.array(target_feature_train_df)]
            test_arr= np.c_[input_features_test_arr,np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_ob_file_path,
                obj=preprocessing_obj
            )
            logging.info('processor joblib is created and saved')
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path
            )
            
        except Exception as e:
            logging.error('Error occured in the Initiate_data_transformation')
            raise CustomException(e,sys)
       
            