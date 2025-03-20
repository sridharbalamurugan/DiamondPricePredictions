import os
import sys 
import pandas as pd
from dataclasses import dataclass  
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustomException

# Initialize the data ingestion configuration
@dataclass
class DataIngestionConfig:
    raw_data_path = os.path.join('artifacts', 'raw.csv')
    train_data_path = os.path.join('artifacts', 'train.csv')
    test_data_path = os.path.join('artifacts', 'test.csv')

# Create a data ingestion class
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion initiated")
        try:
            # Ensure file exists before reading
            file_path = (r"C:\Users\DELL\newmlproject\DiamondPricePredictions\notebook\data\gemstone.csv")


            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Dataset file not found: {file_path}")

            df = pd.read_csv(file_path)
            logging.info('Dataset read as pandas dataframe')

            # Ensure artifacts directory exists
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            # Train-test split
            logging.info("Train test split")
            train_set, test_set = train_test_split(df, test_size=0.30, random_state=42)

            # Save train and test data
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("preprocessor pickle is created and ")

            # ✅ **Return train first, then test**
            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            logging.error(f"Error occurred in Data Ingestion: {str(e)}")
            raise CustomException(e, sys)
