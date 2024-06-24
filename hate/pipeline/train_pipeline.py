import sys
from hate.logger import logging
from hate.exception import CustomException
from hate.components.data_ingestion import DataIngestion
from hate.components.data_validation import DataValidation
from hate.entity.config_entity import (DataIngestionConfig, DataValidationConfig)
from hate.entity.artifact_entity import (DataIngestionArtifacts, DataValidationArtifacts)

class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()

    def start_data_ingestion(self) -> DataIngestionArtifacts:
        logging.info("Entered the start_data_ingestion method of DataIngestionTrainPipeline class")
        try:
            logging.info("Getting the data from s3 bucket")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifacts = data_ingestion.initiate_data_ingestion()
            logging.info("Got the train and valid from s3 bucket")
            logging.info("Exited the start_data_ingestion method from TrainPipeline class")
            return data_ingestion_artifacts
        except Exception as e:
            raise CustomException(e, sys)
        


    def start_data_validation(self) -> DataValidationConfig:
        logging.info("Entered the start_data_validation method of DataValidationTrainPipeline class")
        try:
            data_validation = DataValidation(data_validation_config=self.data_validation_config)
            data_validation_artifacts = data_validation.initiate_data_validation()
            logging.info("Exited the start_data_validation method from DataValidationTrainPipeline class")

            return data_validation_artifacts
        
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == '__main__':
    train_pipeline = TrainPipeline()
    data_ingestion_artifacts = train_pipeline.start_data_ingestion()
    if data_ingestion_artifacts:
        train_pipeline.start_data_validation()
