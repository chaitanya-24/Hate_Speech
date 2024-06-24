import sys
import os
from hate.logger import logging
from hate.exception import CustomException
from zipfile import ZipFile
from hate.cloud_storage.s3_operation import S3Operation
from hate.entity.config_entity import DataIngestionConfig
from hate.entity.artifact_entity import DataIngestionArtifacts

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config
        self.s3 = S3Operation()

    def get_data_from_s3(self) -> None:
        try:
            logging.info("Entered the get_data_from_s3 method of Data Ingestion class")
            os.makedirs(self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR, exist_ok=True)
            self.s3.sync_folder_from_s3(
                self.data_ingestion_config.BUCKET_NAME,
                self.data_ingestion_config.ZIP_FILE_NAME,
                self.data_ingestion_config.ZIP_FILE_PATH
            )
            logging.info("Exited the get_data_from_s3 method of Data Ingestion class")
        except Exception as e:
            raise CustomException(e, sys)

    def unzip_and_clean(self):
        logging.info("Entered the unzip_and_clean method of Data Ingestion class")
        try:
            if not os.path.exists(self.data_ingestion_config.ZIP_FILE_PATH):
                raise FileNotFoundError(f"{self.data_ingestion_config.ZIP_FILE_PATH} does not exist.")
            with ZipFile(self.data_ingestion_config.ZIP_FILE_PATH, 'r') as zip_ref:
                zip_ref.extractall(self.data_ingestion_config.ZIP_FILE_DIR)
            logging.info("Exited the unzip_and_clean method of Data Ingestion class")
            return self.data_ingestion_config.DATA_ARTIFACTS_DIR, self.data_ingestion_config.NEW_DATA_ARTIFACTS_DIR
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifacts:
        logging.info("Entered the initiate_data_ingestion method of Data Ingestion class")
        try:
            self.get_data_from_s3()
            logging.info("Fetched the data from s3 bucket")
            imbalanced_data_file_path, raw_data_file_path = self.unzip_and_clean()
            logging.info("Unzipped file and split into train and valid")
            data_ingestion_artifacts = DataIngestionArtifacts(
                imbalance_data_file_path=imbalanced_data_file_path,
                raw_data_file_path=raw_data_file_path
            )
            logging.info("Exited the initiate_data_ingestion method of the Data Ingestion class")
            logging.info(f"Data Ingestion artifact: {data_ingestion_artifacts}")
            return data_ingestion_artifacts
        except Exception as e:
            raise CustomException(e, sys)
