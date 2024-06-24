import pandas as pd
import os
import sys
from hate.logger import logging
from hate.exception import CustomException
from hate.entity.config_entity import DataValidationConfig
from hate.entity.artifact_entity import DataValidationArtifacts
from hate.constants import RAW_DATA_COLUMNS, IMBALANCED_DATA_COLUMNS

class DataValidation:
    def __init__(self, data_validation_config: DataValidationConfig):
        self.data_validation_config = data_validation_config

    def validate_columns(self, file_path: str, required_columns: list) -> bool:
        try:
            df = pd.read_csv(file_path)
            df_columns = df.columns.tolist()
            return all(column in df_columns for column in required_columns)
        except Exception as e:
            raise CustomException(e, sys)
        

    def initiate_data_validation(self) -> DataValidationArtifacts:
        logging.info("Entered the initiate_data_validation method of DataValidation class")
        try:
            imbalanced_data_valid = self.validate_columns(self.data_validation_config.IMBALANCED_DATA_PATH, IMBALANCED_DATA_COLUMNS)
            raw_data_valid = self.validate_columns(self.data_validation_config.RAW_DATA_PATH, RAW_DATA_COLUMNS)

            report_lines = [
                f"Imbalanced data validation results: {imbalanced_data_valid}", 
                f"Raw data validation results: {raw_data_valid}"
            ]
            report_content = "\n".join(report_lines)

            os.makedirs(os.path.dirname(self.data_validation_config.REPORT_FILE_PATH), exist_ok=True)
            with open(self.data_validation_config.REPORT_FILE_PATH, 'w') as f:
                f.write(report_content)

            data_validation_artifacts = DataValidationArtifacts(report_file_path=self.data_validation_config.REPORT_FILE_PATH)
            logging.info(f"Data validation artifact: {data_validation_artifacts}")
            logging.info("Exited the initiate_data_validation method of DataValidation class")
            return data_validation_artifacts
        

        except Exception as e:
            raise CustomException(e, sys)