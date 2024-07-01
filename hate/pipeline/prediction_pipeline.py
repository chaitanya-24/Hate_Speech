import os 
import sys
import keras
import pickle
from PIL import Image
from hate.constants import *
from hate.logger import logging
from hate.exception import CustomException
from hate.components.data_transforamation import DataTransformation
from hate.entity.config_entity import DataTransformationConfig
from hate.entity.artifact_entity import DataIngestionArtifacts
from hate.cloud_storage.s3_operation import S3Operation
from keras.utils import pad_sequences


class PredictionPipeline:
    def __init__(self):
        self.bucket_name = BUCKETNAME
        self.model_name = MODEL_NAME
        self.model_file = "model.h5"  # Specify the model file name
        self.model_folder = "model.h5"  # Specify the folder name in the S3 bucket
        self.model_path = os.path.join("artifacts", "PredictModel")
        self.s3operation = S3Operation()
        self.data_transformation = DataTransformation(data_transformation_config=DataTransformationConfig, data_ingestion_artifacts=DataIngestionArtifacts)


    def get_model_from_s3(self) -> str:
        """
        Method Name: get_model_from_s3
        Description: This method reads the model from s3 bucket
        Output: Returns the model path (best_model_path)
        """
        logging.info("Entered get_model_from_s3 method of PredictionPipeline class")
        try:
            os.makedirs(self.model_path, exist_ok=True)
            # Custom command to download only model.h5
            command = f"aws s3 cp s3://{self.bucket_name}/{self.model_folder}/{self.model_file} {self.model_path}/{self.model_file}"
            logging.info(f"Executing command: {command}")
            os.system(command)
            best_model_path = os.path.join(self.model_path, self.model_file)
            logging.info(f"Model should be at path: {best_model_path}")
            if not os.path.exists(best_model_path):
                logging.error(f"Model file not found at path: {best_model_path}")
                raise FileNotFoundError(f"No file or directory found at {best_model_path}")
            logging.info("Exited the get_model_from_s3 method of PredictPipeline class")
            return best_model_path

        except Exception as e:
            raise CustomException(e, sys) 
        

    def predict(self, best_model_path, text):
        """load image, returns cuda tensors"""
        logging.info("Running the predict function")

        try:
            load_model = keras.models.load_model(best_model_path)
            with open('tokenizer.pickle', 'rb') as handle:
                load_tokenizer = pickle.load(handle)

            text = self.data_transformation.concat_data_cleaning(text)
            text = [text]
            print(text)

            seq = load_tokenizer.texts_to_sequences(text)
            padded = pad_sequences(seq, maxlen=300)
            print(seq)
            pred = load_model.predict(padded)
            pred
            print("pred", pred)
            if pred > 0.5:
                print("Hate and Abusive")
                return "Hate and Abusive"
            
            else:
                print("No Hate")
                return "No Hate"

        except Exception as e:
            raise CustomException(e, sys)
        

    def run_pipeline(self, text):
        logging.info("Entered the run_pipeline method of PredictionPipeline class")
        try:
            best_model_path = self.get_model_from_s3()
            prediction = self.predict(best_model_path, text)
            logging.info("Exited the run_pipeline method of PredictionPipeline class")
            return prediction

        except Exception as e:
            raise CustomException(e, sys)
