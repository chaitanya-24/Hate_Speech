import os
import sys
import keras 
import pickle
import numpy as np
import pandas as pd
from hate.exception import CustomException
from hate.logger import logging
from hate.constants import *
from keras.utils import pad_sequences
from hate.cloud_storage.s3_operation import S3Operation
from sklearn.metrics import confusion_matrix
from hate.entity.config_entity import ModelEvaluationConfig
from hate.entity.artifact_entity import ModelEvaluationArtifacts, ModelTrainerArtifacts, DataTransformationArtifacts
import mlflow
from mlflow import log_metric, log_param

class ModelEvaluation:
    def __init__(self, model_evaluation_config: ModelEvaluationConfig, 
                 model_trainer_artifacts: ModelTrainerArtifacts, 
                 data_transformation_artifacts: DataTransformationArtifacts):
        
        try:
            self.model_evaluation_config = model_evaluation_config
            self.model_trainer_artifacts = model_trainer_artifacts
            self.data_transformation_artifacts = data_transformation_artifacts
            self.s3operation = S3Operation()

        except Exception as e:
            raise CustomException(e, sys) 
        

    def get_best_model_from_s3(self):
        """
        :return: Fetch best model from s3 storage and store inside best model directory path
        """
        try:
            logging.info("Entered the get_best_model_from_s3 method of ModelEvaluation class")

            os.makedirs(self.model_evaluation_config.BEST_MODEL_DIR_PATH, exist_ok=True)
            
            self.s3operation.sync_folder_from_s3(self.model_evaluation_config.BUCKET_NAME,
                                                 self.model_evaluation_config.MODEL_NAME,
                                                 self.model_evaluation_config.BEST_MODEL_DIR_PATH)
            
            best_model_path = os.path.join(self.model_evaluation_config.BEST_MODEL_DIR_PATH,
                                           self.model_evaluation_config.MODEL_NAME)

            logging.info("Exited the get_best_model_from_s3 method of ModelEvaluation class")
            return best_model_path

        except Exception as e:
            raise CustomException(e, sys)



    def evaluate(self):

        """
        :param model: Currently trained model or best model from s3 storage
        :param data_loader: Data loader for validation dataset
        :return: loss
        """

        try:
            logging.info("Entering into the evaluate function of Model Evaluation class")
            print(self.model_trainer_artifacts.x_test_path)

            x_test = pd.read_csv(self.model_trainer_artifacts.x_test_path, index_col=0)
            y_test = pd.read_csv(self.model_trainer_artifacts.y_test_path, index_col=0)

            with open('tokenizer.pickle', 'rb') as handle:
                tokenizer = pickle.load(handle)

            load_model = keras.models.load_model(self.model_trainer_artifacts.trained_model_path)

            x_test = x_test['tweet'].astype(str)

            x_test = x_test.squeeze()
            y_test = y_test.squeeze()

            test_sequences = tokenizer.texts_to_sequences(x_test)
            test_sequences_matrix = pad_sequences(test_sequences, maxlen=MAX_LEN)
            print(f"----------{test_sequences_matrix}------------------")

            print(f"-----------------{x_test.shape}--------------")
            print(f"-----------------{y_test.shape}--------------")

            accuracy = load_model.evaluate(test_sequences_matrix, y_test)
            logging.info(f"The test accuracy is {accuracy}")

            lstm_prediction = load_model.predict(test_sequences_matrix)
            res = []
            for prediction in lstm_prediction:
                if prediction[0] < 0.5:
                    res.append(0)
                else:
                    res.append(1)

            print(confusion_matrix(y_test, res))
            logging.info(f"The confusion_metrix is {confusion_matrix(y_test, res)} ")
            return accuracy


        except Exception as e:
            raise CustomException(e, sys)



    def initiate_model_evaluation(self) -> ModelEvaluationArtifacts:
        """
        Method Name: initiate_model_evaluation
        Description: This method is responsible for initiating the model evaluation

        Output: Returns model evaluation artifact
        On Failure: Raise Exception
        """

        try:
            logging.info(f"Entered the initiate_model_evaluation method of ModelEvaluation class")
            with mlflow.start_run():
                trained_model = keras.models.load_model(self.model_trainer_artifacts.trained_model_path)
                with open('tokenizer.pickle', 'rb') as handle:
                    load_tokenizer = pickle.load(handle)

                trained_model_accuracy = self.evaluate()

                logging.info("Fetch best model from s3 storage")
                best_model_path = self.get_best_model_from_s3()

                logging.info("Check if the best model present in the s3 storage or not?")
                if os.path.isfile(best_model_path) is False:
                    is_model_accepted = True
                    logging.info("The best model is false and  currently trained model accepted is true")

                else:
                    logging.info("The best model fetched from s3 storage")
                    best_model = keras.models.load_model(best_model_path)
                    best_model_accuracy = self.evaluate()

                    logging.info("Comparing loss between best_model_loss and trained_model_loss?")
                    if best_model_accuracy > trained_model_accuracy:
                        is_model_accepted = True
                        logging.info("Trained model not accepted")
                    else:
                        is_model_accepted = False
                        logging.info("Trained model accepted")  

                model_evaluation_artifacts = ModelEvaluationArtifacts(is_model_accepted=is_model_accepted)
                logging.info("Returning the ModelEvaluationArtifacts")
                return model_evaluation_artifacts


        except Exception as e:
            raise CustomException(e, sys)