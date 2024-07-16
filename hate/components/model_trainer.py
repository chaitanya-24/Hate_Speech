import os 
import sys
import pickle
import pandas as pd
from hate.logger import logging
from hate.exception import CustomException
from hate.constants import *
from hate.entity.config_entity import ModelTrainerConfig
from hate.entity.artifact_entity import ModelTrainerArtifacts, DataTransformationArtifacts
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
from hate.ml.model import ModelArchitecture
from tensorflow.keras.preprocessing.sequence import pad_sequences


class ModelTrainer:
    def __init__(self, data_transformation_artifacts: DataTransformationArtifacts, model_trainer_config: ModelTrainerConfig):
        self.data_transformation_artifacts = data_transformation_artifacts
        self.model_trainer_config = model_trainer_config


    def splitting_data(self, csv_path):
        try:
            logging.info("Entered the splitting_data function")
            logging.info("Reading the data")
            df = pd.read_csv(csv_path, index_col=False)
            logging.info("Splitting the data into x and y")
            x = df[TWEET]
            y = df[LABEL]

            logging.info("Applying train test split on the data")
            x_train, x_test, y_train, y_test = train_test_split(x, y,random_state=RANDOM_STATE)
            print(len(x_train), len(y_train))
            print(len(x_test), len(y_test))
            print(type(x_train), type(y_train))
            logging.info("Exited the splitting_data function")
            return x_train, x_test, y_train, y_test

        except Exception as e:
            raise CustomException(e, sys) 


    def tokenizing(self, x_train):
        try:
            logging.info("Entered the tokenizing function")
            tokenizer = Tokenizer(num_words=self.model_trainer_config.MAX_WORDS)
            tokenizer.fit_on_texts(x_train)
            sequences = tokenizer.texts_to_sequences(x_train)
            logging.info(f"Converting text to sequences: {sequences}")
            sequences_matrix = pad_sequences(sequences, maxlen=self.model_trainer_config.MAX_LEN)
            logging.info(f"Converting sequences to matrix: {sequences_matrix}")
            return sequences_matrix, tokenizer
        
        except Exception as e:
            raise CustomException(e, sys)
        

    def initiate_model_trainer(self) -> ModelTrainerArtifacts:
        try:
            logging.info("Entered the initiate_model_trainer function")
            x_train, x_test, y_train, y_test = self.splitting_data(csv_path=self.data_transformation_artifacts.transformed_data_path)
            model_architecture = ModelArchitecture()
            model = model_architecture.get_model()

            logging.info(f"Xtrain size is: {x_train.shape}")
            logging.info(f"Xtest size is: {x_test.shape}")

            sequences_matrix, tokenizer = self.tokenizing(x_train)

            logging.info("Entered into model training")
            model.fit(sequences_matrix, y_train, epochs=self.model_trainer_config.EPOCH, validation_split=self.model_trainer_config.VALIDATION_SPLIT)
            logging.info("Exited the model training")

            with open('tokenizer.pickle', 'wb') as handle:
                pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.makedirs(self.model_trainer_config.MODEL_TRAINER_ARTIFACTS_DIR, exist_ok=True)


            logging.info("Saving the model")
            model.save(self.model_trainer_config.MODEL_FILE_NAME)
            x_test.to_csv(self.model_trainer_config.X_TEST_DATA_PATH)
            y_test.to_csv(self.model_trainer_config.Y_TEST_DATA_PATH)

            x_train.to_csv(self.model_trainer_config.X_TRAIN_DATA_PATH)


            model_trainer_artifacts = ModelTrainerArtifacts(
                trained_model_path=self.model_trainer_config.MODEL_FILE_NAME,
                x_test_path=self.model_trainer_config.X_TEST_DATA_PATH,
                y_test_path=self.model_trainer_config.Y_TEST_DATA_PATH,
            )
            logging.info("Returning the ModelTrainerArtifacts")
            return model_trainer_artifacts


        except Exception as e:
            raise CustomException(e, sys)