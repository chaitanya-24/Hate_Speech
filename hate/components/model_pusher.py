import sys 
from hate.logger import logging
from hate.exception import CustomException
from hate.entity.config_entity import ModelPusherConfig
from hate.entity.artifact_entity import ModelPusherArtifacts
from hate.cloud_storage.s3_operation import S3Operation


class ModelPusher:
    def __init__(self, model_pusher_config: ModelPusherConfig):
        """
        :param model_pusher_config: Configuration for model pusher
        """

        self.model_pusher_config = model_pusher_config
        self.s3operation = S3Operation()

    
    def initiate_model_pusher(self) -> ModelPusherArtifacts:
        
        """
            Method Name : initiate_model_pusher
            Description : This method initiates model pusher

            Output      : Model Pusher Artifacts
        
        """

        logging.info("Entered initiate_model_pusher method of ModelPusher class")
        try:
            self.s3operation.sync_folder_to_s3(
                self.model_pusher_config.TRAINED_MODEL_PATH,
                self.model_pusher_config.BUCKET_NAME,
                self.model_pusher_config.MODEL_NAME)
            
            logging.info("Uploaded best model to s3 storage")

            model_pusher_artifact = ModelPusherArtifacts(
                bucket_name=self.model_pusher_config.BUCKET_NAME
            )
            logging.info("Exited the initiate_model_pusher method of ModelTrainer class")
            return model_pusher_artifact

        except Exception as e:
            raise CustomException(e, sys)