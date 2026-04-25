import os

import mlflow
import mlflow.transformers
import dagshub
import logging
from utils.models_utils import get_best_f1
from config.constant import model_name,training_args,registered_model_name
from transformers import pipeline

from dotenv import load_dotenv
load_dotenv(override=True)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class ModelPusher:
    def __init__(self, experiment_name="sentiment_analysis"):
        try:
           # dagshub.init(
           #     repo_owner='inkumsah2012',
           #     repo_name='shopease-sentiment-project',
           #     mlflow=True
           # )
 
            dagshub_token = os.getenv("Shop_env_DAGSHUB_TOKEN")

            if not dagshub_token:
                raise EnvironmentError("Shop_env_DAGSHUB_TOKEN environment variable is not set")

            os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
            os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

            dagshub_url = "https://dagshub.com"
            repo_owner = "inkumsah2012"
            repo_name = "shopease-sentiment-project"

            # Set up MLflow tracking URI
            mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")
            mlflow.set_experiment(experiment_name)
            self.experiment_name = experiment_name
        except Exception as e:
            logging.error(f"error while initializing mlflow: {e}")

 
    def update_model_pusher(self, trainer, metrics):
        try:
            new_f1 = metrics["eval_f1"]
            old_f1 = get_best_f1(self.experiment_name)

            print(f"New f1 score: {new_f1}")
            print(f"Old f1 score: {old_f1}")

            if old_f1 is None or new_f1 > old_f1:
                with mlflow.start_run():
                    # log metrics
                    mlflow.log_metric("accuracy", metrics["eval_accuracy"])
                    mlflow.log_metric("f1", new_f1)

                    # log parameters
                    mlflow.log_param("model_name", model_name)
                    mlflow.log_param("epochs", training_args.num_train_epochs)
                    mlflow.log_param("train_batch_size", training_args.per_device_train_batch_size)
                    mlflow.log_param("eval_batch_size", training_args.per_device_eval_batch_size)

                    # create pipeline
                    sentiment_pipeline = pipeline(
                        task="text-classification",
                        model=trainer.model,
                        tokenizer=model_name,
                        return_all_scores=True
                    )

                    # log model
                    mlflow.transformers.log_model(
                        transformers_model=sentiment_pipeline,
                        artifact_path="model",
                        registered_model_name=registered_model_name
                    )

                logging.info("Model and metrics have been successfully pushed to MLflow.")
            else:
                logging.info("Model not pushed to MLflow because performance did not improve.")

        except Exception as e:
            logging.error(f"Failed to push model and result to MLflow: {e}")