import mlflow
import dagshub
import logging
from mlflow.tracking import MlflowClient
#from config.constant import model_name as default_model_name
from config.constant import model_name, num_of_labels,registered_model_name
import dagshub
dagshub.init(repo_owner='inkumsah2012', repo_name='shopease-sentiment-project', mlflow=True)

logging.basicConfig(level=logging.INFO)

def get_best_model(experiment_name = "sentiment_analysis"):
    try:
        client = MlflowClient()

        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            return None

        runs = client.search_runs([experiment.experiment_id])
        if not runs:
            return None

        best_model = sorted(
            runs,
            key = lambda x: x.data.metrics.get("f1", 0),
            reverse = True
        )[0]
        return best_model
    except Exception as e:
        logging.error(f"error occured while loading the models {e}")

def get_best_f1(experiment_name = "sentiment_analysis"):
    best_runs = get_best_model(experiment_name)
    if best_runs is None:
        return None
    return best_runs.data.metrics.get("f1", 0)

def load_registered_model(model_name = registered_model_name):
    model_uri = f"models:/{model_name}/latest"
    sentiment_pipeline = mlflow.transformers.load_model(model_uri)
    return sentiment_pipeline



