import torch
import pandas as pd
<<<<<<< HEAD
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import transformers
from transformers import Trainer, TrainingArguments
import os
from config.constant import Train_Data, Test_Data, model_name, training_args, num_of_labels
from src.data_cleaning import clean_data
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

class Training:
    def __init__(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name=model_name,
            num_labels=num_of_labels
        )

    def compute_metrics(self, p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {"accuracy": acc, "f1": f1}
=======
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
import transformers
import os
import logging
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from config.constant import Train_Data, Test_Data, model_name, training_args, num_of_labels
from src.data_cleaning import clean_data
from src.data_preprocessing import Prepare_sentiment_data


class Training:
    def __init__(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name,num_labels= num_of_labels )

    def compute_metrics(self, p):
        preds = np.argmax(p.predictions, axis=1)
        labels = p.label_ids
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="weighted")
        return {"accuracy": acc, "f1": f1}
>>>>>>> f3d98b0 (Add app, training pipeline, Docker files, and project structure)

    def model_training(self, train_dataset, test_dataset):
        try:
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                compute_metrics=self.compute_metrics
            )
            trainer.train()
<<<<<<< HEAD
            logging.info(f"Model has been successfully trained..")
            return trainer
        except Exception as e:
            logging.error(f"error occurred while training the model {e}")
=======
            logging.info("Model has been successfully trained.")
            return trainer
        except Exception as e:
            logging.error(f"Error occurred while training the model: {e}")
>>>>>>> f3d98b0 (Add app, training pipeline, Docker files, and project structure)

    def model_evaluation(self, trainer):
        results = trainer.evaluate()
        return results

<<<<<<< HEAD
#def train_and_evaluate():
#    try:
#        train_dataset = torch.load(Train_Data)
#        test_dataset = torch.load(Test_Data)
#        train = Training()
#        trainer = train.model_training(train_dataset, test_dataset)
#        results = train.model_evaluation(trainer)
#        return results, trainer
#    except Exception as e:
#        logging.error(f"error occurred while training and evaluating the model {e}")

#train_and_evaluate()
=======

>>>>>>> f3d98b0 (Add app, training pipeline, Docker files, and project structure)
