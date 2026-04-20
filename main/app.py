import logging
from io import StringIO

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import pandas as pd
import numpy as np
from pipeline.prediction import predict_sentiment
from pipeline.training import train_and_evaluate

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

app = FastAPI(title="Shop ease sentiment API")


# pydantic schema for input dataset
class TextRequest(BaseModel):
    text: str


predictor = predict_sentiment()
logging.info("model successfully loaded")


@app.post("/predict_sentiment")
def predict_text(request: TextRequest):
    try:
        result = predictor.predict(request.text)
        print(result)

        top_label = max(result, key=lambda x: x["score"])
        return {
            "label": top_label["label"],
            "confidence": float(top_label["score"])
        }

    except Exception as e:
        logging.error(f"error occurred while predicting the sentiment {e}")
        return {"error": str(e)}


@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode("utf-8")))

        if "text" not in df.columns:
            return {"error": "CSV file must contain a 'text' column."}

        # predict each review
        result_list = []

        for idx, row in df.iterrows():
            try:
                review = str(row["text"])
                result = predictor.predict(review)

                if result is None or len(result) == 0:
                    raise ValueError("Empty result from the model")

                top_label = max(result, key=lambda x: x["score"])
                result_row = row.to_dict()
                result_row["sentiment_label"] = top_label["label"]
                result_row["confidence_label"] = float(top_label["score"])
                result_list.append(result_row)

            except Exception as e:
                logging.error(f"error during batch prediction at row {idx}: {e}")

        return result_list

    except Exception as e:
        logging.error(f"error occurred while processing batch file {e}")
        return {"error": str(e)}


@app.get("/train")
def train_model():
    try:
        train_and_evaluate()
        return {"message": "training completed successfully"}

    except Exception as e:
        logging.error(f"error occurred while training the model {e}")
        return {"error": str(e)}