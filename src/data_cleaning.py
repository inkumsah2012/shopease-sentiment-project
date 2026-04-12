# data_cleaning.py

import pandas as pd
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import logging
import re

from src.data_ingestion import data_ingestion
from config.constant import Cleaned_Data

logging.basicConfig(level=logging.INFO)

sentiment_data = data_ingestion()


class DataCleaning:
    def __init__(self):
        self._ensure_nltk()
        self.nlp = self._load_nlp()

    def _load_nlp(self) -> spacy.language.Language:
        for model in ("en_core_web_sm", "xx_ent_wiki_sm"):
            try:
                return spacy.load(model)
            except OSError:
                continue
        return spacy.blank("xx")

    def _ensure_nltk(self) -> None:
        try:
            _ = stopwords.words("english")
        except LookupError:
            nltk.download("stopwords")

        try:
            word_tokenize("test")
        except LookupError:
            nltk.download("punkt")

        try:
            nltk.data.find("tokenizers/punkt_tab/english/")
        except LookupError:
            nltk.download("punkt_tab")
        except Exception:
            pass

    def clean_text(self, text: str) -> str:
        text = str(text).lower()
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)
        text = re.sub(r"[^a-z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def lemmatize(self, text: str) -> str:
        doc = self.nlp(text)
        return " ".join(token.lemma_ if token.lemma_ else token.text for token in doc)

    def remove_stopwords(self, text: str) -> str:
        tokens = word_tokenize(text)
        sw = set(stopwords.words("english"))
        tokens = [t for t in tokens if t not in sw]
        return " ".join(tokens)


def clean_data(sentiment_data: pd.DataFrame):
    try:
        cleaner = DataCleaning()
        sentiment_data["clean_text"] = sentiment_data["review"].apply(cleaner.clean_text)
        sentiment_data["lemma_text"] = sentiment_data["clean_text"].apply(cleaner.lemmatize)
        sentiment_data["final_text"] = sentiment_data["lemma_text"].apply(cleaner.remove_stopwords)

        sentiment_data["label"] = sentiment_data["rating"].apply(
            lambda r: 0 if r in (1, 2) else (1 if r == 3 else 2)
        )

        sentiment_data = sentiment_data[["review", "final_text", "label"]]
        logging.info("Data successfully cleaned...")
        print(sentiment_data.head(5))
        sentiment_data.to_csv(Cleaned_Data, index=False)
        return sentiment_data

    except Exception as e:
        logging.error(f"Error occurred while cleaning the data: {e}")


clean_data(sentiment_data)