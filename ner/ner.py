import os
import pandas as pd
import sys
import spacy
import nltk
from ast import literal_eval
from nltk import sent_tokenize
nltk.download('punkt')
nltk.download('punkt_tab')
from utils.utils import load_dataset

class NER:
    def __init__(self):
        self.model =model = spacy.load("en_core_web_trf")

    def ner_df(self,script):
      sentences = sent_tokenize(script)
      ner_output = []
      for sentence in sentences:
        ner =set()
        doc = self.model(sentence)
        for entity in doc.ents:
          if entity.label_ == 'PERSON':
            name = entity.text.split(" ")[0].strip()
            ner.add(name)
        ner_output.append(ner)
      return ner_output

    def get_ner(self,dataset_path,save_path=None):
        if save_path is not None and os.path.exists(save_path):
            df = pd.read_csv(save_path)
            df["ners"] = df["ners"].apply(lambda x: literal_eval(x) if isinstance(x,str) else x)
            return df

        df = load_dataset(dataset_path)
        df["ners"] = df["script"].apply(self.get_ner)

        if save_path is not None:
            df.to_csv(save_path,index=False)
            return df