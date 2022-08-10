import pandas as pd
import numpy as np
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, pipeline
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification
import tensorflow as tf
import colorama
from colorama import Fore
#import pycountry
#from geopy.geocoders import Nominatim

tokenizer1 = AutoTokenizer.from_pretrained("distilbert-base-uncased")
disaster_model = TFAutoModelForSequenceClassification.from_pretrained("sacculifer/dimbat_disaster_distilbert")
dtype_model = TFAutoModelForSequenceClassification.from_pretrained("sacculifer/dimbat_disaster_type_distilbert")

tokenizer2 = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
demotion_model = AutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base")

tokenizer3 = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
dgeo_model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

classifier1 = pipeline("text-classification", tokenizer=tokenizer1, model=disaster_model)
classifier2 = pipeline("text-classification", tokenizer=tokenizer1, model=dtype_model)
classifier3 = pipeline("text-classification", tokenizer=tokenizer2, model=demotion_model)
classifier4 = pipeline("ner", model=dgeo_model, tokenizer=tokenizer3, aggregation_strategy='average')

number_tweets = 0

while number_tweets < 100:
      print(Fore.YELLOW + "Input text:")
      text = str(input())

      output = classifier1(text)
      if output[0]['label'] == 'LABEL_1':
          print("Disaster detected")
          output_dtype = classifier2(text)
          if output_dtype[0]['label'] == 'LABEL_1':
             print("Disaster type: disease ", output_dtype[0]['score'])
          elif output_dtype[0]['label'] == 'LABEL_2':
             print("Disaster type: earthquake ", output_dtype[0]['score'])
          elif output_dtype[0]['label'] == 'LABEL_3':
             print("Disaster type: flood ", output_dtype[0]['score'])
          elif output_dtype[0]['label'] == 'LABEL_4':
             print("Disaster type: hurricane or tornado ", output_dtype[0]['score'])
          elif output_dtype[0]['label'] == 'LABEL_5':
             print("Disaster type: wildfire ", output_dtype[0]['score'])
          elif output_dtype[0]['label'] == 'LABEL_6':
             print("Disaster type: industrial accident ", output_dtype[0]['score'])
          elif output_dtype[0]['label'] == 'LABEL_7':
             print("Disaster type: societal crime ", output_dtype[0]['score'])
          elif output_dtype[0]['label'] == 'LABEL_8':
             print("Disaster type: transportation accident ", output_dtype[0]['score'])
          elif output_dtype[0]['label'] == 'LABEL_9':
             print("Disaster type: meteor crash ", output_dtype[0]['score'])
          elif output_dtype[0]['label'] == 'LABEL_0':
             print("Disaster type: haze ", output_dtype[0]['score'])
          output_geo = classifier4(text)
          output_geo = pd.DataFrame(output_geo)
          ner_loc = output_geo[output_geo["entity_group"] == "LOC"]
          if not ner_loc.empty :
             i = 0
             for x in ner_loc["score"]:
                 if x > 0.5:
                     print("Location: ", ner_loc["word"].iloc[i])
                 i += 1
          output_emo = classifier3(text)
          print("Sentiment: ", output_emo[0]['label'], " ", output_emo[0]['score'])
      else:
          print(" No disaster")