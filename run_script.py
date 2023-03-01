import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
import os, pickle
from dirty_cat import (SimilarityEncoder, TargetEncoder,
                       MinHashEncoder, GapEncoder)
from preprocess_functions import preprocess_df
##### FIX MICHAELA'S CARD NUMBER
cur_path = os.path.abspath('.')
folder = input("What Folder?")
with open(os.path.join(cur_path,'model','gbct_model_20230228.pkl'), 'rb') as file : 
    model = pickle.load(file)
objs = os.listdir(os.path.join(cur_path,folder))
file = [x for x in objs if x.endswith('_transaction_download.csv')][0]


df = preprocess_df(pd.read_csv(os.path.join(cur_path,folder,file)))

predictions = model.predict(df)
df['prediction'] = predictions
out_dict = df.groupby('prediction')['amount'].sum().to_dict()
personal = out_dict[0]
joint = out_dict[1]
print("="*80)
print("="*80)
print(' '*80)
print(f"Estimated personal expenses : {personal}")
print(f"Estimated joint expenses : {joint}")
print(' '*80)
print("="*80)
print("="*80)
df.to_csv(os.path.join(cur_path,folder,'predictions_out.csv'))