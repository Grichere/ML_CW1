from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

trn = pd.read_csv('CW1_train.csv').dropna(subset=['outcome'])
tst = pd.read_csv('CW1_test.csv')

numeric_features = ['carat', 'depth', 'price', 'table'] + \
                   [f'a{i}' for i in range(1,11)] + \
                   [f'b{i}' for i in range(1,11)]
categorical_features = ['cut', 'color', 'clarity']

pipeline = Pipeline([
    ('prep', ColumnTransformer([
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ])),
    ('model', HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, random_state=42))
])

pipeline.fit(trn.drop(columns=['outcome']), trn['outcome'])
yhat = pipeline.predict(tst)
pd.DataFrame({'yhat': yhat}).to_csv('CW1_submission_K24069034.csv', index=False)
print("Submission ready")
