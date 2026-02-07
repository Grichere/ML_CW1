import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import RidgeCV
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

trn = pd.read_csv('CW1_train.csv')
X = trn.drop(columns=['outcome'])
y = trn['outcome']

numeric_features = ['carat', 'depth', 'price', 'table'] + [f'a{i}' for i in range(1,11)] + [f'b{i}' for i in range(1,11)]
categorical_features = ['cut', 'color', 'clarity']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        
    ],
    remainder='drop'
)

pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', RidgeCV(alphas=np.logspace(-3, 3, 10)))
])

X_trn, X_val, y_trn, y_val = train_test_split(X, y, test_size=0.2, random_state=123)

pipeline.fit(X_trn, y_trn)
y_val_hat = pipeline.predict(X_val)

r2 = r2_score(y_val, y_val_hat)
print(f"Train/Val R²: {r2:.5f}")
print(f"Best alpha: {pipeline.named_steps['model'].alpha_:.5f}")
