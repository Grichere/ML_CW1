import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

trn = pd.read_csv('CW1_train.csv')
X = trn.drop(columns=['outcome'])
y = trn['outcome']

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


X_trn, X_val, y_trn, y_val = train_test_split(X, y, test_size=0.2, random_state=123)
pipeline.fit(X_trn, y_trn)
score = pipeline.score(X_val, y_val)

print(f"Pipeline R²: {score:.5f}")