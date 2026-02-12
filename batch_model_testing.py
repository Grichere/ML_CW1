import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

def run_optimization():
    df = pd.read_csv('CW1_train.csv')

    X = df.drop(columns=['outcome'])
    y = df['outcome']

    X_eng = X.copy()
    X_eng['depth_sq'] = X_eng['depth'] ** 2

    base_numeric = ['carat', 'depth', 'price', 'table'] + \
                   [f'a{i}' for i in range(1,11)] + \
                   [f'b{i}' for i in range(1,11)]
    categorical = ['cut', 'color', 'clarity']

    feature_sets = {
        "Base": base_numeric,
        "Base - table": [f for f in base_numeric if f != 'table'],
        "Base + XYZ": base_numeric + ['x', 'y', 'z'],
        "Base + Y": base_numeric + ['y'],
        "Base + XYZ - carat": [f for f in base_numeric if f != 'carat'] + ['x', 'y', 'z'],
        "Base + Depth^2": base_numeric + ['depth_sq'],
    }

    param_grid = [
        {'learning_rate': 0.1, 'max_iter': 100, 'max_depth': None, 'l2': 0},
        
        {'learning_rate': 0.05, 'max_iter': 500, 'max_depth': 10, 'l2': 0},
        
        {'learning_rate': 0.05, 'max_iter': 500, 'max_depth': 10, 'l2': 1.0},
        
        {'learning_rate': 0.01, 'max_iter': 2000, 'max_depth': 15, 'l2': 0}
    ]

    X_trn, X_val, y_trn, y_val = train_test_split(X_eng, y, test_size=0.2, random_state=123)

    results = []
    
    print(f"\n{'Feature Set':<20} | {'LR':<5} | {'Trees':<5} | {'Depth':<5} | {'L2':<3} | {'R2 Score':<8}")
    print("-" * 75)

    for feat_name, num_feats in feature_sets.items():
        preprocessor = ColumnTransformer([
            ('num', 'passthrough', num_feats),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical)
        ], remainder='drop') 

        for params in param_grid:
            pipe = Pipeline([
                ('prep', preprocessor),
                ('model', HistGradientBoostingRegressor(
                    learning_rate=params['learning_rate'],
                    max_iter=params['max_iter'],
                    max_depth=params['max_depth'],
                    l2_regularization=params['l2'],
                    random_state=42
                ))
            ])
            
            pipe.fit(X_trn, y_trn)
            score = r2_score(y_val, pipe.predict(X_val))
            
            print(f"{feat_name:<20} | {params['learning_rate']:<5} | {params['max_iter']:<5} | {str(params['max_depth']):<5} | {params['l2']:<3} | {score:.5f}")
            
            results.append({
                'features': feat_name,
                'params': params,
                'score': score
            })

    best_result = max(results, key=lambda x: x['score'])
    print(f"Best Feature Set: {best_result['features']}")
    print(f"R² Score: {best_result['score']:.5f}")
    print(f"Parameters: {best_result['params']}")

if __name__ == "__main__":
    run_optimization()
