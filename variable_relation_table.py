import pandas as pd
import numpy as np

trn = pd.read_csv("CW1_train.csv")

# Numeric-only features (drop target)
num_cols = trn.select_dtypes(include='number').columns.drop('outcome')

# Correlation matrix
corr = trn[num_cols].corr()

# Get pairs with |corr| above a threshold, e.g. 0.9
def high_corr_pairs(corr, threshold=0.9):
    pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            if abs(corr.iloc[i, j]) >= threshold:
                pairs.append((cols[i], cols[j], corr.iloc[i, j]))
    return pd.DataFrame(pairs, columns=['feature_1', 'feature_2', 'correlation'])

high_corr = high_corr_pairs(corr, threshold=0.9)
print(high_corr.sort_values(by='correlation', key=np.abs, ascending=False))


from statsmodels.stats.outliers_influence import variance_inflation_factor

print("\nTop features by VIF:")

X = trn[num_cols].copy().drop(columns=['price'])  # example: maybe drop obvious proxies first
X = X.dropna()

vif_data = []
for i, col in enumerate(X.columns):
    vif = variance_inflation_factor(X.values, i)
    vif_data.append((col, vif))

vif_table = pd.DataFrame(vif_data, columns=['feature', 'VIF']).sort_values('VIF', ascending=False)
print(vif_table.head(20))
