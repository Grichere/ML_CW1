import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Load data
trn = pd.read_csv('CW1_train.csv')
tst = pd.read_csv('CW1_test.csv')

# Basic stats
print(trn.describe())
print(trn.info())  # Check dtypes, missing values

# Target distribution
plt.figure(figsize=(10, 4))
sns.histplot(trn['outcome'], kde=True)
plt.title('Outcome distribution')
plt.xlabel('Outcome')
plt.ylabel('Frequency')
plt.show()

# Numeric features
numeric_cols = trn.select_dtypes(include='number').columns.drop('outcome')
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i, col in enumerate(numeric_cols[:8]):  # First 8 numerics
    sns.histplot(trn[col], kde=True, ax=axes[i//4, i%4])
    axes[i//4, i%4].set_title(col, fontsize=12, fontweight='bold')
    axes[i//4, i%4].set_xlabel('Value')
    axes[i//4, i%4].set_ylabel('Frequency')
plt.tight_layout()
plt.show()

# Categorical features
categorical_cols = ['cut', 'color', 'clarity']
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, col in enumerate(categorical_cols):
    sns.countplot(data=trn, x=col, ax=axes[i])
    axes[i].set_title(f'{col} distribution')
    axes[i].set_xlabel(col.capitalize())
    axes[i].set_ylabel('Frequency')
plt.tight_layout()
plt.show()

# Correlation with outcome
import math

# Numeric variables: scatter + regression line, show Pearson r in title
numeric_cols = trn.select_dtypes(include='number').columns.drop('outcome')
n = len(numeric_cols)
cols = 3
rows = math.ceil(n / cols)
fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, max(4, rows * 3)))

# flatten axes into a list regardless of shape
axes_flat = []
for a in axes:
    try:
        axes_flat.extend(list(a))
    except TypeError:
        axes_flat.append(a)

for i, col in enumerate(numeric_cols):
    ax = axes_flat[i]
    sns.regplot(x=col, y='outcome', data=trn, ax=ax, scatter_kws={'s': 10, 'alpha': 0.6}, line_kws={'color': 'red'})
    r = trn[col].corr(trn['outcome'])
    ax.set_title(f'{col} vs outcome (r={r:.2f})')
    ax.set_xlabel(col)
    ax.set_ylabel('Outcome')

# remove any unused axes
for j in range(i + 1, len(axes_flat)):
    fig.delaxes(axes_flat[j])

plt.tight_layout()
plt.show()

# Categorical variables: boxplots of outcome by category
cat_cols = ['cut', 'color', 'clarity']
fig, axes = plt.subplots(1, len(cat_cols), figsize=(5 * len(cat_cols), 4))
if len(cat_cols) == 1:
    axes = [axes]
for i, col in enumerate(cat_cols):
    sns.boxplot(x=col, y='outcome', data=trn, ax=axes[i])
    axes[i].set_title(f'Outcome by {col}')
    axes[i].set_xlabel(col.capitalize())
    axes[i].set_ylabel('Outcome')
plt.tight_layout()
plt.show()
