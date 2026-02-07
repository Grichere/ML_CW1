import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

trn = pd.read_csv('CW1_train.csv')

submission = pd.read_csv('CW1_submission_K24069034.csv') 
yhat = submission['yhat']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

sns.histplot(trn['outcome'], kde=True, ax=ax1, color='blue', alpha=0.7)
ax1.set_title('Training Outcome Distribution')
ax1.set_xlabel('Outcome')
ax1.set_ylabel('Frequency')

sns.histplot(yhat, kde=True, ax=ax2, color='orange', alpha=0.7)
ax2.set_title('Model Predictions')
ax2.set_xlabel('Outcome')
ax2.set_ylabel('Frequency')

plt.tight_layout()
plt.show()

