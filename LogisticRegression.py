from pandas import read_csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score,f1_score,accuracy_score
import lightgbm as lgb
from lightgbm.callback import early_stopping, log_evaluation, record_evaluation

train_dataset = read_csv('/data/train_data.csv')
train_dataset.text = train_dataset.text.str.replace('\n', ' ')


train_dataset.text = train_dataset.text.str.replace('\n', ' ')
class_0 = train_dataset[train_dataset['label'] == 0]
class_1 = train_dataset[train_dataset['label'] == 1]

# 确定两个类别的样本数量
num_class_0 = len(class_0)
num_class_1 = len(class_1)

# 计算较小的样本数量
min_num = min(num_class_0, num_class_1)

# 对样本较多的类别进行随机抽样
if num_class_0 > min_num:
    class_0 = class_0.sample(n=min_num, random_state=42)  # random_state 确保结果的可复现性
elif num_class_1 > min_num:
    class_1 = class_1.sample(n=min_num, random_state=42)

# 合并数据集
balanced_train_dataset = pd.concat([class_0, class_1])

# 如果需要，可以打乱数据集的顺序
balanced_train_dataset = balanced_train_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

pipe = make_pipeline(TfidfVectorizer(
    min_df=5,
    max_df=0.8,
    # max_features=10000,
    ngram_range=(3,5),
    ),
    LogisticRegression(max_iter=1200)
    # MultinomialNB()
)
param_grid = {
#     'multinomialnb__alpha': [0.1, 1, 10],
    'logisticregression__C': [100],
    # 'tfidfvectorizer__ngram_range': [(3, 5), (5, 5)],
    'tfidfvectorizer__norm': ['l2']
}
grid = GridSearchCV(pipe, param_grid, cv=5,verbose=3)
grid.fit(balanced_train_dataset['text'], balanced_train_dataset['label'])

print(f'best cross-val-score: {grid.best_score_}')
print(f'best params:\n{grid.best_params_}')
best_model = grid.best_estimator_
test = read_csv('../input/567testdata/Mistral7B_CME_v7_15_percent_corruption.csv', sep=',')
test['generated_new'] = grid.best_estimator_.predict(test['text'])
# test[["id", "generated"]].to_csv("submission.csv", index=False)
test.to_csv('submission.csv', index=False)
accuracy = accuracy_score(test['generated'], test['generated_new'])
print(f'Best accuracy score:{accuracy}')

# Print out the best cross-validation score, best parameters, F1 score, and AUC
print(f'Best cross-validation score: {grid.best_score_}')
print(f'Best parameters:\n{grid.best_params_}')
print(f'F1 score: {f1}')
print(f'AUC value: {auc}')