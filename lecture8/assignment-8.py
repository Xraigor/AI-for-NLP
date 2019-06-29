import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import pickle
import jieba


def cut(string):
    return ' '.join(jieba.cut(string))

## 读取数据集
content = pd.read_csv("E:/Pycharm_Dic/assignment-8/sqlResult_1558435.csv", encoding='gb18030')
content = content.fillna('')
content['xinhua'] = np.where(content['source'].str.contains('新华'), 1, 0)
content['title'] = content['title'].fillna('')
content['title'] = content['title'].apply(cut)

content = content.sample(n=5000)

xinhuashe_news = content[content['source'].str.contains('新华')]
true_rate = len(xinhuashe_news)/len(content)

X_title = content['title'].values
y = content['xinhua'].values

## 词向量

vectoriezr = TfidfVectorizer(max_features=30000)
X_title = vectoriezr.fit_transform(X_title)

## 打乱原数据,分割训练集和数据集
indices = np.arange(len(y))
np.random.shuffle(indices)
split_test = 0.25
train_indices = indices[int(len(indices)*split_test):]
test_indices = indices[:int(len(indices)*split_test)]

X_t_train,X_t_test,y_train,y_test = (X_title[train_indices],X_title[test_indices],y[train_indices],y[test_indices])

## 高斯贝叶斯分类器
clf = GaussianNB()
clf.fit(X_t_train.toarray(), y_train)



## 载入评价函数
def get_performance(clf, x_, y_):
    y_hat = clf.predict(x_.toarray())
    print('precision is: {}'.format(precision_score(y_, y_hat)))
    print('recall is: {}'.format(recall_score(y_, y_hat)))
    print('roc_auc is:{}'.format(roc_auc_score(y_, y_hat)))
    print('confusion matrix: \n{}'.format(confusion_matrix(y_, y_hat, labels=[0,1])))


get_performance(clf, X_t_train,y_train)

## 决策树分类器
from sklearn.tree import DecisionTreeClassifier
clf_tree = DecisionTreeClassifier(class_weight={0:5,1:4},criterion='entropy',max_features=5000)
clf_tree.fit(X_t_train,y_train)

get_performance(clf_tree, X_t_train, y_train)


## 测试集预测
y_hat_test = clf.predict(X_t_test.toarray()[:1000])
print(y_hat_test)

## 错误预测的数据
candidate_indices = []
for index, (y,yhat) in enumerate(zip(y_test,y_hat_test)):
    if y == 0 and yhat == 1:
        candidate_indices.append(test_indices[index])

candidate_indices

## 载入模型文件
with open('clf', 'wb') as picklefile:
    pickle.dump(clf,picklefile)

with open('clf', 'rb') as training_model:
    model = pickle.load(training_model)

y_hat_test = model.predict(X_t_test.toarray()[:1000])
y_hat_test