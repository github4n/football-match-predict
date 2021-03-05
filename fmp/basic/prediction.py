# coding:utf-8

import pandas as pd
import xgboost as xgb
from time import time
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import scale
import joblib
from sklearn import preprocessing
import numpy as np
from pathlib import Path

filepath = 'xgboost_model.model'


def binary(string):
    """把比赛结果转化成 0 和 1"""
    if string == 'H':
        return 1
    else:
        return 0


def handle(data):
    df = pd.DataFrame(data)
    features = df[['home_team', 'away_team']]
    target = df.result.apply(binary)
    # print(features)
    # print(target)
    # 将类型进行转化 xgb只接受有限的类型
    lbl = preprocessing.LabelEncoder()
    features['home_team'] = lbl.fit_transform(features['home_team'].astype(str))
    features['away_team'] = lbl.fit_transform(features['away_team'].astype(str))
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=2,
                                                        stratify=target)
    return x_train, y_train


def train(data):
    """训练"""
    # 处理数据 获取特征值和标签
    x_train, y_train = handle(data)

    # 检查模型文件是否已存在
    model_file = Path(filepath)
    if model_file.exists():
        clf = joblib.load(filepath)
    else:
        clf = xgb.XGBClassifier(seed=42, use_label_encoder=False)

    # 训练
    clf.fit(x_train, y_train)
    # 保存模型
    joblib.dump(clf, filepath)


def predict(data):
    """预测"""

    # 读取模型
    model = joblib.load(filepath)
    # 处理数据 获取特征值和标签
    x_train, y_train = handle(data)
    # 预测
    predict_result = model.predict(x_train)
    # 准确率
    cp = sum(y_train == predict_result) / float(len(y_train))

    return predict_result, cp


def statistical(arr):
    """统计"""
    # 求均值
    arr_mean = np.mean(arr)
    # 求方差
    arr_var = np.var(arr)
    # 求标准差
    arr_std = np.std(arr, ddof=1)
    print("平均值为：%f" % arr_mean)
    print("方差为：%f" % arr_var)
    print("标准差为:%f" % arr_std)
    return
