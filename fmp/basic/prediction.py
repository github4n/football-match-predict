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


def binary(string):
    """主场胜利 是与否"""
    if string == 'H':
        return 1
    else:
        return 0


def preprocess_features(x):
    """把离散的类型特征转为亚编码特征"""
    output = pd.DataFrame(index=x.index)
    for col, col_data in x.iteritems():
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix=col)
        output = output.join(col_data)
    return output


def train(features, target):
    """训练模型"""

    clf = xgb.XGBClassifier(seed=42)
    # 训练
    clf.fit(features, target)
    # 保存模型
    joblib.dump(clf, 'xgboost_model.model')


def predict(features):
    """使用模型预测"""

    # 读取模型
    model = joblib.load('xgboost_model.model')
    # 预测
    predict_result = model.predict(features)

    print(predict_result)


# 数据处理逻辑
def handle(data):
    """处理逻辑"""
    df_dict = {}
    for k, v in data.items():
        df_dict[k] = pd.DataFrame(v)

    # data_frame = df_dict['2005-2006']
    data_frame = df_dict['2006-2007']
    data_frame['result'] = data_frame.result.apply(binary)

    x_all = data_frame.drop(['match_date', 'match_season', 'result'], 1)
    y_all = data_frame['result']
    x_all = preprocess_features(x_all)
    x_all = x_all[x_all.columns]

    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.3, random_state=2, stratify=y_all)

    # 训练
    # train(x_train, y_train)
    # 预测
    predict(x_train)

    return
