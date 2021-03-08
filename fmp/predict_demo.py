# 导入必须的包
import pandas as pd  # 数据分析包
import numpy as np
import os

"""1. 读取数据"""

# 获取地址中的所有文件
local_path = './/dataset//'  # 存放数据的路径
res_name = []  # 存放数据名的列表
filecsv_list = []  # 获取数据名后存放的列表


def file_name(file_path):
    # root:当前目录路径   dirs：当前目录下所有子目录   files：当前路径下所有非目录文件
    for root, dirs, files in os.walk(file_path):
        files.sort()  # 排序，让列表里面的元素有顺序
        for i, file in enumerate(files):
            if os.path.splitext(file)[1] == '.csv':
                filecsv_list.append(file)
                res_name.append('raw_data_' + str(i + 1))


# 读取 csv 数据
file_name(local_path)

# 时间列表
time_list = [filecsv_list[i][0:4] for i in range(len(filecsv_list))]

# 用 Pandas.read_csv() 接口读取数据
for i in range(len(res_name)):
    res_name[i] = pd.read_csv(local_path + filecsv_list[i], error_bad_lines=False, warn_bad_lines=False).dropna(
        axis=0, how='all')

# 删除行数不是 380 的文件名
for i in range(len(res_name), 0, -1):
    # 采用从大到小的遍历方式，然后进行删除不满足条件的。
    if res_name[i - 1].shape[0] != 380:
        key = 'res_name[' + str(i) + ']'
        res_name.pop(i - 1)
        time_list.pop(i - 1)
        continue

# 挑选信息列
# HomeTeam: 主场球队名
# AwayTeam: 客场球队名
# FTHG: 主场球队全场进球数
# FTAG: 客场球队全场进球数
# FTR: 比赛结果 ( H= 主场赢, D= 平局, A= 客场赢)

columns_req = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
playing_statistics = []  # 创造处理后数据名存放处
playing_data = {}  # 键值对存储数据
for i in range(len(res_name)):
    playing_statistics.append('playing_statistics_' + str(i + 1))
    playing_statistics[i] = res_name[i][columns_req]

print('原始数据:')
print(playing_statistics[2].tail())  # 查看末尾五条数据

"""2. 根据已有信息构造特征"""


# 计算每个队周累计净胜球数量
def get_goals_diff(playing_stat):
    # 创建一个字典，每个 team 的 name 作为 key
    teams = {}
    for i in playing_stat.groupby('HomeTeam').mean().T.columns:
        teams[i] = []
    # 对于每一场比赛
    for i in range(len(playing_stat)):
        # 全场比赛，主场队伍的进球数
        HTGS = playing_stat.iloc[i]['FTHG']
        # 全场比赛，客场队伍的进球数
        ATGS = playing_stat.iloc[i]['FTAG']

        # 把主场队伍的净胜球数添加到 team 这个 字典中对应的主场队伍下
        teams[playing_stat.iloc[i].HomeTeam].append(HTGS - ATGS)
        # 把客场队伍的净胜球数添加到 team 这个 字典中对应的客场队伍下
        teams[playing_stat.iloc[i].AwayTeam].append(ATGS - HTGS)
    # 创建一个 GoalsDifference 的 dataframe
    # 行是 team 列是 matchweek,
    # 39解释：19个球队，每个球队分主场客场2次，共38个赛次，但是range取不到最后一个值，故38+1=39
    GoalsDifference = pd.DataFrame(data=teams, index=[i for i in range(1, 39)]).T

    GoalsDifference[0] = 0
    # 累加每个队的周比赛的净胜球数
    for i in range(2, 39):
        GoalsDifference[i] = GoalsDifference[i] + GoalsDifference[i - 1]
    return GoalsDifference


def get_gss(playing_stat):
    # 得到净胜球数统计
    GD = get_goals_diff(playing_stat)
    j = 0
    #  主客场的净胜球数
    HTGD = []
    ATGD = []
    # 全年一共380场比赛
    for i in range(380):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam
        HTGD.append(GD.loc[ht][j])
        ATGD.append(GD.loc[at][j])
        if ((i + 1) % 10) == 0:
            j = j + 1
    # 把每个队的 HTGD ATGD 信息补充到 dataframe 中
    playing_stat.loc[:, 'HTGD'] = HTGD
    playing_stat.loc[:, 'ATGD'] = ATGD
    return playing_stat


playing_statistics[2] = get_gss(playing_statistics[2])

print("构建特征'净胜球数量'后的数据")
print(playing_statistics[2].tail())


#  统计主客场队伍到当前比赛周的累计得分
# 把比赛结果转换为得分，赢得三分，平局得一分，输不得分
def get_points(result):
    if result == 'W':
        return 3
    elif result == 'D':
        return 1
    else:
        return 0


def get_cuml_points(matchres):
    matchres_points = matchres.applymap(get_points)
    for i in range(2, 39):
        matchres_points[i] = matchres_points[i] + matchres_points[i - 1]
    matchres_points.insert(column=0, loc=0, value=[0 * i for i in range(20)])
    return matchres_points


def get_matchres(playing_stat):
    # 创建一个字典，每个 team 的 name 作为 key
    teams = {}
    for i in playing_stat.groupby('HomeTeam').mean().T.columns:
        teams[i] = []
    # 把比赛结果分别记录在主场队伍和客场队伍中
    # H：代表 主场 赢
    # A：代表 客场 赢
    # D：代表 平局
    for i in range(len(playing_stat)):
        if playing_stat.iloc[i].FTR == 'H':
            # 主场 赢，则主场记为赢，客场记为输
            teams[playing_stat.iloc[i].HomeTeam].append('W')
            teams[playing_stat.iloc[i].AwayTeam].append('L')
        elif playing_stat.iloc[i].FTR == 'A':
            # 客场 赢，则主场记为输，客场记为赢
            teams[playing_stat.iloc[i].AwayTeam].append('W')
            teams[playing_stat.iloc[i].HomeTeam].append('L')
        else:
            # 平局
            teams[playing_stat.iloc[i].AwayTeam].append('D')
            teams[playing_stat.iloc[i].HomeTeam].append('D')

    return pd.DataFrame(data=teams, index=[i for i in range(1, 39)]).T


def get_agg_points(playing_stat):
    matchres = get_matchres(playing_stat)
    cum_pts = get_cuml_points(matchres)
    HTP = []
    ATP = []
    j = 0
    for i in range(380):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam
        HTP.append(cum_pts.loc[ht][j])
        ATP.append(cum_pts.loc[at][j])

        if ((i + 1) % 10) == 0:
            j = j + 1
    # 主场累计得分
    playing_stat.loc[:, 'HTP'] = HTP
    # 客场累计得分
    playing_stat.loc[:, 'ATP'] = ATP
    return playing_stat


playing_statistics[2] = get_agg_points(playing_statistics[2])

print("构建特征'累计得分'后的数据:")
print(playing_statistics[2].tail())


# 定义 target ，也就是否 主场赢
def only_hw(string):
    if string == 'H':
        return 'H'
    else:
        return 'NH'


playing_stat = playing_statistics[2]
playing_stat['FTR'] = playing_stat.FTR.apply(only_hw)
print("把比赛结果'H A D'用'H NH'来表示:")
print(playing_stat.tail())

"""3. 训练和预测"""


def change_type(X_all):
    """转换特征数据类型"""
    from sklearn.preprocessing import scale

    cols = [['HTGD', 'ATGD', 'HTP', 'ATP']]
    for col in cols:
        X_all[col] = scale(X_all[col])

    return X_all


def one_hot_encode(X_all):
    """把离散特征转为独热编码特征"""
    df = pd.DataFrame(index=X_all.index)
    for col, col_data in X_all.iteritems():
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix=col)
        df = df.join(col_data)
    return df


def train_predict(playing_stat):
    """训练和预测"""
    from sklearn.model_selection import train_test_split
    import xgboost as xgb
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import make_scorer
    from sklearn.metrics import f1_score

    # 把数据分为特征值和标签值
    X_all = playing_stat.drop(['FTR'], 1)
    print('最终的预测数据:')
    print(X_all.tail())
    X_all = change_type(X_all)
    X_all = one_hot_encode(X_all)

    y_all = playing_stat['FTR']
    y_all = y_all.map({'NH': 0, 'H': 1})  # 把标签映射为0和1

    # 将数据集随机分成为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.3, random_state=2, stratify=y_all)

    # 初始化模型
    clf = xgb.XGBClassifier(seed=42, eval_metric='logloss')
    # 调整模型
    f1_scorer = make_scorer(f1_score, pos_label=1)
    parameters = {'n_estimators': [90, 100, 110], 'max_depth': [5, 6, 7]}
    grid_obj = GridSearchCV(clf, scoring=f1_scorer, param_grid=parameters, cv=5)
    grid_obj = grid_obj.fit(X_train, y_train)
    # 得到最佳的模型
    clf = grid_obj.best_estimator_

    # 训练
    clf.fit(X_train, y_train)

    # 预测
    X_pred = clf.predict(X_test)

    cp = sum(y_test == X_pred) / float(len(X_pred))

    print('准确率:', cp)

    # # 保存模型和加载模型
    # import joblib
    #
    # # 保存模型
    # joblib.dump(clf, 'xgboost_model_demo.model')
    # # 读取模型
    # xgb = joblib.load('xgboost_model_demo.model')
    # # 然后我们尝试来进行一个预测
    # sample1 = X_test.sample(n=10, random_state=2)
    # y_test_1 = y_test.sample(n=10, random_state=2)
    # print(sample1)
    # print(y_test_1)
    # # 进行预测
    # y_pred = xgb.predict(sample1)
    # print("实际值:%s \n预测值:%s" % (y_test_1.values, y_pred))


train_predict(playing_stat)
