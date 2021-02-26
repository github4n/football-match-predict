# 导入必须的包
import warnings

warnings.filterwarnings('ignore')  # 防止警告文件的包
import pandas as pd  # 数据分析包
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
    # print(res_name)
    # print(filecsv_list)


# 1.1 读取 csv 数据
file_name(local_path)

# 1.2 时间列表
time_list = [filecsv_list[i][0:4] for i in range(len(filecsv_list))]

# 1.3 用 Pandas.read_csv() 接口读取数据
for i in range(len(res_name)):
    res_name[i] = pd.read_csv(local_path + filecsv_list[i], error_bad_lines=False, warn_bad_lines=False).dropna(
        axis=0, how='all')
    print('第%2s个文件是%s,数据大小为%s' % (i + 1, filecsv_list[i], res_name[i].shape))

# 1.4 删除空值的接口
res_name[14] = res_name[14].dropna(axis=0, how='all')

# 1.5 删除行数不是 380 的文件名
for i in range(len(res_name), 0, -1):
    # 采用从大到小的遍历方式，然后进行删除不满足条件的。
    if res_name[i - 1].shape[0] != 380:
        key = 'res_name[' + str(i) + ']'
        print('删除的数据是：%s年的数据，文件名：%s大小是：%s' % (time_list[i - 1], key, res_name[i - 1].shape))
        res_name.pop(i - 1)
        time_list.pop(i - 1)
        continue

"""2. 数据清洗和预处理"""

# 2.1 挑选信息列
# 将挑选的信息放在一个新的列表中
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
    print(time_list[i], 'playing_statistics[' + str(i) + ']', playing_statistics[i].shape)


# 2.2 分析原始数据

# 2.2.1 统计所有主场球队都会胜利的准确率
def predictions_0(data):
    """
    当我们统计所有主场球队都赢，那么我们预测的结果是什么
    返回值是预测值和实际值
    """
    predictions = []
    for _, game in data.iterrows():

        if game['FTR'] == 'H':
            predictions.append(1)
        else:
            predictions.append(0)
    # 返回预测结果
    return pd.Series(predictions)


# 那我们对19年全部主场球队都赢的结果进行预测，获取预测的准确率。
avg_acc_sum = 0
for i in range(len(playing_statistics)):
    predictions = predictions_0(playing_statistics[i])
    acc = sum(predictions) / len(playing_statistics[i])
    avg_acc_sum += acc
    print("%s年数据主场全胜预测的准确率是%s" % (time_list[i], acc))
print('共%s年的平均准确率是：%s' % (len(playing_statistics), avg_acc_sum / len(playing_statistics)))


# 2.2.2 统计所有客场球队都会胜利的准确率
def predictions_1(data):
    """
    当我们统计所有客场球队都赢，那么我们预测的结果是什么
    返回值是预测值和实际值
    """
    predictions = []
    for _, game in data.iterrows():

        if game['FTR'] == 'A':
            predictions.append(1)
        else:
            predictions.append(0)
    # 返回预测结果
    return pd.Series(predictions)


# 那我们对19年客场球队都赢的结果进行预测，获取预测的准确率。
for i in range(len(playing_statistics)):
    predictions = predictions_1(playing_statistics[i])
    acc = sum(predictions) / len(playing_statistics[i])
    print("%s年数据客场全胜预测的准确率是%s" % (time_list[i], acc))

"""3. 特征工程"""


# 3.1.1 计算每个队周累计净胜球数量
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
    print(GoalsDifference)

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


for i in range(len(playing_statistics)):
    playing_statistics[i] = get_gss(playing_statistics[i])

print('查看构造特征后的05-06年的后五条数据:')
print(playing_statistics[2].tail())


# 3.1.2 统计主客场队伍到当前比赛周的累计得分
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


for i in range(len(playing_statistics)):
    playing_statistics[i] = get_agg_points(playing_statistics[i])

print('查看构造特征后的05-06年的后五条数据:')
print(playing_statistics[2].tail())


# 3.1.3 统计某支队伍最近三场比赛的表现
def get_form(playing_stat, num):
    form = get_matchres(playing_stat)
    form_final = form.copy()
    for i in range(num, 39):
        form_final[i] = ''
        j = 0
        while j < num:
            form_final[i] += form[i - j]
            j += 1
    return form_final


def add_form(playing_stat, num):
    form = get_form(playing_stat, num)
    # M 代表 unknown， 因为没有那么多历史
    h = ['M' for i in range(num * 10)]
    a = ['M' for i in range(num * 10)]
    j = num
    for i in range((num * 10), 380):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam

        past = form.loc[ht][j]
        h.append(past[num - 1])

        past = form.loc[at][j]
        a.append(past[num - 1])

        if ((i + 1) % 10) == 0:
            j = j + 1

    playing_stat['HM' + str(num)] = h
    playing_stat['AM' + str(num)] = a

    return playing_stat


def add_form_df(playing_statistics):
    playing_statistics = add_form(playing_statistics, 1)
    playing_statistics = add_form(playing_statistics, 2)
    playing_statistics = add_form(playing_statistics, 3)
    return playing_statistics


for i in range(len(playing_statistics)):
    playing_statistics[i] = add_form_df(playing_statistics[i])

print('查看构造特征后的05-06年的后五条数据:')
print(playing_statistics[2].tail())


# 3.1.4 加入比赛周特征（第几个比赛周）
def get_mw(playing_stat):
    j = 1
    MatchWeek = []
    for i in range(380):
        MatchWeek.append(j)
        if ((i + 1) % 10) == 0:
            j = j + 1
    playing_stat['MW'] = MatchWeek
    return playing_stat


for i in range(len(playing_statistics)):
    playing_statistics[i] = get_mw(playing_statistics[i])

print('查看构造特征后的05-06年的后五条数据:')
print(playing_statistics[2].tail())

# 3.1.5 合并比赛的信息
# 将各个DataFrame表合并在一张表中
playing_stat = pd.concat(playing_statistics, ignore_index=True)

# HTGD, ATGD ,HTP, ATP的值 除以 week 数，得到平均分
cols = ['HTGD', 'ATGD', 'HTP', 'ATP']
playing_stat.MW = playing_stat.MW.astype(float)
for col in cols:
    playing_stat[col] = playing_stat[col] / playing_stat.MW

print('查看构造特征后数据集的后5五条数据:')
print(playing_stat.tail())

# 3.3 分析我们构造的数据
# 比赛总数
n_matches = playing_stat.shape[0]

# 特征数
n_features = playing_stat.shape[1] - 1

# 主场获胜的数目
n_homewins = len(playing_stat[playing_stat.FTR == 'H'])

# 主场获胜的比例
win_rate = (float(n_homewins) / (n_matches)) * 100

# Print the results
print("比赛总数: {}".format(n_matches))
print("总特征数: {}".format(n_features))
print("主场胜利数: {}".format(n_homewins))
print("主场胜率: {:.2f}%".format(win_rate))


# 3.4 解决样本不均衡问题
# 定义 target ，也就是否 主场赢
def only_hw(string):
    if string == 'H':
        return 'H'
    else:
        return 'NH'


playing_stat['FTR'] = playing_stat.FTR.apply(only_hw)

# 3.5 将数据分为特征值和标签值
# 把数据分为特征值和标签值
X_all = playing_stat.drop(['FTR'], 1)
y_all = playing_stat['FTR']
print('特征值的长度:', len(X_all))


# 3.6 数据归一化、标准化
def convert_1(data):
    max = data.max()
    min = data.min()
    return (data - min) / (max - min)


r_data = convert_1(X_all['HTGD'])
# 数据标准化
from sklearn.preprocessing import scale

cols = [['HTGD', 'ATGD', 'HTP', 'ATP']]
for col in cols:
    X_all[col] = scale(X_all[col])

# 3.7 转换特征数据类型
# 把这些特征转换成字符串类型
X_all.HM1 = X_all.HM1.astype('str')
X_all.HM2 = X_all.HM2.astype('str')
X_all.HM3 = X_all.HM3.astype('str')
X_all.AM1 = X_all.AM1.astype('str')
X_all.AM2 = X_all.AM2.astype('str')
X_all.AM3 = X_all.AM3.astype('str')


def preprocess_features(X):
    '''把离散的类型特征转为哑编码特征 '''
    output = pd.DataFrame(index=X.index)
    for col, col_data in X.iteritems():
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix=col)
        output = output.join(col_data)
    return output


X_all = preprocess_features(X_all)
print("Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns)))

"""4.建立机器学习模型并进行预测"""

from sklearn.model_selection import train_test_split

# 把标签映射为0和1
y_all = y_all.map({'NH': 0, 'H': 1})
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.3, random_state=2, stratify=y_all)

# 4.3 建立机器学习模型并评估
from time import time
from sklearn.metrics import f1_score


def train_classifier(clf, X_train, y_train):
    ''' 训练模型 '''
    # 记录训练时长
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    print("训练时间 {:.4f} 秒".format(end - start))


def predict_labels(clf, features, target):
    ''' 使用模型进行预测 '''
    # 记录预测时长
    start = time()
    y_pred = clf.predict(features)
    end = time()
    print("预测时间 in {:.4f} 秒".format(end - start))
    return f1_score(target, y_pred, pos_label=1), sum(target == y_pred) / float(len(y_pred))


def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' 训练并评估模型 '''
    # Indicate the classifier and the training set size
    print("训练 {} 模型，样本数量 {}。".format(clf.__class__.__name__, len(X_train)))
    # 训练模型
    train_classifier(clf, X_train, y_train)
    # 在测试集上评估模型
    f1, acc = predict_labels(clf, X_train, y_train)
    print("训练集上的 F1 分数和准确率为: {:.4f} , {:.4f}。".format(f1, acc))

    f1, acc = predict_labels(clf, X_test, y_test)
    print("测试集上的 F1 分数和准确率为: {:.4f} , {:.4f}。".format(f1, acc))


# 4.3.2 分别初始化，训练和评估模型
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# 分别建立三个模型
clf_A = LogisticRegression(random_state=42)
clf_B = SVC(random_state=42, kernel='rbf', gamma='auto')
clf_C = xgb.XGBClassifier(seed=42)

train_predict(clf_A, X_train, y_train, X_test, y_test)
print('')
train_predict(clf_B, X_train, y_train, X_test, y_test)
print('')
train_predict(clf_C, X_train, y_train, X_test, y_test)
print('')

# 4.4 超参数调整
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import xgboost as xgb

# 设置想要自动调参的参数
parameters = {'n_estimators': [90, 100, 110],
              'max_depth': [5, 6, 7],
              }
# 初始化模型
clf = xgb.XGBClassifier(seed=42)
f1_scorer = make_scorer(f1_score, pos_label=1)
# 使用 grdi search 自动调参
grid_obj = GridSearchCV(clf,
                        scoring=f1_scorer,
                        param_grid=parameters,
                        cv=5)
grid_obj = grid_obj.fit(X_train, y_train)
# 得到最佳的模型
clf = grid_obj.best_estimator_
# print(clf)
# 查看最终的模型效果
f1, acc = predict_labels(clf, X_train, y_train)
print("F1 score and accuracy score for training set: {:.4f} , {:.4f}。".format(f1, acc))

f1, acc = predict_labels(clf, X_test, y_test)
print("F1 score and accuracy score for test set: {:.4f} , {:.4f}。".format(f1, acc))

# 4.5 保存模型和加载模型
import joblib

# 保存模型
joblib.dump(clf, 'xgboost_model.model')
# 读取模型
xgb = joblib.load('xgboost_model.model')
# 然后我们尝试来进行一个预测
sample1 = X_test.sample(n=5, random_state=2)
y_test_1 = y_test.sample(n=5, random_state=2)
print(sample1)
print(y_test_1)
# 进行预测
y_pred = xgb.predict(sample1)
print("实际值:%s \n预测值:%s" % (y_test_1.values, y_pred))
