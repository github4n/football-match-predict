import pandas as pd
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

data = res_name[0][['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']]
print(type(data))

for _, row in data.iterrows():
    row_dict = {
        'home_team': row['HomeTeam'],
        'away_team': row['AwayTeam'],
        'home_team_goals': row['FTHG'],
        'away_team_goals': row['FTAG'],
        'result': row['FTR'],
    }
    print(_, row_dict)
