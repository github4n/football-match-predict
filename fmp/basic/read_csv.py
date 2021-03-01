import pandas as pd


# HomeTeam: 主场球队名
# AwayTeam: 客场球队名
# FTHG: 主场球队全场进球数
# FTAG: 客场球队全场进球数
# FTR: 比赛结果 ( H= 主场赢, D= 平局, A= 客场赢)

def read_csv_line(filepath):
    data = pd.read_csv(filepath, error_bad_lines=False, warn_bad_lines=False).dropna(axis=0, how='all')
    content = data[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'Date']]

    # 根据年份确定赛季
    year_set = set()
    for item in content['Date']:
        year = '20' + item.split(r'/')[2]
        year_set.add(year)
    year_list = list(year_set)
    year_list.sort()
    match_season = '-'.join(year_list)

    # 当循环碰到yield时 循环暂停 把yield后面的值 传递给调用者
    for k, s in content.iterrows():
        date_list = s['Date'].split(r'/')
        date_list.reverse()
        date_str = '20' + '-'.join(date_list)

        line_dict = {
            'match_date': date_str,
            'match_season': match_season,
            'home_team': s['HomeTeam'],
            'away_team': s['AwayTeam'],
            'home_team_goals': s['FTHG'],
            'away_team_goals': s['FTAG'],
            'result': s['FTR'],
        }
        yield line_dict


if __name__ == '__main__':
    local_path = r'../dataset/2000-01.csv'  # 存放数据的路径
    g = read_csv_line(local_path)

    for i in range(380):
        print(next(g))
