# coding:utf-8

# 数据处理逻辑
def handle(data):
    season_data = data['2005-2006']

    new_season_data = []
    for item in season_data:
        item['FTHG'] = item['home_team_goals'] - item['away_team_goals']
        item['ATGS'] = item['away_team_goals'] - item['home_team_goals']
        print(item)
        new_season_data.append(item)

    return
