from django.db import models


class MatchData(models.Model):
    """比赛数据"""

    RESULT_CHOICE = [
        ('H', '主场赢'),
        ('D', '平局'),
        ('A', '客场赢'),
    ]

    home_team = models.CharField(max_length=200, blank=True, help_text="主场球队")
    away_team = models.CharField(max_length=200, blank=True, help_text="客场球队")
    home_team_goals = models.SmallIntegerField(default=0, help_text="主场球队进球数")
    away_team_goals = models.SmallIntegerField(default=0, help_text="客场球队进球数")
    match_date = models.DateField(max_length=100, blank=True, help_text="比赛日期")
    match_season = models.CharField(max_length=20, blank=True, help_text="赛季")
    result = models.CharField(max_length=1, choices=RESULT_CHOICE, blank=True, help_text="结果")
