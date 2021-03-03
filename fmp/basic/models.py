from django.db import models


class MatchData(models.Model):
    """比赛数据"""

    RESULT_CHOICE = [
        ('H', '主场赢'),
        ('D', '平局'),
        ('A', '客场赢'),
    ]

    home_team = models.CharField(max_length=200, blank=True, verbose_name="主场球队")
    away_team = models.CharField(max_length=200, blank=True, verbose_name="客场球队")
    home_team_goals = models.SmallIntegerField(default=0, verbose_name="主场球队进球数")
    away_team_goals = models.SmallIntegerField(default=0, verbose_name="客场球队进球数")
    match_date = models.DateField(max_length=100, blank=True, verbose_name="比赛日期")
    match_season = models.CharField(max_length=20, blank=True, verbose_name="赛季")
    result = models.CharField(max_length=1, choices=RESULT_CHOICE, blank=True, verbose_name="结果")
    is_trained = models.BooleanField(default=False, verbose_name="是否已训练")

    def __str__(self):
        return self.match_season + ': ' + self.home_team + ' VS ' + self.away_team

    class Meta:
        verbose_name_plural = '数据集'
        verbose_name = '数据集'
        ordering = ('-match_date',)
