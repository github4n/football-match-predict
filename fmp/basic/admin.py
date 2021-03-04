from django.contrib import admin
from django.forms import model_to_dict
from django.http import JsonResponse
from import_export.admin import ImportExportModelAdmin
from import_export import resources
from simpleui.admin import AjaxAdmin
from .models import MatchData
from .prediction import train, predict


def get_all_season():
    season_list = MatchData.objects.order_by('match_season').values_list('match_season').distinct().filter(
        is_trained=False)
    options = []
    for k, v in enumerate(season_list):
        options.append({'key': k + 1, 'label': v[0]})
    return options


class MatchDataResource(resources.ModelResource):
    class Meta:
        model = MatchData


@admin.register(MatchData)
class MatchDataAdmin(ImportExportModelAdmin, AjaxAdmin):
    # 定义列表页面的需要显示的字段
    list_display = (
        'team', 'home_team', 'away_team', 'home_team_goals', 'away_team_goals', 'match_date', 'match_season', 'result',
        'is_trained'
    )

    def team(self, obj):
        return obj.home_team + ' VS ' + obj.away_team

    team.short_description = '比赛'

    # 设置搜索字段
    search_fields = ['home_team', 'away_team', 'match_date']

    # 设置可以筛选的字段
    list_filter = ['match_season', 'result']

    # 每页显示条目数
    list_per_page = 20

    # 自定义按钮
    actions = ['train', ]

    def train(self, request, queryset):
        post = request.POST
        checkbox_value = post['checkbox']

        if not checkbox_value:
            return JsonResponse(data={
                'status': 'error',
                'msg': '未选中数据！'
            })

        checkbox_value_list = checkbox_value.split(',')
        queryset = MatchData.objects.filter(match_season__in=checkbox_value_list)
        obj_list = [model_to_dict(obj) for obj in queryset]
        train(obj_list)
        # queryset.update(is_trained=True)

        return JsonResponse(data={
            'status': 'success',
            'msg': '处理成功！'
        })

    train.short_description = '训练'
    train.type = 'success'
    train.icon = 'el-icon-s-check'
    train.layer = {
        # 这里指定对话框的标题
        'title': '请选择赛季',

        # 确认按钮显示文本
        'confirm_button': '确认',
        # 取消按钮显示文本
        'cancel_button': '取消',

        # 弹出层对话框的宽度，默认50%
        'width': '40%',
        # 表单中 label的宽度，对应element-ui的 label-width，默认80px
        'labelWidth': "80px",

        # 定义表单元素
        'params': [{
            'type': 'checkbox',
            'key': 'checkbox',
            'value': [],
            'label': '',
            'options': get_all_season()
        }]
    }
