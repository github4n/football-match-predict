from django.forms import model_to_dict
from django.http import JsonResponse
from django.shortcuts import render
from .models import MatchData
from .read_csv import read_csv_line
from .prediction import handle
from django.core.serializers import serialize
from django.core.serializers import python


def index(request):
    response = {
        'code': 200,
        'message': '请求成功'
    }
    # 获取赛季列表
    season_list = MatchData.objects.values_list('match_season').distinct()

    # 根据赛季分组
    group_result = {}
    for season in season_list:
        queryset = MatchData.objects.filter(match_season=season[0])
        dict_list = [model_to_dict(obj) for obj in queryset]
        group_result[season[0]] = dict_list

    handle(group_result)

    response['data'] = group_result
    return JsonResponse(response, json_dumps_params={
        'indent': 4,
        'ensure_ascii': False
    })


def import_cvs(request):
    response = {
        'code': 200,
        'message': '请求成功'
    }
    csv_file = request.FILES['csv_file']
    g = read_csv_line(csv_file)
    instance_list = []
    while True:
        try:
            line_dict = next(g)
            instance_list.append(MatchData(**line_dict))
        except StopIteration:
            break
    queryset = MatchData.objects.bulk_create(instance_list)  # 批量创建
    response['message'] = 'success, created %d rows' % len(queryset)
    return JsonResponse(response, json_dumps_params={
        'indent': 4,
        'ensure_ascii': False
    })
