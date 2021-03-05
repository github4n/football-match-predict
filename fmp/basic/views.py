import json

from django.forms import model_to_dict
from django.http import JsonResponse
from django.shortcuts import render
from rest_framework.views import APIView

from .models import MatchData
from .read_csv import read_csv_line
from .prediction import predict


class PredictView(APIView):
    def get(self, request):
        return render(request, 'predict-test.html')

    def post(self, request):
        response = {
            'code': 200,
            'message': '请求成功'
        }
        season = request.data.get('season')
        num = request.data.get('num', 1)
        num = int(num) + 1

        queryset = MatchData.objects.filter(match_season=season)[:num]
        obj_list = [model_to_dict(obj) for obj in queryset]
        predict_result, cp = predict(obj_list)
        for i in range(len(predict_result)):
            obj_list[i]['predict_result'] = int(predict_result[i])
            obj_list[i]['match_date'] = obj_list[i]['match_date'].strftime("%Y-%m-%d")

        print(obj_list)

        print(cp)
        response['data'] = obj_list
        return JsonResponse(response, json_dumps_params={
            'indent': 4,
            'ensure_ascii': False
        })


def get_seasons(request):
    response = {
        'code': 200,
        'message': '请求成功'
    }
    season_list = MatchData.objects.order_by('match_season').values_list('match_season').distinct().filter(
        is_trained=False)
    options = []
    for k, v in enumerate(season_list):
        options.append({'label': v[0], 'value': v[0]})

    response['data'] = options
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
