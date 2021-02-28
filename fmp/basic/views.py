from django.forms import model_to_dict
from django.http import JsonResponse
from django.shortcuts import render
from .models import MatchData


def index(request):
    response = {
        'code': 200,
        'message': '请求成功'
    }
    queryset = MatchData.objects.aggregate()
    # # 按月分组统计数据
    # queryset = queryset.values('customer', 'customer__name', ).order_by().annotate(
    #     year=ExtractYear('stock_out_time'),
    #     month=ExtractMonth('stock_out_time'),
    #     amount_1=Sum("roundsteelstockoutorderdetail__amount"),
    #     amount_2=Sum("squaresteelstockoutorderdetail__amount")
    # )
    response['data'] = []
    for instance in queryset:
        response['data'].append(model_to_dict(instance))

    return JsonResponse(response, json_dumps_params={
        'indent': 4,
        'ensure_ascii': False
    })


