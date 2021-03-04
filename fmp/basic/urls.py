# coding:utf-8
from django.urls import path, include
from . import views

app_name = 'basic'

urlpatterns = [
    path('get-seasons/', views.get_seasons),
    path('predict-test/', views.predict_test),
    path('import-csv/', views.import_cvs),
]
