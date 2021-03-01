# coding:utf-8
from django.urls import path, include
from . import views

app_name = 'basic'

urlpatterns = [
    path('index/', views.index),
    path('import-csv/', views.import_cvs),
]
