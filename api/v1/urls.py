from django.urls import include, path
from api.v1 import views

urlpatterns = [
    path('predict/', views.PredictView.as_view(), name='predict')
]