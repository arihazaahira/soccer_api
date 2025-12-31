from django.urls import path
from .views import EvaluateActionView

urlpatterns = [
    path("evaluate/", EvaluateActionView.as_view(), name="evaluate_action"),
]
