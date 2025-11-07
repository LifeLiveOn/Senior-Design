from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import DetectionRecordViewSet

router = DefaultRouter()
router.register(r'detections', DetectionRecordViewSet, basename='detection')

urlpatterns = [
    path('', include(router.urls)),
]
