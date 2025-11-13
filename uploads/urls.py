from django.urls import path
from .views import UploadImageForCase, ListImagesForCase

urlpatterns = [
    path("cases/<int:case_id>/images/upload", UploadImageForCase.as_view()),
    path("cases/<int:case_id>/images", ListImagesForCase.as_view()),
]
