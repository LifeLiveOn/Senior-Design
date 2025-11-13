from django.urls import path
from .views import CaseListCreate, upload_case_image

urlpatterns = [
    path("cases/", CaseListCreate.as_view(), name="case-list-create"),
    path("cases/<int:case_id>/images/upload", upload_case_image, name="case-image-upload"),
    # (optionally)
    # path("cases/<int:case_id>/images", list_case_images, name="case-images-list"),
]
