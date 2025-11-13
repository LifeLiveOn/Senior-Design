from django.urls import path
from .views import CaseListCreate, upload_case_image, UserCreateView

urlpatterns = [
    path("cases/", CaseListCreate.as_view(), name="case-list-create"),
    path(
        "cases/<int:case_id>/images/upload/",
        upload_case_image,
        name="case-image-upload",
    ),
    path(
        "users/register/",
        UserCreateView.as_view(),
        name="user-create",
    ),
]
