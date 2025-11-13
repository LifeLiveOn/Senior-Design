from django.conf import settings
from django.db import models


class InspectionCase(models.Model):
    # “Client house” — title can be address or custom label
    title = models.CharField(max_length=200)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="inspection_cases",
    )
    created_at = models.DateTimeField(auto_now_add=True)


class ImageAsset(models.Model):
    case = models.ForeignKey(
        InspectionCase,
        on_delete=models.CASCADE,
        related_name="images",
    )
    # allow null/blank so migrations won’t complain about existing rows
    file = models.ImageField(
        upload_to="cases/%Y/%m/%d/",
        null=True,
        blank=True,
    )
    status = models.CharField(max_length=32, default="uploaded")
    # also allow null/blank here to avoid the prompt you’re seeing
    uploaded_at = models.DateTimeField(
        auto_now_add=True,
        null=True,
        blank=True,
    )
