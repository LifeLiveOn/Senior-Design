# server/urls.py
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("admin/", admin.site.urls),

    # Cases (client houses)
    path("api/", include("inspections.urls")),

    # Upload + list images for a case
    path("api/", include("uploads.urls")),
]

# Serve uploaded images in dev so Streamlit can see them
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
