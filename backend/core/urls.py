from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r"customers", views.CustomerViewSet, basename="customer")
router.register(r"houses", views.HouseViewSet, basename="house")
router.register(r"house-images", views.HouseImageViewSet,
                basename="houseimage")
router.register(r"agent-logs", views.AgentCustomerLogViewSet,
                basename="agentlog")

urlpatterns = [
    path("", include(router.urls)),
    path("auth/google/", views.google_auth, name="google_auth"),
]

urlpatterns += router.urls
