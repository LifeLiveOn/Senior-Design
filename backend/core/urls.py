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
    path("login", views.sign_in, name="login"),
    path("google/auth", views.auth_receive, name="google_auth"),
    path("sign-out", views.sign_out, name="sign_out"),
    path("index", views.index, name="index"),
]
