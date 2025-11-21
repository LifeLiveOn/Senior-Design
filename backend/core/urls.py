from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register("customers", views.CustomerViewSet, basename="customer")
router.register("houses", views.HouseViewSet, basename="house")
router.register("house-images", views.HouseImageViewSet, basename="houseimage")
router.register("agent-logs", views.AgentCustomerLogViewSet, basename="agentlog")

urlpatterns = [
    path("", views.sign_in, name="home"),
    path("v1/", include(router.urls)),  # FIXED — adds trailing slash
    path("login/", views.sign_in, name="login"),
    path("google/auth/", views.auth_receive, name="google_auth"),
    path("sign-out/", views.sign_out, name="sign_out"),
    path("login/google/modal/", views.google_login_modal, name="google_login_modal"),
]
