import os
from urllib import response
from django.shortcuts import render, redirect

from rest_framework.decorators import api_view, permission_classes, authentication_classes
from django.views.decorators.csrf import csrf_exempt
from google.oauth2 import id_token
from google.auth.transport import requests
from rest_framework.response import Response
from django.contrib.auth.models import User
from rest_framework import viewsets, permissions

from .models import AgentCustomerLog, HouseImage, House, Customer
from rest_framework.permissions import AllowAny
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework.permissions import BasePermission
from django.contrib.auth import get_user_model
from .utils import upload_file_to_bucket

from .serializers import (
    CustomerSerializer,
    HouseSerializer,
    HouseImageSerializer,
    AgentCustomerLogSerializer
)

User = get_user_model()


class DebugOrJWTAuthenticated(BasePermission):
    """
    Allow access if:
    - debug_user exists in session, OR
    - user is authenticated via JWT
    """

    def has_permission(self, request, view):
        if "debug_user" in request.session:
            return True
        return request.user and request.user.is_authenticated


@csrf_exempt
def sign_in(request):
    return render(request, "sign_in.html")


@csrf_exempt
@api_view(["POST"])
@permission_classes([AllowAny])
@authentication_classes([])
def auth_receive(request):
    """
    Google calls this URL after user signs in.
    """

    token = request.POST.get("credential")
    if not token:
        return Response({"error": "Missing credential"}, status=400)

    try:
        user_data = id_token.verify_oauth2_token(
            token,
            requests.Request(),
            os.environ.get("GOOGLE_CLIENT_ID")
        )
    except ValueError:
        return Response({"error": "Invalid token."}, status=400)

    user, created = User.objects.get_or_create(
        email=user_data["email"],
        defaults={
            "username": user_data["email"],
            "first_name": user_data.get("name", "")
        }
    )

    refresh = RefreshToken.for_user(user)

    # Debug login (only for local dev)
    request.session["debug_user"] = {
        "email": user.email,
        "name": user.first_name,
        "access": str(refresh.access_token),
        "refresh": str(refresh)
    }

    # return redirect("index")
    response = Response({
        "user": {
            "email": user.email,
            "name": user.first_name,
        }
    })
    response.set_cookie(
        key="access",
        value=str(refresh.access_token),
        httponly=True,
        secure=False,         # True in production
        samesite="Lax",       # MUST match CSRF settings
        path="/"
    )

    response.set_cookie(
        key="refresh",
        value=str(refresh),
        httponly=True,
        secure=False,
        samesite="Lax",
        path="/"
    )
    return response


def sign_out(request):
    if "debug_user" in request.session:
        del request.session["debug_user"]
    return redirect("login")


@api_view(["GET"])
@permission_classes([AllowAny])
def index(request):
    if "debug_user" in request.session:
        user = request.session.get("debug_user")
    else:
        user = None
        return Response({"message": "User not authenticated or pass token of jwt"})
    return Response({"message": "Hello from the backend! with session test", "user": user})


class isAgentOwner(permissions.BasePermission):
    def has_object_permission(self, request, view, obj):
        # Customer
        if hasattr(obj, "agent"):
            return obj.agent == request.user

        # House
        if hasattr(obj, "customer"):
            return obj.customer.agent == request.user

        # HouseImage
        if hasattr(obj, "house"):
            return obj.house.customer.agent == request.user

        return False


class isAgentOwner(permissions.BasePermission):
    def has_object_permission(self, request, view, obj):
        # Customer
        if hasattr(obj, "agent"):
            return obj.agent == request.user

        # House
        if hasattr(obj, "customer"):
            return obj.customer.agent == request.user

        # HouseImage
        if hasattr(obj, "house"):
            return obj.house.customer.agent == request.user

        return False


class CustomerViewSet(viewsets.ModelViewSet):
    """
    support CRUD operations for Customer model
    API endpoints for managing customers.
    ex:
    - List all customers -> GET /customers/
    - Retrieve a specific customer -> 
    - Create a new customer
    - Update an existing customer
    - Delete a customer
    """
    queryset = Customer.objects.all()
    serializer_class = CustomerSerializer
    permission_classes = [DebugOrJWTAuthenticated,
                          isAgentOwner]

    # get create
    def get_queryset(self):
        return self.queryset.filter(agent=self.request.user)

    # post create
    def perform_create(self, serializer):
        serializer.save(agent=self.request.user)

    # update
    # delete


class HouseViewSet(viewsets.ModelViewSet):
    queryset = House.objects.all()
    serializer_class = HouseSerializer
    permission_classes = [DebugOrJWTAuthenticated,
                          isAgentOwner]

    def get_queryset(self):
        return self.queryset.filter(customer__agent=self.request.user)

    def perform_create(self, serializer):
        customer = serializer.validated_data["customer"]
        if customer.agent != self.request.user:
            raise permissions.PermissionDenied(
                "You do not own this customer.")

        serializer.save()


class HouseImageViewSet(viewsets.ModelViewSet):
    queryset = HouseImage.objects.all()
    serializer_class = HouseImageSerializer
    permission_classes = [DebugOrJWTAuthenticated, isAgentOwner]

    def get_queryset(self):
        return HouseImage.objects.filter(house__customer__agent=self.request.user)

    def perform_create(self, serializer):
        file_obj = self.request.FILES.get("file")

        if not file_obj:
            raise Exception("No file uploaded")

        url = upload_file_to_bucket(file_obj, "roofvision-images")

        if not url:
            raise Exception("Failed to upload image")

        serializer.save(image_url=url)


class AgentCustomerLogViewSet(viewsets.ReadOnlyModelViewSet):
    serializer_class = AgentCustomerLogSerializer
    permission_classes = [DebugOrJWTAuthenticated]

    def get_queryset(self):
        return AgentCustomerLog.objects.filter(agent=self.request.user)
