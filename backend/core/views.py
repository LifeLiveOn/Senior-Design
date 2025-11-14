from django.shortcuts import render, redirect

from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from django.contrib.auth.models import User
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework import viewsets, permissions
from .models import AgentCustomerLog, HouseImage, House, Customer
from rest_framework.permissions import AllowAny

# Create your views here.
from .utils import verify_google_token
from .serializers import (
    CustomerSerializer,
    HouseSerializer,
    HouseImageSerializer,
    AgentCustomerLogSerializer
)


@api_view(["GET"])
@permission_classes([AllowAny])
def index(request):
    return Response({"message": "Hello from the backend!"})


@api_view(["POST"])
@permission_classes([AllowAny])
def google_auth(request):
    email = request.data.get("email")
    name = request.data.get("name", "")
    if not email:
        return Response({"error": "Email is required."}, status=400)

    user, created = User.objects.get_or_create(
        email=email, defaults={"username": email, "first_name": name}
    )

    refresh = RefreshToken.for_user(user)

    return Response({
        "message": f"Login successful welcome back! {user.username}",
        "email": email,
        "name": name,
        "access": str(refresh.access_token),
        "refresh": str(refresh),
        "is_new": created
    })


class isAgentOwner(permissions.BasePermission):
    """
    custom permission: only allow the agent that owns a customer to view/edit it
    """

    def has_object_permission(self, request, view, obj):
        return obj.agent == request.user if hasattr(obj, "agent") else obj.customer.agent == request.user


class CustomerViewSet(viewsets.ModelViewSet):
    queryset = Customer.objects.all()
    serializer_class = CustomerSerializer
    permission_classes = [permissions.IsAuthenticated, isAgentOwner]

    def get_queryset(self):
        return self.queryset.filter(agent=self.request.user)

    def perform_create(self, serializer):
        serializer.save(agent=self.request.user)


class HouseViewSet(viewsets.ModelViewSet):
    serializer_class = HouseSerializer
    permission_classes = [permissions.IsAuthenticated, isAgentOwner]

    def get_queryset(self):
        return self.queryset.filter(agent=self.request.user)

    def perform_create(self, serializer):
        customer = serializer.validated_data["customer"]
        if customer.agent != self.request.user:
            raise permissions.PermissionDenied("You do not own this customer.")

        serializer.save()


class HouseImageViewSet(viewsets.ModelViewSet):
    serializer_class = HouseImageSerializer
    permission_classes = [permissions.IsAuthenticated, isAgentOwner]

    def get_queryset(self):
        return HouseImage.objects.filter(house__customer__agent=self.request.user)

    def perform_create(self, serializer):
        house = serializer.validated_data["house"]
        if house.customer.agent != self.request.user:
            raise permissions.PermissionDenied("You do not own this house.")

        serializer.save()


class AgentCustomerLogViewSet(viewsets.ReadOnlyModelViewSet):
    serializer_class = AgentCustomerLogSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return AgentCustomerLog.objects.filter(agent=self.request.user)
