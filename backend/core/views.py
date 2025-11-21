"""
Core API views for the backend.

This module provides:
- Authentication entrypoints (Google OAuth callback, sign-in/out helpers)
- DRF ViewSets for `Customer`, `House`, `HouseImage`, and agent logs
- Utility permission classes for JWT or session-based debug access
- A prediction endpoint that runs RF-DETR inference and stores results
"""

from .serializers import (
    CustomerSerializer,
    HouseSerializer,
    HouseImageSerializer,
    AgentCustomerLogSerializer
)
import os
from urllib import response
from django.http import HttpResponseRedirect
from django.shortcuts import render, redirect

from rest_framework.decorators import api_view, permission_classes, authentication_classes
from django.views.decorators.csrf import csrf_exempt
from google.oauth2 import id_token
from google.auth.transport import requests
from rest_framework.response import Response
from django.contrib.auth.models import User
from rest_framework import viewsets, permissions

from .authentication import CookieJWTAuthentication

from .models import AgentCustomerLog, HouseImage, House, Customer
from rest_framework.permissions import AllowAny
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework.permissions import BasePermission
from django.contrib.auth import get_user_model
from .utils import upload_file_to_bucket, upload_local_file_to_bucket
from .services import RFDETRService
from django.utils import timezone
from model_utils import run_rfdetr_inference, run_rfdetr_inference_tiled


User = get_user_model()


class DebugOrJWTAuthenticated(BasePermission):
    """Permission that allows access for debug sessions or JWT users.

    Access is granted when either of these is true:
    - `debug_user` exists in the Django session (development/testing aid)
    - The request has a valid authenticated user (Cookie JWT)
    """

    def has_permission(self, request, view):
        """Return True if the request is from a debug session or JWT user."""
        if "debug_user" in request.session:
            return True
        return request.user and request.user.is_authenticated


@authentication_classes([CookieJWTAuthentication])
def sign_in(request):
    """Render the sign-in page.

    Returns the template `backend/sign_in.html`. The actual JWT auth
    happens after Google login via `auth_receive` which sets HttpOnly
    cookies for access/refresh tokens.
    """
    return render(request, "backend/sign_in.html")


def google_login_modal(request):
    """Render the Google login modal partial used by the frontend UI."""
    return render(request, "backend/modals/google_login_modal.html")

# @csrf_exempt


@api_view(["POST"])
@permission_classes([AllowAny])
@authentication_classes([])
def auth_receive(request):
    """Handle Google OAuth callback, mint JWTs, and redirect to frontend.

    - URL: `/google/auth`
    - Method: POST
    - Body: `{ "credential": "<Google ID Token>" }`

    Verifies the Google ID token, creates or fetches the user, issues
    SimpleJWT refresh/access tokens, sets them as HttpOnly cookies, and
    redirects to the React app customers page.
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

    print("User authenticated:", user.email, "Created:", created)
    # -----------------------------
    # REDIRECT TO REACT CUSTOMERS PAGE
    # -----------------------------
    redirect_url = "http://localhost:3000/customers"

    response = HttpResponseRedirect(redirect_url)

    # Set HttpOnly cookies
    response.set_cookie(
        key="access",
        value=str(refresh.access_token),
        httponly=True,
        secure=False,
        samesite="Lax",
        path="/",
    )
    response.set_cookie(
        key="refresh",
        value=str(refresh),
        httponly=True,
        secure=False,
        samesite="Lax",
        path="/",
    )

    return response


@api_view(["GET"])
@permission_classes([AllowAny])
def sign_out(request):
    """Clear JWT cookies and redirect to the `login` named route."""
    response = redirect("login")
    response.delete_cookie("access")
    response.delete_cookie("refresh")
    return response


@api_view(["GET"])
@permission_classes([AllowAny])
def index(request):
    """Simple health/auth check endpoint.

    If `debug_user` is present in the session, returns a greeting payload
    with that user. Otherwise returns a not-authenticated message.
    """
    if "debug_user" in request.session:
        user = request.session.get("debug_user")
    else:
        user = None
        return Response({"message": "User not authenticated or pass token of jwt"})
    return Response({"message": "Hello from the backend! with session test", "user": user})


class AgentCustomerLogViewSet(viewsets.ReadOnlyModelViewSet):
    """Read-only viewset to list logs for the authenticated agent."""
    serializer_class = AgentCustomerLogSerializer
    permission_classes = [DebugOrJWTAuthenticated]

    def get_queryset(self):
        """Return logs scoped to the authenticated agent only."""
        return AgentCustomerLog.objects.filter(agent=self.request.user)


class isAgentOwner(permissions.BasePermission):
    """Object-level permission that ensures the agent owns the resource.

    Supports `Customer`, `House`, and `HouseImage` by walking the
    ownership chain to verify `request.user` is the agent.
    """

    def has_object_permission(self, request, view, obj):
        """Return True if the current user owns the object or its parent."""
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
    """CRUD endpoints for customers owned by the authenticated agent.

    Examples:
    - List:    GET    /customers/
    - Create:  POST   /customers/
    - Detail:  GET    /customers/{id}/
    - Update:  PUT    /customers/{id}/
    - Partial: PATCH  /customers/{id}/
    - Delete:  DELETE /customers/{id}/
    """
    queryset = Customer.objects.all()
    serializer_class = CustomerSerializer

    def get_permissions(self):
        """Use ownership checks for object-level operations."""
        if self.action in ["retrieve", "update", "partial_update", "destroy"]:
            # Object-level operations require ownership check
            return [DebugOrJWTAuthenticated(), isAgentOwner()]
        return [DebugOrJWTAuthenticated()]

    # get create
    def get_queryset(self):
        """Limit to customers owned by the authenticated agent."""
        return self.queryset.filter(agent=self.request.user)

    # post create
    def perform_create(self, serializer):
        """Automatically set the `agent` field to the current user."""
        serializer.save(agent=self.request.user)

    # update
    # delete


class HouseViewSet(viewsets.ModelViewSet):
    """CRUD endpoints for houses belonging to the agent's customers."""
    queryset = House.objects.all()
    serializer_class = HouseSerializer
    permission_classes = [DebugOrJWTAuthenticated,
                          isAgentOwner]

    def get_queryset(self):
        """Limit to houses whose customers are owned by the agent."""
        return self.queryset.filter(customer__agent=self.request.user)

    def perform_create(self, serializer):
        """Validate customer ownership and create the house record."""
        customer = serializer.validated_data["customer"]
        if customer.agent != self.request.user:
            raise permissions.PermissionDenied(
                "You do not own this customer.")

        serializer.save()


class HouseImageViewSet(viewsets.ModelViewSet):
    """CRUD endpoints for house images, uploading to cloud storage on create."""
    queryset = HouseImage.objects.all()
    serializer_class = HouseImageSerializer
    permission_classes = [DebugOrJWTAuthenticated, isAgentOwner]

    def get_queryset(self):
        """Limit to images under houses owned by the agent's customers."""
        return HouseImage.objects.filter(house__customer__agent=self.request.user)

    def perform_create(self, serializer):
        """Upload the provided file to storage and persist its URL.

        Expects a multipart field named `file` containing the image.
        """
        file_obj = self.request.FILES.get("file")

        if not file_obj:
            raise Exception("No file uploaded")

        url = upload_file_to_bucket(file_obj, "roofvision-images")

        if not url:
            raise Exception("Failed to upload image")

        serializer.save(image_url=url)


@api_view(["POST"])
def run_prediction(request, house_id):
    """Run RF-DETR inference for all images of the given house.

    Body (JSON):
    - `mode` (str): "normal" or "tiled" (default: "normal")
    - `threshold` (float): detection confidence threshold (default: 0.4)
    - `tile_size` (int): tile size if tiled mode used (default: 560)

    For each image, stores a predicted image in cloud storage and updates
    `predicted_url` and `predicted_at`. Returns a summary payload.
    """
    try:
        house = (
            House.objects
            .prefetch_related("images")
            .get(id=house_id)
        )
    except House.DoesNotExist:
        return Response({"error": "House not found"}, status=404)

    if not house.images.exists():
        return Response({"message": "No images found for this house."})

    mode = request.data.get("mode", "normal")
    threshold = float(request.data.get("threshold", 0.4))
    tile_size = int(request.data.get("tile_size", 560))

    bucket_name = "roofvision-images"
    results = []

    for img in house.images.all():
        try:
            # This function already stores detections internally
            _, pred_path = RFDETRService.predict(
                image_path_or_url=img.image_url,
                mode=mode,
                threshold=threshold,
                tile_size=tile_size
            )

            predicted_url = None
            if pred_path:
                predicted_url = upload_local_file_to_bucket(
                    pred_path, bucket_name)

                img.predicted_url = predicted_url
                img.predicted_at = timezone.now()
                img.save()

            results.append({
                "image_id": img.id,
                "original_image": img.image_url,
                "predicted_image": predicted_url,
            })

        except Exception as e:
            results.append({
                "image_id": img.id,
                "original_image": img.image_url,
                "error": str(e)
            })

    return Response({
        "house_id": house_id,
        "total_images": len(results),
        "results": results
    })
