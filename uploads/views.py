# uploads/views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.permissions import IsAuthenticated
from django.shortcuts import get_object_or_404

from inspections.models import InspectionCase, ImageAsset
from inspections.serializers import ImageSerializer


class UploadImageForCase(APIView):
    """
    POST /api/cases/<case_id>/images/upload
    form-data: file=<image>
    """
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, case_id):
        # Make sure the case belongs to the logged-in user
        case = get_object_or_404(
            InspectionCase,
            id=case_id,
            created_by=request.user,
        )

        file_obj = request.FILES.get("file")
        if not file_obj:
            return Response({"detail": "file is required"}, status=400)

        image = ImageAsset.objects.create(
            case=case,
            file=file_obj,
            status="uploaded",
        )

        data = ImageSerializer(image, context={"request": request}).data
        return Response(data, status=201)


class ListImagesForCase(APIView):
    """
    GET /api/cases/<case_id>/images
    returns: [{id, url, status, uploaded_at}, ...]
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, case_id):
        case = get_object_or_404(
            InspectionCase,
            id=case_id,
            created_by=request.user,
        )

        qs = case.images.order_by("-uploaded_at")
        data = ImageSerializer(qs, many=True, context={"request": request}).data
        return Response(data)
