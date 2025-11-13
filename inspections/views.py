# inspections/views.py

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny

from rest_framework.decorators import api_view, permission_classes, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status

from .models import InspectionCase, ImageAsset
from .serializers import CaseSerializer


class CaseListCreate(APIView):
    permission_classes = [AllowAny]  # no auth required

    def get(self, request):
        # For now, return all cases (no per-user filtering)
        qs = InspectionCase.objects.all().order_by("-created_at")
        return Response(CaseSerializer(qs, many=True).data)

    def post(self, request):
        title = (request.data.get("title") or "").strip()
        if not title:
            return Response({"detail": "Title is required."}, status=400)

        # created_by is nullable in the model now, so this is fine
        case = InspectionCase.objects.create(title=title)
        return Response(CaseSerializer(case).data, status=201)


@api_view(["POST"])
@permission_classes([AllowAny])               # ⬅ open for now
@parser_classes([MultiPartParser, FormParser])  # ⬅ handle multipart/form-data
def upload_case_image(request, case_id):
    """
    POST /api/cases/<case_id>/images/upload
    Form-data: file=<image>
    """
    # 1) Make sure the case exists
    try:
        case = InspectionCase.objects.get(pk=case_id)
    except InspectionCase.DoesNotExist:
        return Response(
            {"detail": "Case not found."},
            status=status.HTTP_404_NOT_FOUND,
        )

    # 2) Grab the uploaded file
    file_obj = request.FILES.get("file")
    if not file_obj:
        return Response(
            {"detail": "No file uploaded. Use form field name 'file'."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    # 3) Create the ImageAsset
    image = ImageAsset.objects.create(
        case=case,
        file=file_obj,
        status="uploaded",
    )

    # 4) Return a simple JSON response
    return Response(
        {
            "id": image.id,
            "case": image.case_id,
            "file": image.file.url if image.file else None,
            "status": image.status,
            "uploaded_at": image.uploaded_at,
        },
        status=status.HTTP_201_CREATED,
    )
