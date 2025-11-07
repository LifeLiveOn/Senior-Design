from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.core.files.base import ContentFile
from google.cloud import storage
import os
from .models import DetectionRecord
from .serializers import DetectionRecordSerializer

class DetectionRecordViewSet(viewsets.ModelViewSet):
    serializer_class = DetectionRecordSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        return DetectionRecord.objects.filter(agent=self.request.user)
    
    @action(detail=False, methods=['post'])
    def upload_image(self, request):
        """
        Upload image to GCS and save metadata to database.
        Expects: image (file), client_name, client_email, client_phone, metadata (optional JSON)
        """
        try:
            image_file = request.FILES.get('image')
            client_name = request.data.get('client_name')
            client_email = request.data.get('client_email', '')
            client_phone = request.data.get('client_phone', '')
            metadata = request.data.get('metadata', {})
            
            if not image_file or not client_name:
                return Response(
                    {'error': 'image and client_name are required'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Upload to Google Cloud Storage
            bucket_name = os.getenv('GCS_BUCKET_NAME', 'your-bucket-name')
            project_id = os.getenv('GCS_PROJECT_ID', 'your-project-id')
            
            storage_client = storage.Client(project=project_id)
            bucket = storage_client.bucket(bucket_name)
            
            blob_name = f"detections/{request.user.username}/{image_file.name}"
            blob = bucket.blob(blob_name)
            blob.upload_from_string(
                image_file.read(),
                content_type=image_file.content_type
            )
            
            gcs_url = f"https://storage.googleapis.com/{bucket_name}/{blob_name}"
            
            # Save record to database
            record = DetectionRecord.objects.create(
                agent=request.user,
                client_name=client_name,
                client_email=client_email,
                client_phone=client_phone,
                image_url=gcs_url,
                original_filename=image_file.name,
                metadata=metadata
            )
            
            serializer = self.get_serializer(record)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=False, methods=['get'])
    def my_records(self, request):
        """Get all detection records for logged-in agent."""
        records = self.get_queryset()
        serializer = self.get_serializer(records, many=True)
        return Response(serializer.data)
