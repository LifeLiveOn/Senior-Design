from rest_framework import serializers
from .models import DetectionRecord

class DetectionRecordSerializer(serializers.ModelSerializer):
    agent_name = serializers.CharField(source='agent.username', read_only=True)
    
    class Meta:
        model = DetectionRecord
        fields = ['id', 'agent', 'agent_name', 'client_name', 'client_email', 
                  'client_phone', 'image_url', 'original_filename', 'uploaded_at', 'metadata']
        read_only_fields = ['agent', 'image_url', 'uploaded_at']
