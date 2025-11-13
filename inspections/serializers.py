from rest_framework import serializers
from .models import InspectionCase, ImageAsset

class CaseSerializer(serializers.ModelSerializer):
    class Meta:
        model = InspectionCase
        fields = ["id", "title", "created_at"]

class ImageSerializer(serializers.ModelSerializer):
    # expose a URL the FE/Streamlit can render
    url = serializers.SerializerMethodField()

    class Meta:
        model = ImageAsset
        fields = ["id", "url", "status", "uploaded_at"]

    def get_url(self, obj):
        request = self.context.get("request")
        if obj.file and hasattr(obj.file, "url"):
            return request.build_absolute_uri(obj.file.url) if request else obj.file.url
        return None
