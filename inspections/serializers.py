from rest_framework import serializers
from django.contrib.auth.models import User   #  add this
from .models import InspectionCase, ImageAsset


class CaseSerializer(serializers.ModelSerializer):
    class Meta:
        model = InspectionCase
        fields = ["id", "title", "created_at"]


class ImageSerializer(serializers.ModelSerializer):
    # expose a URL Streamlit can render
    url = serializers.SerializerMethodField()

    class Meta:
        model = ImageAsset
        fields = ["id", "url", "status", "uploaded_at"]

    def get_url(self, obj):
        request = self.context.get("request")
        if obj.file and hasattr(obj.file, "url"):
            return request.build_absolute_uri(obj.file.url) if request else obj.file.url
        return None


class UserCreateSerializer(serializers.ModelSerializer):
    # password not be readable in responses
    password = serializers.CharField(write_only=True)

    class Meta:
        model = User
        fields = ["id", "username", "password", "email"]

    def create(self, validated_data):
        password = validated_data.pop("password")
        # this hashes the password correctly
        user = User.objects.create_user(password=password, **validated_data)
        return user
