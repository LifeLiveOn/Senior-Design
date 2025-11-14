from rest_framework import serializers
from .models import Customer, House, HouseImage, AgentCustomerLog

# this serializer file defines serializers for the models, meaning that it converts model instances to JSON and vice versa


class HouseImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = HouseImage
        fields = ["id", "house", "image_url", "uploaded_at"]
        read_only_fields = ["id", "uploaded_at"]


class HouseSerializer(serializers.ModelSerializer):
    images = HouseImageSerializer(many=True, read_only=True)

    class Meta:
        model = House
        fields = ["id", "customer", "address",
                  "description", "created_at", "images"]
        read_only_fields = ["id", "created_at", "images"]


class CustomerSerializer(serializers.ModelSerializer):
    houses = HouseSerializer(many=True, read_only=True)

    class Meta:
        model = Customer
        fields = ["id", "agent", "name", "email",
                  "phone", "created_at", "houses", "address"]
        read_only_fields = ["id", "created_at", "agent", "houses"]


class AgentCustomerLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = AgentCustomerLog
        fields = ["id", "agent", "customer", "action", "timestamp", "details"]
        read_only_fields = ["id", "timestamp"]
