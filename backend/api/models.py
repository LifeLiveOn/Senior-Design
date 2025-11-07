from django.db import models
from django.contrib.auth.models import User

class DetectionRecord(models.Model):
    agent = models.ForeignKey(User, on_delete=models.CASCADE)
    client_name = models.CharField(max_length=255)
    client_email = models.EmailField(blank=True, null=True)
    client_phone = models.CharField(max_length=20, blank=True, null=True)
    image_url = models.URLField()  # GCS URL
    original_filename = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    metadata = models.JSONField(default=dict, blank=True)  # Store inference results, timestamps, etc.
    
    class Meta:
        ordering = ['-uploaded_at']
    
    def __str__(self):
        return f"{self.client_name} - {self.uploaded_at}"
