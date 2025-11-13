from django.contrib import admin
from django.utils.html import format_html

from .models import InspectionCase, ImageAsset


class ImageAssetInline(admin.TabularInline):
    """
    Show images inline on each InspectionCase page.
    """
    model = ImageAsset
    extra = 0
    readonly_fields = ("preview", "status", "uploaded_at")
    fields = ("preview", "file", "status", "uploaded_at")

    @admin.display(description="Preview")
    def preview(self, obj):
        if obj.file and hasattr(obj.file, "url"):
            return format_html(
                '<img src="{}" style="max-height: 100px; max-width: 150px;" />',
                obj.file.url,
            )
        return "-"


@admin.register(InspectionCase)
class InspectionCaseAdmin(admin.ModelAdmin):
    """
    “Client house” – this is what you and your sponsor care about.
    """
    list_display = ("id", "title", "created_by", "created_at", "image_count")
    search_fields = ("title", "created_by__username")
    list_filter = ("created_at",)
    inlines = [ImageAssetInline]

    @admin.display(description="# Images")
    def image_count(self, obj):
        return obj.images.count()


@admin.register(ImageAsset)
class ImageAssetAdmin(admin.ModelAdmin):
    """
    Fallback view for images if you ever want to see them separately.
    """
    list_display = ("id", "case", "status", "uploaded_at")
    list_filter = ("status", "uploaded_at")
    search_fields = ("case__title",)
