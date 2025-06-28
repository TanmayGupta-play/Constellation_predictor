from django.contrib import admin
from .models import LocationData, ConstellationQuery

@admin.register(LocationData)
class LocationDataAdmin(admin.ModelAdmin):
    list_display = ['user', 'latitude', 'longitude', 'created_at']
    list_filter = ['created_at', 'user']
    readonly_fields = ['created_at']
    search_fields = ['user__username']

@admin.register(ConstellationQuery)
class ConstellationQueryAdmin(admin.ModelAdmin):
    list_display = ['location', 'created_at', 'query_text_preview']
    list_filter = ['created_at']
    readonly_fields = ['created_at']
    search_fields = ['query_text', 'location__user__username']
    
    def query_text_preview(self, obj):
        return obj.query_text[:50] + "..." if len(obj.query_text) > 50 else obj.query_text
    query_text_preview.short_description = "Query Preview"