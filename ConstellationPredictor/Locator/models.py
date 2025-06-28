# models.py
from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

class LocationData(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    latitude = models.FloatField()
    longitude = models.FloatField()
    accuracy = models.FloatField(default=0) 
    created_at = models.DateTimeField(default=timezone.now)
    
    class Meta:
        ordering = ['-created_at']
        
    def __str__(self):
        return f"Location ({self.latitude}, {self.longitude}) - {self.created_at}"

class ConstellationQuery(models.Model):
    location = models.ForeignKey(LocationData, on_delete=models.CASCADE)
    query_text = models.TextField() 
    gemini_response = models.TextField()  
    visible_constellations = models.JSONField(default=list) 
    created_at = models.DateTimeField(default=timezone.now)
    
    class Meta:
        ordering = ['-created_at']
        
    def __str__(self):
        return f"Query for {self.location} - {self.created_at}"