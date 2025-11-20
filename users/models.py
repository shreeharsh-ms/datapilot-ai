from django.db import models
from django.contrib.auth.models import AbstractUser
from django.conf import settings
import uuid
from datetime import datetime

# Custom User model for MongoDB integration
class User(AbstractUser):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    email = models.EmailField(unique=True)
    company = models.CharField(max_length=255, blank=True, null=True)
    avatar = models.URLField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    # MongoDB specific fields
    mongo_id = models.CharField(max_length=100, blank=True, null=True)
    
    def __str__(self):
        return self.username

class UserProfile(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    bio = models.TextField(blank=True, null=True)
    location = models.CharField(max_length=255, blank=True, null=True)
    website = models.URLField(blank=True, null=True)
    social_links = models.JSONField(default=dict, blank=True)
    
    # Analytics preferences
    default_chart_type = models.CharField(max_length=50, default='line')
    theme_preference = models.CharField(max_length=20, default='light')
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

class Dataset(models.Model):
    DATASET_TYPES = (
        ('csv', 'CSV'),
        ('excel', 'Excel'),
        ('json', 'JSON'),
        ('sql', 'SQL Database'),
        ('api', 'API'),
    )
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, null=True)
    file = models.FileField(upload_to='datasets/')
    file_type = models.CharField(max_length=10, choices=DATASET_TYPES)
    file_size = models.BigIntegerField(default=0)
    
    # MongoDB document reference
    mongo_collection = models.CharField(max_length=255, blank=True, null=True)
    
    # Dataset statistics
    row_count = models.IntegerField(default=0)
    column_count = models.IntegerField(default=0)
    missing_values = models.IntegerField(default=0)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return self.name

class Analysis(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, null=True)
    
    # Analysis configuration
    filters = models.JSONField(default=dict)
    selected_columns = models.JSONField(default=list)
    transformations = models.JSONField(default=list)
    
    # MongoDB analysis results
    mongo_analysis_id = models.CharField(max_length=100, blank=True, null=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name_plural = "Analyses"
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.name} - {self.dataset.name}"

class Insight(models.Model):
    INSIGHT_TYPES = (
        ('trend', 'Trend'),
        ('anomaly', 'Anomaly'),
        ('correlation', 'Correlation'),
        ('outlier', 'Outlier'),
        ('pattern', 'Pattern'),
        ('summary', 'Summary'),
    )
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    analysis = models.ForeignKey(Analysis, on_delete=models.CASCADE)
    insight_type = models.CharField(max_length=20, choices=INSIGHT_TYPES)
    title = models.CharField(max_length=255)
    description = models.TextField()
    confidence = models.FloatField(default=0.0)  # 0.0 to 1.0
    data = models.JSONField(default=dict)  # Raw insight data
    
    # MongoDB reference
    mongo_insight_id = models.CharField(max_length=100, blank=True, null=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-confidence', '-created_at']
    
    def __str__(self):
        return f"{self.insight_type}: {self.title}"