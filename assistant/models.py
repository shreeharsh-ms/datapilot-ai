
from mongoengine import Document, IntField,     StringField, ListField, DictField, DateTimeField, BooleanField, ReferenceField
from datetime import datetime
import uuid

class DataCleaningOperation(Document):
    """Model to store data cleaning operations"""
    meta = {'collection': 'data_cleaning_operations'}
    
    # Basic info
    name = StringField(required=True)
    description = StringField()
    operation_type = StringField(required=True)  # missing, duplicates, standardize, convert
    status = StringField(default='pending')  # pending, completed, failed
    
    # Dataset references
    input_dataset_id = StringField(required=True)
    output_dataset_id = StringField()
    
    # Operation parameters
    parameters = DictField()
    
    # User and timing
    user_id = StringField(required=True)
    created_at = DateTimeField(default=datetime.utcnow)
    completed_at = DateTimeField()
    
    # Results
    result_stats = DictField()
    error_message = StringField()

class DataCleaningTemplate(Document):
    """Model to store reusable data cleaning templates"""
    meta = {'collection': 'data_cleaning_templates'}
    
    name = StringField(required=True)
    description = StringField()
    template_type = StringField(required=True)
    parameters = DictField(required=True)
    user_id = StringField(required=True)
    is_public = BooleanField(default=False)
    created_at = DateTimeField(default=datetime.utcnow)
    usage_count = IntField(default=0)

class ChatLog(Document):
    user_id = StringField(required=True)
    dataset_id = StringField()
    messages = DictField()
    created_at = DateTimeField()

class Workspace(Document):
    name = StringField(required=True, max_length=100)
    description = StringField(max_length=500)
    owner_id = StringField(required=True)
    members = ListField(StringField())
    pinned = BooleanField(default=False)
    color = StringField(default="yellow")  # yellow, blue, green, purple, red
    dataset_count = IntField(default=0)
    created_at = DateTimeField(default=datetime.utcnow)
    updated_at = DateTimeField(default=datetime.utcnow)
    
    meta = {
        'collection': 'workspaces',
        'indexes': [
            'owner_id',
            'pinned',
            'created_at'
        ]
    }

class WorkspaceActivity(Document):
    workspace_id = StringField(required=True)
    user_id = StringField(required=True)
    user_name = StringField(default="User")
    action = StringField(required=True)  # 'import', 'share', 'transform', 'create'
    description = StringField(required=True)
    details = DictField()
    created_at = DateTimeField(default=datetime.utcnow)
    
    meta = {
        'collection': 'workspace_activities',
        'indexes': [
            'workspace_id',
            'created_at'
        ]
    }