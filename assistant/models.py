# models.py
from mongoengine import Document, StringField, ListField, DictField, DateTimeField, ObjectIdField,IntField,FloatField,BooleanField, ReferenceField
from datetime import datetime
from bson import ObjectId


class TransformationPipeline(Document):
    """Transformation pipeline definition"""
    name = StringField(required=True, max_length=200)
    description = StringField(default='')
    owner_id = StringField(required=True)
    input_dataset_id = ObjectIdField(required=True)
    output_dataset_id = ObjectIdField()
    steps = ListField(DictField(), default=list)
    total_steps = IntField(default=0)
    current_step = IntField(default=0)
    status = StringField(default='draft', choices=['draft', 'running', 'completed', 'failed'])
    execution_stats = DictField(default=dict)
    created_from_template = StringField()
    
    # Execution tracking fields
    execution_results = ListField(DictField(), default=list)  # Store step execution results
    final_dataset_id = ObjectIdField()  # Final output dataset
    last_executed = DateTimeField()  # Last execution time
    execution_count = IntField(default=0)  # Number of times executed
    last_error = StringField()  # Last error message if failed
    
    # Workspace support
    workspace_id = StringField(default='default-workspace')
    
    # Timestamps
    created_at = DateTimeField(default=datetime.utcnow)
    updated_at = DateTimeField(default=datetime.utcnow)
    completed_at = DateTimeField()
    
    meta = {
        'collection': 'transformation_pipelines',
        'indexes': [
            'owner_id',
            'status',
            'created_at',
            'workspace_id',
            'last_executed'
        ]
    }
    
    def clean(self):
        """Update timestamps before saving"""
        self.updated_at = datetime.utcnow()
        if self.status == 'completed' and not self.completed_at:
            self.completed_at = datetime.utcnow()
    
    def get_execution_summary(self):
        """Get summary of pipeline execution"""
        successful_steps = len([r for r in self.execution_results if r.get('success', False)])
        failed_steps = len(self.execution_results) - successful_steps
        
        return {
            'total_steps': self.total_steps,
            'executed_steps': len(self.execution_results),
            'successful_steps': successful_steps,
            'failed_steps': failed_steps,
            'completion_rate': (successful_steps / self.total_steps * 100) if self.total_steps > 0 else 0,
            'last_execution': self.last_executed.isoformat() if self.last_executed else None,
            'execution_count': self.execution_count
        }
class DataSet(Document):
    """Dataset model for storing uploaded data"""
    meta = {'collection': 'datasets'}
    
    # Basic info
    name = StringField(required=True, max_length=200)
    file_name = StringField(required=True)
    description = StringField(default='')
    
    # File information
    file_path = StringField()
    file_size = IntField(default=0)
    file_type = StringField(default='csv')  # csv, excel, json
    
    # Data information
    row_count = IntField(default=0)
    column_count = IntField(default=0)
    columns = ListField(StringField(), default=list)
    data = DictField()  # Store sample data or full data
    
    # User and workspace
    user_id = StringField(required=True)
    workspace_id = StringField(default='default-workspace')
    
    # Status
    status = StringField(default='active')  # active, processing, archived
    
    # Timestamps
    created_at = DateTimeField(default=datetime.utcnow)
    updated_at = DateTimeField(default=datetime.utcnow)
    
    def clean(self):
        """Update timestamps before saving"""
        self.updated_at = datetime.utcnow()

class PipelineStep(Document):
    """Individual step in a transformation pipeline"""
    pipeline_id = ObjectIdField(required=True)
    step_number = IntField(required=True)
    step_type = StringField(required=True)
    operation = StringField(required=True)
    parameters = DictField(default=dict)
    status = StringField(default='pending')
    output_dataset_id = ObjectIdField()
    result_stats = DictField(default=dict)
    preview_data = ListField(default=list)
    preview_columns = ListField(default=list)
    execution_time = FloatField(default=0.0)
    error_message = StringField()
    executed_at = DateTimeField()
    
    meta = {
        'collection': 'pipeline_steps',
        'indexes': [
            'pipeline_id',
            'step_number',
            'status'
        ]
    }


class AggregationOperation(Document):
    """Model to store aggregation operations"""
    meta = {'collection': 'aggregation_operations'}
    
    # Basic info
    name = StringField(required=True)
    description = StringField()
    operation_type = StringField(required=True)  # groupby, pivot, window, rollup
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

class AggregationTemplate(Document):
    """Model to store reusable aggregation templates"""
    meta = {'collection': 'aggregation_templates'}
    
    name = StringField(required=True)
    description = StringField()
    template_type = StringField(required=True)
    parameters = DictField(required=True)
    user_id = StringField(required=True)
    is_public = BooleanField(default=False)
    created_at = DateTimeField(default=datetime.utcnow)
    usage_count = IntField(default=0)
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
    """Track user activities in workspaces"""
    workspace_id = StringField(required=True)  # Add this required field
    user_id = StringField(required=True)
    user_name = StringField(required=True)
    action = StringField(required=True)  # create, update, delete, execute, etc.
    description = StringField(required=True)
    details = DictField(default=dict)
    created_at = DateTimeField(default=datetime.utcnow)
    
    meta = {
        'collection': 'workspace_activities',
        'indexes': [
            'workspace_id',
            'user_id',
            'action',
            'created_at'
        ]
    }
class AggregationOperation(Document):
    """Aggregation operation record"""
    name = StringField(required=True, max_length=200)
    operation_type = StringField(required=True)  # groupby, pivot, window, rollup
    input_dataset_id = StringField(required=True)
    output_dataset_id = StringField(required=True)
    user_id = StringField(required=True)
    parameters = DictField(default=dict)
    result_stats = DictField(default=dict)
    status = StringField(default='pending')  # pending, running, completed, failed
    created_at = DateTimeField(default=datetime.utcnow)
    completed_at = DateTimeField()
    
    meta = {
        'collection': 'aggregation_operations',
        'indexes': [
            'user_id',
            'operation_type',
            'created_at'
        ]
    }

class AggregationTemplate(Document):
    """Template for reusable aggregation operations"""
    name = StringField(required=True, max_length=200)
    description = StringField(default='')
    template_type = StringField(required=True)  # groupby, pivot, etc.
    parameters = DictField(default=dict)
    user_id = StringField(required=True)
    is_public = BooleanField(default=False)
    usage_count = IntField(default=0)
    created_at = DateTimeField(default=datetime.utcnow)
    updated_at = DateTimeField(default=datetime.utcnow)
    
    meta = {
        'collection': 'aggregation_templates',
        'indexes': [
            'user_id',
            'template_type',
            'is_public'
        ]
    }
    
    def save(self, *args, **kwargs):
        self.updated_at = datetime.utcnow()
        return super().save(*args, **kwargs)