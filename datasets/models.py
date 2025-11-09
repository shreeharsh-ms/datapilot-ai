from mongoengine import Document, StringField, DateTimeField, FileField, DictField
from datetime import datetime

class Dataset(Document):
    owner_id = StringField(required=True)
    file_name = StringField(required=True)
    file_type = StringField(required=True)
    description = StringField(max_length=500)
    metadata = DictField(default=dict)
    uploaded_at = DateTimeField(default=datetime.utcnow)
    file_url = StringField(required=True)
    file_path = StringField(required=True)

    meta = {
        'collection': 'datasets',
        'indexes': [
            'owner_id',
            'uploaded_at'
        ]
    }

class DatasetAnalysis(Document):
    meta = {"collection": "dataset_analysis"}
    dataset_id = StringField(required=True)
    summary = DictField()       # stats: mean, nulls, etc.
    visuals = DictField()       # base64 charts
    created_at = DateTimeField(default=datetime.utcnow)
