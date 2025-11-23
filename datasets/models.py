from mongoengine import Document, StringField, DateTimeField, DictField, DynamicField
from datetime import datetime
import uuid
import secrets


# -----------------------------
# Main Dataset Model
# -----------------------------
class Dataset(Document):
    owner_id = StringField(required=True)
    uploaded_at = DateTimeField(default=datetime.utcnow)

    name = StringField(required=True)
    source_type = StringField(required=True, default="file")  # file / mongodb / postgresql / mysql / api

    connection_info = DictField(default=dict)  # DB/API details
    file_info = DictField(default=dict)        # uploaded file info

    description = StringField(max_length=500, default="")
    metadata = DictField(default=dict)

    # legacy fields
    file_name = StringField()
    file_type = StringField()
    file_url = StringField()
    file_path = StringField()

    meta = {
        "collection": "datasets",
        "indexes": ["owner_id", "uploaded_at", "source_type"]
    }

    def clean(self):
        # Backward compatibility
        if self.file_name and not self.name:
            self.name = self.file_name
        if self.file_path and not self.file_info.get("path"):
            self.file_info["path"] = self.file_path
        if self.file_url and not self.file_info.get("url"):
            self.file_info["url"] = self.file_url
        if self.file_type and not self.file_info.get("content_type"):
            self.file_info["content_type"] = self.file_type

        # Auto-generate endpoint + token for API dataset
        if self.is_api_based:
            if not self.connection_info.get("endpoint_id"):
                self.connection_info["endpoint_id"] = str(uuid.uuid4())
            if not self.connection_info.get("token"):
                self.connection_info["token"] = secrets.token_hex(16)
            if "strict_schema" not in self.connection_info:
                self.connection_info["strict_schema"] = False

    @property
    def display_name(self):
        return (
            self.name or
            self.file_name or
            self.file_info.get("original_name") or
            "Unnamed Dataset"
        )

    @property
    def file_path_display(self):
        return self.file_info.get("path") or self.file_path

    @property
    def file_url_display(self):
        return self.file_info.get("url") or self.file_url

    @property
    def is_file_based(self):
        return self.source_type == "file"

    @property
    def is_database_based(self):
        return self.source_type in ["mongodb", "postgresql", "mysql"]

    @property
    def is_api_based(self):
        return self.source_type == "api"

    def get_connection_type_display(self):
        return {
            "file": "File Upload",
            "mongodb": "MongoDB",
            "postgresql": "PostgreSQL",
            "mysql": "MySQL",
            "api": "API (Webhook)"
        }.get(self.source_type, self.source_type)

    def get_api_endpoint(self, request=None):
        """Return full push URL"""
        if not self.is_api_based:
            return None
        endpoint_id = self.connection_info.get("endpoint_id")
        if request:
            return request.build_absolute_uri(f"/api/incoming/{endpoint_id}/")
        return f"/api/incoming/{endpoint_id}/"


# -----------------------------
# Dataset Analysis / Summary
# -----------------------------
class DatasetAnalysis(Document):
    dataset_id = StringField(required=True)
    summary = DictField(default=dict)
    visuals = DictField(default=dict)
    created_at = DateTimeField(default=datetime.utcnow)

    meta = {
        "collection": "dataset_analysis",
        "indexes": ["dataset_id", "created_at"]
    }


# -----------------------------
# Incoming API Data Storage
# -----------------------------
class DatasetIncomingData(Document):
    dataset_id = StringField(required=True)
    user_id = StringField(required=True)
    data = DynamicField()  # supports dict OR list (array of JSON)
    received_at = DateTimeField(default=datetime.utcnow)

    meta = {
        "collection": "dataset_incoming_data",
        "indexes": ["dataset_id", "user_id", "received_at"]
    }
