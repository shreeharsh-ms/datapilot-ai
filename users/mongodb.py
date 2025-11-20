from pymongo import MongoClient
from django.conf import settings
import json
from bson import ObjectId
import pandas as pd

class MongoDBManager:
    def __init__(self):
        self.client = None
        self.db = None
        self.connect()
    
    def connect(self):
        """Connect to MongoDB"""
        try:
            self.client = MongoClient(settings.MONGO_URL)
            self.db = self.client[settings.MONGO_DB]
            print("Connected to MongoDB successfully")
        except Exception as e:
            print(f"Error connecting to MongoDB: {e}")
    
    def create_user(self, user_data):
        """Create user in MongoDB"""
        try:
            users_collection = self.db.users
            result = users_collection.insert_one(user_data)
            return str(result.inserted_id)
        except Exception as e:
            print(f"Error creating user in MongoDB: {e}")
            return None
    
    def create_dataset(self, dataset_data):
        """Store dataset in MongoDB"""
        try:
            datasets_collection = self.db.datasets
            result = datasets_collection.insert_one(dataset_data)
            return str(result.inserted_id)
        except Exception as e:
            print(f"Error creating dataset in MongoDB: {e}")
            return None
    
    def store_analysis_results(self, analysis_data):
        """Store analysis results in MongoDB"""
        try:
            analyses_collection = self.db.analyses
            result = analyses_collection.insert_one(analysis_data)
            return str(result.inserted_id)
        except Exception as e:
            print(f"Error storing analysis in MongoDB: {e}")
            return None
    
    def get_user_datasets(self, user_id):
        """Get all datasets for a user"""
        try:
            datasets_collection = self.db.datasets
            datasets = list(datasets_collection.find({"user_id": user_id}))
            return datasets
        except Exception as e:
            print(f"Error getting user datasets: {e}")
            return []
    
    def get_analysis_results(self, analysis_id):
        """Get analysis results from MongoDB"""
        try:
            analyses_collection = self.db.analyses
            analysis = analyses_collection.find_one({"_id": ObjectId(analysis_id)})
            return analysis
        except Exception as e:
            print(f"Error getting analysis results: {e}")
            return None

# MongoDB manager instance
mongo_manager = MongoDBManager()