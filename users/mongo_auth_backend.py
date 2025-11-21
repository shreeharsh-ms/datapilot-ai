from django.contrib.auth.backends import BaseBackend
from .models import CustomUser

class MongoAuthBackend(BaseBackend):
    def authenticate(self, request, username=None, password=None):
        user = CustomUser.objects(username=username).first()
        if user and user.check_password(password):
            return user
        return None

    def get_user(self, user_id):
        return CustomUser.objects(id=user_id).first()
