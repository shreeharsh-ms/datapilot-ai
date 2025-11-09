# datapilot_ai/urls.py

from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/users/', include('users.urls')),
    path("datasets/", include("datasets.urls")),
    path('api/assistant/', include('assistant.urls')),
]
