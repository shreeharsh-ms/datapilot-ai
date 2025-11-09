from django.urls import path
from . import views

urlpatterns = [
    path("upload/", views.upload_dataset, name="upload_dataset"),
    path("", views.dashboard, name="dashboard"),
    path("delete/<str:dataset_id>/", views.delete_dataset, name="delete_dataset"),
    path("signed-url/<str:dataset_id>/", views.get_signed_url, name="get_signed_url"),
   path('<str:dataset_id>/preview/', views.preview_dataset, name='preview_dataset'),
]

