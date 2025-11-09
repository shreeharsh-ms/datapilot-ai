from django.urls import path
from . import views

urlpatterns = [

    # ----------------------
    # UI PAGES
    # ----------------------
    path('dashboard/', views.dashboard, name='assistant_dashboard'),
    path('workspace/', views.workspace, name='assistant_workspace'),

    # Workspace sub-pages
    path('workspace/table-view/', views.table_view, name='assistant_table_view'),
    path('workspace/transformation/', views.transformation, name='assistant_transformation'),
    path('workspace/schema/', views.schema_page, name='assistant_schema'),

    # ----------------------
    # WORKSPACE API
    # ----------------------
    path('api/workspace/create/', views.create_workspace, name='create_workspace'),
    path('api/workspace/<str:workspace_id>/pin/', views.toggle_pin_workspace, name='toggle_pin_workspace'),
    path('api/workspace/<str:workspace_id>/edit/', views.edit_workspace, name='edit_workspace'),
    path('api/workspace/<str:workspace_id>/delete/', views.delete_workspace, name='delete_workspace'),
    path('api/workspace/activities/', views.get_workspace_activities, name='get_workspace_activities'),

    # ----------------------
    # DATASET API
    # ----------------------
    path('api/dataset/<str:dataset_id>/preview/', views.get_dataset_preview, name='get_dataset_preview'),
    path('api/dataset/<str:dataset_id>/columns/', views.get_dataset_columns, name='get_dataset_columns'),

    # ----------------------
    # JOIN OPERATIONS API
    # ----------------------
    path('api/join/create/', views.create_join_operation, name='create_join_operation'),
    path('api/join/preview/', views.preview_join_operation, name='preview_join_operation'),
  path('api/cleaning/missing-values/', views.handle_missing_values, name='handle_missing_values'),
    path('api/cleaning/remove-duplicates/', views.remove_duplicates, name='remove_duplicates'),
    path('api/cleaning/standardize-formats/', views.standardize_formats, name='standardize_formats'),
    path('api/cleaning/convert-data-types/', views.convert_data_types, name='convert_data_types'),
    path('api/cleaning/preview/', views.preview_cleaning_operation, name='preview_cleaning_operation'),
    path('api/cleaning/execute/', views.execute_cleaning_operation, name='execute_cleaning_operation'),
    path('api/cleaning/history/', views.get_cleaning_history, name='get_cleaning_history'),
    path('api/cleaning/templates/', views.manage_cleaning_templates, name='manage_cleaning_templates'),
]
