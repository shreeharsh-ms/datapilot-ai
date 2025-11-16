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
        path('api/aggregation/groupby/', views.groupby_aggregation, name='groupby_aggregation'),
    path('api/aggregation/pivot/', views.pivot_table, name='pivot_table'),
    path('api/aggregation/window/', views.window_functions, name='window_functions'),
    path('api/aggregation/rollup/', views.rollup_cube, name='rollup_cube'),
    path('api/aggregation/preview/', views.preview_aggregation, name='preview_aggregation'),
    path('api/aggregation/execute/', views.execute_aggregation, name='execute_aggregation'),
    path('api/aggregation/history/', views.get_aggregation_history, name='get_aggregation_history'),
    path('api/aggregation/templates/', views.manage_aggregation_templates, name='manage_aggregation_templates'),
    path('api/aggregation/statistics/', views.get_column_statistics, name='get_column_statistics'),
    # Filter & Sort URLs
path('api/filter-sort/filter/', views.filter_data, name='filter_data'),
path('api/filter-sort/sort/', views.sort_data, name='sort_data'),
path('api/filter-sort/top-n/', views.top_n_records, name='top_n_records'),
path('api/filter-sort/random-sample/', views.random_sampling, name='random_sampling'),
path('api/filter-sort/preview/', views.preview_filter_sort, name='preview_filter_sort'),
path('api/filter-sort/execute/', views.execute_filter_sort, name='execute_filter_sort'),
# Feature Engineering URLs
path('api/feature-engineering/calculated-columns/', views.create_calculated_columns, name='create_calculated_columns'),
path('api/feature-engineering/datetime-extraction/', views.datetime_extraction, name='datetime_extraction'),
path('api/feature-engineering/text-processing/', views.text_processing, name='text_processing'),
path('api/feature-engineering/one-hot-encoding/', views.one_hot_encoding, name='one_hot_encoding'),
path('api/feature-engineering/preview/', views.preview_feature_engineering, name='preview_feature_engineering'),
path('api/feature-engineering/execute/', views.execute_feature_engineering, name='execute_feature_engineering'),
# ML Preparation URLs
path('api/ml-preparation/train-test-split/', views.train_test_split, name='train_test_split'),
path('api/ml-preparation/feature-scaling/', views.feature_scaling, name='feature_scaling'),
path('api/ml-preparation/outlier-detection/', views.outlier_detection, name='outlier_detection'),
path('api/ml-preparation/cross-validation/', views.cross_validation, name='cross_validation'),
path('api/ml-preparation/preview/', views.preview_ml_preparation, name='preview_ml_preparation'),
path('api/ml-preparation/execute/', views.execute_ml_preparation, name='execute_ml_preparation'),
    path('api/pipeline/create/', views.create_pipeline, name='create_pipeline'),

    path('api/pipeline/<str:pipeline_id>/execute/', views.execute_full_pipeline, name='execute_pipeline'),
    path('api/pipeline/step/execute/', views.execute_pipeline_step, name='execute_pipeline_step'),
    path('api/pipeline/preview/', views.preview_pipeline_step, name='preview_pipeline_step'),
    path('api/pipeline/<str:pipeline_id>/', views.get_pipeline_details, name='get_pipeline_details'),
    path('api/pipelines/', views.get_user_pipelines, name='get_user_pipelines'),
    
    # Pipeline Templates
    path('api/pipeline/templates/', views.manage_pipeline_templates, name='manage_pipeline_templates'),
    # Add these to your existing urlpatterns
path('api/pipeline/templates/create-from-pipeline/', views.save_pipeline_as_template, name='save_pipeline_as_template'),
path('api/pipeline/create-from-template/', views.create_pipeline_from_template, name='create_pipeline_from_template'),
path('api/datasets/', views.get_user_datasets, name='get_user_datasets'),
# Add to your existing urlpatterns
path('api/transform/preview/', views.preview_transformation_step, name='preview_transformation_step'),
path('api/pipeline/execute/', views.execute_pipeline, name='execute_pipeline'),

# get all pipelines for a user
path('api/pipeline/', views.get_pipelines, name='get_pipelines'),
 path('api/pipeline/<str:pipeline_id>/run/', views.run_pipeline, name='run_pipeline'),
    path('api/pipeline/<str:pipeline_id>/edit/', views.edit_pipeline, name='edit_pipeline'),
    path('api/pipeline/<str:pipeline_id>/delete/', views.delete_pipeline, name='delete_pipeline'),


path("run-json-pipeline/", views.run_pipeline_from_json, name="run_json_pipeline"),


# Visualization UI Page
path("workspace/analyze/", views.analyze_data_page, name="assistant_analyze"),

]
