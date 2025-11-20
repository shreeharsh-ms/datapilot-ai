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
    path('workspace/analyze/', views.analyze_data_page, name="assistant_analyze"),
    path('workspace/visualization/', views.visualization_page, name='assistant_visualization'),
    path('table/', views.assistant_table, name="assistant_table"),

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
    path('api/datasets/', views.get_user_datasets, name='get_user_datasets'),
    path('api/dataset/<str:dataset_id>/preview/', views.get_dataset_preview, name='get_dataset_preview'),
    path('api/dataset/<str:dataset_id>/columns/', views.get_dataset_columns, name='get_dataset_columns'),

    # ----------------------
    # DATA TRANSFORMATION API
    # ----------------------
    
    # Join Operations
    path('api/join/create/', views.create_join_operation, name='create_join_operation'),
    path('api/join/preview/', views.preview_join_operation, name='preview_join_operation'),
    
    # Data Cleaning
    path('api/cleaning/missing-values/', views.handle_missing_values, name='handle_missing_values'),
    path('api/cleaning/remove-duplicates/', views.remove_duplicates, name='remove_duplicates'),
    path('api/cleaning/standardize-formats/', views.standardize_formats, name='standardize_formats'),
    path('api/cleaning/convert-data-types/', views.convert_data_types, name='convert_data_types'),
    path('api/cleaning/preview/', views.preview_cleaning_operation, name='preview_cleaning_operation'),
    path('api/cleaning/execute/', views.execute_cleaning_operation, name='execute_cleaning_operation'),
    path('api/cleaning/history/', views.get_cleaning_history, name='get_cleaning_history'),
    path('api/cleaning/templates/', views.manage_cleaning_templates, name='manage_cleaning_templates'),
    
    # Aggregation
    path('api/aggregation/groupby/', views.groupby_aggregation, name='groupby_aggregation'),
    path('api/aggregation/pivot/', views.pivot_table, name='pivot_table'),
    path('api/aggregation/window/', views.window_functions, name='window_functions'),
    path('api/aggregation/rollup/', views.rollup_cube, name='rollup_cube'),
    path('api/aggregation/preview/', views.preview_aggregation, name='preview_aggregation'),
    path('api/aggregation/execute/', views.execute_aggregation, name='execute_aggregation'),
    path('api/aggregation/history/', views.get_aggregation_history, name='get_aggregation_history'),
    path('api/aggregation/templates/', views.manage_aggregation_templates, name='manage_aggregation_templates'),
    path('api/aggregation/statistics/', views.get_column_statistics, name='get_column_statistics'),
    
    # Filter & Sort
    path('api/filter-sort/filter/', views.filter_data, name='filter_data'),
    path('api/filter-sort/sort/', views.sort_data, name='sort_data'),
    path('api/filter-sort/top-n/', views.top_n_records, name='top_n_records'),
    path('api/filter-sort/random-sample/', views.random_sampling, name='random_sampling'),
    path('api/filter-sort/preview/', views.preview_filter_sort, name='preview_filter_sort'),
    path('api/filter-sort/execute/', views.execute_filter_sort, name='execute_filter_sort'),
    
    # Feature Engineering
    path('api/feature-engineering/calculated-columns/', views.create_calculated_columns, name='create_calculated_columns'),
    path('api/feature-engineering/datetime-extraction/', views.datetime_extraction, name='datetime_extraction'),
    path('api/feature-engineering/text-processing/', views.text_processing, name='text_processing'),
    path('api/feature-engineering/one-hot-encoding/', views.one_hot_encoding, name='one_hot_encoding'),
    path('api/feature-engineering/preview/', views.preview_feature_engineering, name='preview_feature_engineering'),
    path('api/feature-engineering/execute/', views.execute_feature_engineering, name='execute_feature_engineering'),
    
    # ML Preparation
    path('api/ml-preparation/train-test-split/', views.train_test_split, name='train_test_split'),
    path('api/ml-preparation/feature-scaling/', views.feature_scaling, name='feature_scaling'),
    path('api/ml-preparation/outlier-detection/', views.outlier_detection, name='outlier_detection'),
    path('api/ml-preparation/cross-validation/', views.cross_validation, name='cross_validation'),
    path('api/ml-preparation/preview/', views.preview_ml_preparation, name='preview_ml_preparation'),
    path('api/ml-preparation/execute/', views.execute_ml_preparation, name='execute_ml_preparation'),

    # ----------------------
    # PIPELINE API
    # ----------------------
    path('api/pipeline/', views.get_pipelines, name='get_pipelines'),
    path('api/pipeline/create/', views.create_pipeline, name='create_pipeline'),
    path('api/pipeline/<str:pipeline_id>/', views.get_pipeline_details, name='get_pipeline_details'),
    path('api/pipeline/<str:pipeline_id>/edit/', views.edit_pipeline, name='edit_pipeline'),
    path('api/pipeline/<str:pipeline_id>/delete/', views.delete_pipeline, name='delete_pipeline'),
    path('api/pipeline/<str:pipeline_id>/run/', views.run_pipeline, name='run_pipeline'),
    path('api/pipeline/<str:pipeline_id>/execute/', views.execute_full_pipeline, name='execute_pipeline'),
    
    # Pipeline Execution
    path('api/pipeline/execute/', views.execute_pipeline, name='execute_pipeline'),
    path('api/pipeline/step/execute/', views.execute_pipeline_step, name='execute_pipeline_step'),
    path('api/pipeline/preview/', views.preview_pipeline_step, name='preview_pipeline_step'),
    path('api/transform/preview/', views.preview_transformation_step, name='preview_transformation_step'),
    path("run-json-pipeline/", views.run_pipeline_from_json, name="run_json_pipeline"),
    
    # Pipeline Templates
    path('api/pipeline/templates/', views.manage_pipeline_templates, name='manage_pipeline_templates'),
    path('api/pipeline/templates/create-from-pipeline/', views.save_pipeline_as_template, name='save_pipeline_as_template'),
    path('api/pipeline/create-from-template/', views.create_pipeline_from_template, name='create_pipeline_from_template'),

    # ----------------------
    # VISUALIZATION API
    # ----------------------
    
    # Saved Visualizations (using the new SavedVisualization model)
    path('api/visualizations/save/', views.save_visualization, name='save_visualization'),
    path('api/visualizations/', views.get_saved_visualizations, name='get_visualizations'),
    path('api/visualizations/<str:viz_id>/', views.get_saved_visualization, name='get_visualization'),
    path('api/visualizations/<str:viz_id>/update/', views.update_saved_visualization, name='update_visualization'),
    path('api/visualizations/<str:viz_id>/delete/', views.delete_saved_visualization, name='delete_visualization'),
    path('api/visualizations/<str:viz_id>/duplicate/', views.duplicate_saved_visualization, name='duplicate_visualization'),
    path('api/visualizations/search/', views.search_saved_visualizations, name='search_visualizations'),
    
    # Visualization Generation & Templates
    path('api/visualization/generate/', views.generate_visualization, name='generate_visualization'),
    path('api/visualization/export/<str:viz_id>/', views.export_visualization, name='export_visualization'),
    path('api/visualization/templates/', views.manage_visualization_templates, name='manage_visualization_templates'),
    path('api/visualization/templates/<str:template_id>/', views.get_visualization_template, name='get_visualization_template'),

    # ----------------------
    # LEGACY/COMPATIBILITY ENDPOINTS
    # ----------------------
    
    # Legacy visualization endpoints (for backward compatibility)
    path('api/assistant/api/visualization/generate/', views.generate_visualization, name='generate_visualization_legacy'),
    path('api/assistant/api/visualization/save/', views.save_visualization, name='save_visualization_legacy'),
    path('api/assistant/api/visualization/list/', views.get_user_visualizations, name='get_user_visualizations_legacy'),
    path('api/assistant/api/visualization/<str:viz_id>/', views.get_visualization, name='get_visualization_legacy'),
    path('api/assistant/api/visualization/<str:viz_id>/update/', views.update_visualization, name='update_visualization_legacy'),
    path('api/assistant/api/visualization/<str:viz_id>/delete/', views.delete_visualization, name='delete_visualization_legacy'),
    
    # Legacy pipeline endpoints
    path('api/pipelines/', views.get_user_pipelines, name='get_user_pipelines_legacy'),

    # ----------------------
    # STATS & ANALYTICS PAGES
    # ----------------------
    path('workspace/stats/', views.stats_display, name='assistant_stats'),
    
    # History and Notes Pages
    path('workspace/history/', views.history_view, name='history'),
    path('workspace/notes/', views.notes_view, name='notes'),
    
    # Stats API endpoints
    path('api/assistant/api/stats/history/', views.history_api, name='history_api'),
    path('api/assistant/api/stats/notes/', views.notes_api, name='notes_api'),
    

]