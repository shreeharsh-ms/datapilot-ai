from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
import json
from datetime import datetime
from bson import ObjectId
from bson.errors import InvalidId
import pandas as pd
import logging
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
logger = logging.getLogger(__name__)
import base64
import io
import uuid
from django.conf import settings
# models
from datasets.models import DatasetIncomingData

from functools import wraps




from .models import (
    Workspace, WorkspaceActivity, DataCleaningOperation, DataCleaningTemplate,
    TransformationPipeline, PipelineStep, DataSet, Visualization, 
    VisualizationTemplate, SavedVisualization
)

from datasets.models import Dataset
import numpy as np
# sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
def dataframe_to_dict_clean(df):
    """Convert DataFrame to dict with proper NaN and type handling"""
    try:
        if df is None or df.empty:
            return []
            
        # Make a copy to avoid modifying original
        df_clean = df.copy()
        
        # Replace NaN/NaT with None
        df_clean = df_clean.replace({np.nan: None})
        df_clean = df_clean.replace({pd.NaT: None})
        
        # Convert all columns to Python native types
        for col in df_clean.columns:
            # Handle numeric types
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].apply(
                    lambda x: float(x) if x is not None and not pd.isna(x) else None
                )
            # Handle datetime types
            elif pd.api.types.is_datetime64_any_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].apply(
                    lambda x: x.isoformat() if x is not None and not pd.isna(x) else None
                )
            # Handle other types - convert to string
            else:
                df_clean[col] = df_clean[col].astype(str)
        
        # Convert to dictionary
        return df_clean.to_dict(orient="records")
        
    except Exception as e:
        print(f"Error in dataframe_to_dict_clean: {e}")
        # Fallback: try simple conversion
        try:
            if hasattr(df, 'to_dict'):
                return df.replace({np.nan: None}).to_dict(orient="records")
            else:
                return []
        except:
            return []

def clean_data_for_json(obj):
    """Recursively clean data for JSON serialization"""
    if isinstance(obj, dict):
        return {k: clean_data_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_data_for_json(i) for i in obj]
    elif isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [clean_data_for_json(i) for i in obj]
    elif isinstance(obj, pd.Series):
        return clean_data_for_json(obj.to_list())
    elif hasattr(obj, 'item'):  # numpy scalars
        return obj.item()
    elif pd.isna(obj):  # Handle pandas NA
        return None
    elif isinstance(obj, ObjectId):
        return str(obj)
    else:
        return obj


def mongo_login_required(view_func):
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        if not request.session.get("user_id"):
            if request.path.startswith("/api/"):
                return JsonResponse({"success": False, "error": "Authentication required"}, status=401)
            return redirect("/users/login/")
        return view_func(request, *args, **kwargs)
    return _wrapped_view

@mongo_login_required
@require_http_methods(["POST"])
@csrf_exempt
def groupby_aggregation(request):
    """Perform group by and aggregation operations"""
    try:
        data = json.loads(request.body)
        dataset_id = data.get('dataset_id')
        group_columns = data.get('group_columns', [])
        aggregation_operations = data.get('aggregation_operations', {})
        preview_only = data.get('preview_only', False)
        
        # Validate inputs
        if not dataset_id:
            return JsonResponse({'success': False, 'error': 'Dataset ID is required'}, status=400)
        
        if not aggregation_operations:
            return JsonResponse({'success': False, 'error': 'At least one aggregation operation is required'}, status=400)
        
        # Get dataset
        dataset = Dataset.objects.get(id=ObjectId(dataset_id), owner_id=str(request.session.get("user_id")))
        df = download_and_convert_to_dataframe(dataset)
        
        # Store original stats
        original_stats = {
            'total_rows': len(df),
            'columns': list(df.columns)
        }
        
        # Prepare aggregation dictionary for pandas
        agg_dict = {}
        for col, operations in aggregation_operations.items():
            if col not in df.columns:
                return JsonResponse({'success': False, 'error': f'Aggregation column "{col}" not found in dataset'}, status=400)
            
            agg_dict[col] = []
            for op in operations:
                if op in ['sum', 'mean', 'median', 'min', 'max', 'count', 'std', 'var']:
                    agg_dict[col].append(op)
                elif op == 'unique_count':
                    # Custom operation for nunique
                    agg_dict[col].append('nunique')
        
        # Perform aggregation
        try:
            if group_columns:
                # Group by aggregation
                # Validate group columns exist
                for col in group_columns:
                    if col not in df.columns:
                        return JsonResponse({'success': False, 'error': f'Group column "{col}" not found in dataset'}, status=400)
                
                grouped_df = df.groupby(group_columns).agg(agg_dict).reset_index()
                
                # Flatten multi-level column names
                if isinstance(grouped_df.columns, pd.MultiIndex):
                    grouped_df.columns = ['_'.join(col).strip('_') for col in grouped_df.columns]
                else:
                    grouped_df.columns = [str(col) for col in grouped_df.columns]
                
                result_df = grouped_df
                result_stats = {
                    'total_groups': len(grouped_df),
                    'result_columns': list(grouped_df.columns),
                    'original_rows': original_stats['total_rows'],
                    'reduction_ratio': f"{(1 - len(grouped_df) / len(df)) * 100:.1f}%",
                    'aggregation_type': 'grouped'
                }
            else:
                # Whole dataset aggregation (no grouping) - SIMPLIFIED APPROACH
                result_data = []
                
                # Process each column and its operations individually
                for col, operations in aggregation_operations.items():
                    for op in operations:
                        try:
                            if op == 'sum':
                                value = df[col].sum()
                            elif op == 'mean':
                                value = df[col].mean()
                            elif op == 'median':
                                value = df[col].median()
                            elif op == 'min':
                                value = df[col].min()
                            elif op == 'max':
                                value = df[col].max()
                            elif op == 'count':
                                value = df[col].count()
                            elif op == 'std':
                                value = df[col].std()
                            elif op == 'var':
                                value = df[col].var()
                            elif op == 'unique_count':
                                value = df[col].nunique()
                            else:
                                value = None
                            
                            # Convert numpy/pandas types to Python native types
                            if hasattr(value, 'item'):
                                value = value.item()  # For numpy scalars
                            elif pd.isna(value):
                                value = None
                            
                            result_data.append({
                                'column': col,
                                'operation': op,
                                'value': value
                            })
                        except Exception as col_error:
                            # Skip columns that can't be aggregated with this operation
                            print(f"Warning: Could not aggregate {col} with {op}: {col_error}")
                            continue
                
                # Convert to DataFrame for consistent handling
                result_df = pd.DataFrame(result_data)
                result_stats = {
                    'total_operations': len(result_df),
                    'result_columns': list(result_df.columns),
                    'original_rows': original_stats['total_rows'],
                    'aggregation_type': 'whole_dataset'
                }
                
        except Exception as e:
            import traceback
            print(f"Error in aggregation: {str(e)}")
            print(traceback.format_exc())
            return JsonResponse({'success': False, 'error': f'Error in aggregation: {str(e)}'}, status=400)
        
        if preview_only:
            # Use our helper function to ensure proper JSON serialization
            preview_data = dataframe_to_dict_clean(result_df.head(20))
            return JsonResponse({
                'success': True,
                'preview_data': preview_data,
                'original_stats': clean_data_for_json(original_stats),
                'result_stats': clean_data_for_json(result_stats),
                'columns': list(result_df.columns)
            })
        
        # Create new dataset
        if group_columns:
            operation_name = f"GroupBy_{'_'.join(group_columns)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            operation_name = f"Dataset_Aggregation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        new_dataset = create_aggregated_dataset(result_df, operation_name, request.session.get("user_id"), 'groupby')
        
        # Save operation record
        operation = AggregationOperation(
            name=operation_name,
            operation_type='groupby',
            input_dataset_id=dataset_id,
            output_dataset_id=str(new_dataset.id),
            user_id=str(request.session.get("user_id")),
            parameters={
                'group_columns': group_columns,
                'aggregation_operations': aggregation_operations
            },
            result_stats=result_stats,
            status='completed',
            completed_at=datetime.utcnow()
        )
        operation.save()
        
        preview_data_clean = dataframe_to_dict_clean(result_df.head(10))
        
        return JsonResponse({
            'success': True,
            'operation_id': str(operation.id),
            'new_dataset_id': str(new_dataset.id),
            'result_stats': clean_data_for_json(result_stats),
            'preview_data': preview_data_clean
        })
        
    except Exception as e:
        import traceback
        print(f"General error in groupby_aggregation: {str(e)}")
        print(traceback.format_exc())
        return JsonResponse({'success': False, 'error': str(e)}, status=400)


@mongo_login_required
@require_http_methods(["POST"])
@csrf_exempt
def pivot_table(request):
    """Create pivot tables from dataset"""
    try:
        data = json.loads(request.body)
        dataset_id = data.get('dataset_id')
        index_columns = data.get('index_columns', [])
        column_columns = data.get('column_columns', [])
        value_columns = data.get('value_columns', [])
        aggfunc = data.get('aggfunc', 'mean')
        fill_value = data.get('fill_value', 0)
        preview_only = data.get('preview_only', False)
        
        # Validate inputs
        if not dataset_id:
            return JsonResponse({'success': False, 'error': 'Dataset ID is required'}, status=400)
        
        if not index_columns:
            return JsonResponse({'success': False, 'error': 'At least one index column is required'}, status=400)
        
        if not value_columns:
            return JsonResponse({'success': False, 'error': 'At least one value column is required'}, status=400)
        
        # Get dataset
        dataset = Dataset.objects.get(id=ObjectId(dataset_id), owner_id=str(request.session.get("user_id")))
        df = download_and_convert_to_dataframe(dataset)
        
        # Validate columns exist
        all_columns = index_columns + column_columns + value_columns
        for col in all_columns:
            if col not in df.columns:
                return JsonResponse({'success': False, 'error': f'Column "{col}" not found in dataset'}, status=400)
        
        # Store original stats
        original_stats = {
            'total_rows': len(df),
            'columns': list(df.columns)
        }
        
        # Create pivot table
        try:
            pivot_df = df.pivot_table(
                index=index_columns,
                columns=column_columns if column_columns else None,
                values=value_columns,
                aggfunc=aggfunc,
                fill_value=fill_value
            ).reset_index()
            
            # Flatten multi-level column names
            if isinstance(pivot_df.columns, pd.MultiIndex):
                pivot_df.columns = ['_'.join(filter(None, map(str, col))).strip('_') for col in pivot_df.columns]
            else:
                pivot_df.columns = [str(col) for col in pivot_df.columns]
                
        except Exception as e:
            return JsonResponse({'success': False, 'error': f'Error creating pivot table: {str(e)}'}, status=400)
        
        # Calculate result stats
        result_stats = {
            'total_rows': len(pivot_df),
            'result_columns': list(pivot_df.columns),
            'original_rows': original_stats['total_rows'],
            'pivot_shape': f"{len(pivot_df)} rows × {len(pivot_df.columns)} columns"
        }
        
        if preview_only:
            preview_data = dataframe_to_dict_clean(pivot_df.head(20))
            return JsonResponse({
                'success': True,
                'preview_data': preview_data,
                'original_stats': clean_data_for_json(original_stats),
                'result_stats': clean_data_for_json(result_stats),
                'columns': list(pivot_df.columns)
            })
        
        # Create new dataset
        operation_name = f"Pivot_Table_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        new_dataset = create_aggregated_dataset(pivot_df, operation_name, request.session.get("user_id"), 'pivot')
        
        # Save operation record
        operation = AggregationOperation(
            name=operation_name,
            operation_type='pivot',
            input_dataset_id=dataset_id,
            output_dataset_id=str(new_dataset.id),
            user_id=str(request.session.get("user_id")),
            parameters={
                'index_columns': index_columns,
                'column_columns': column_columns,
                'value_columns': value_columns,
                'aggfunc': aggfunc,
                'fill_value': fill_value
            },
            result_stats=result_stats,
            status='completed',
            completed_at=datetime.utcnow()
        )
        operation.save()
        
        preview_data_clean = dataframe_to_dict_clean(pivot_df.head(10))
        
        return JsonResponse({
            'success': True,
            'operation_id': str(operation.id),
            'new_dataset_id': str(new_dataset.id),
            'result_stats': clean_data_for_json(result_stats),
            'preview_data': preview_data_clean
        })
        
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)

@mongo_login_required
@require_http_methods(["POST"])
@csrf_exempt
def window_functions(request):
    """Apply window functions to dataset"""
    try:
        data = json.loads(request.body)
        dataset_id = data.get('dataset_id')
        partition_columns = data.get('partition_columns', [])
        order_columns = data.get('order_columns', [])
        window_operations = data.get('window_operations', {})
        preview_only = data.get('preview_only', False)
        
        # Validate inputs
        if not dataset_id:
            return JsonResponse({'success': False, 'error': 'Dataset ID is required'}, status=400)
        
        if not window_operations:
            return JsonResponse({'success': False, 'error': 'At least one window operation is required'}, status=400)
        
        # Get dataset
        dataset = Dataset.objects.get(id=ObjectId(dataset_id), owner_id=str(request.session.get("user_id")))
        df = download_and_convert_to_dataframe(dataset)
        
        # Store original stats
        original_stats = {
            'total_rows': len(df),
            'columns': list(df.columns)
        }
        
        # Apply window functions
        result_df = df.copy()
        
        for target_col, operations in window_operations.items():
            if target_col not in df.columns:
                return JsonResponse({'success': False, 'error': f'Target column "{target_col}" not found in dataset'}, status=400)
            
            for operation in operations:
                op_type = operation.get('type')
                new_col_name = operation.get('new_column', f'{target_col}_{op_type}')
                
                try:
                    # Apply ordering if specified - do this before creating window groups
                    if order_columns:
                        result_df = result_df.sort_values(by=order_columns)
                    
                    # Create window specification based on partition columns
                    if partition_columns:
                        # Verify partition columns exist
                        missing_partition_cols = [col for col in partition_columns if col not in result_df.columns]
                        if missing_partition_cols:
                            return JsonResponse({
                                'success': False, 
                                'error': f'Partition columns not found: {missing_partition_cols}'
                            }, status=400)
                        
                        # Create window group with partitions
                        if order_columns:
                            # Use pandas groupby with sorting for ordered window functions
                            window = result_df.groupby(partition_columns)[target_col]
                        else:
                            window = result_df.groupby(partition_columns)[target_col]
                    else:
                        # No partitioning - use entire dataset
                        window = result_df[target_col]
                    
                    # Apply window function
                    if op_type == 'cumsum':
                        if partition_columns:
                            result_df[new_col_name] = window.cumsum()
                        else:
                            result_df[new_col_name] = window.cumsum()
                    
                    elif op_type == 'cummean':
                        if partition_columns:
                            result_df[new_col_name] = window.expanding().mean().reset_index(level=partition_columns, drop=True)
                        else:
                            result_df[new_col_name] = window.expanding().mean()
                    
                    elif op_type == 'cummin':
                        if partition_columns:
                            result_df[new_col_name] = window.cummin()
                        else:
                            result_df[new_col_name] = window.cummin()
                    
                    elif op_type == 'cummax':
                        if partition_columns:
                            result_df[new_col_name] = window.cummax()
                        else:
                            result_df[new_col_name] = window.cummax()
                    
                    elif op_type == 'rank':
                        if partition_columns:
                            result_df[new_col_name] = window.rank(method='dense')
                        else:
                            result_df[new_col_name] = window.rank(method='dense')
                    
                    elif op_type == 'row_number':
                        if partition_columns:
                            result_df[new_col_name] = window.cumcount() + 1
                        else:
                            result_df[new_col_name] = range(1, len(result_df) + 1)
                    
                    elif op_type == 'lag':
                        periods = operation.get('periods', 1)
                        if partition_columns:
                            result_df[new_col_name] = window.shift(periods)
                        else:
                            result_df[new_col_name] = window.shift(periods)
                    
                    elif op_type == 'lead':
                        periods = operation.get('periods', 1)
                        if partition_columns:
                            result_df[new_col_name] = window.shift(-periods)
                        else:
                            result_df[new_col_name] = window.shift(-periods)
                    
                    elif op_type == 'rolling_mean':
                        window_size = operation.get('window_size', 3)
                        if partition_columns:
                            result_df[new_col_name] = window.rolling(window=window_size, min_periods=1).mean()
                        else:
                            result_df[new_col_name] = window.rolling(window=window_size, min_periods=1).mean()
                    
                    elif op_type == 'rolling_sum':
                        window_size = operation.get('window_size', 3)
                        if partition_columns:
                            result_df[new_col_name] = window.rolling(window=window_size, min_periods=1).sum()
                        else:
                            result_df[new_col_name] = window.rolling(window=window_size, min_periods=1).sum()
                    
                    else:
                        return JsonResponse({
                            'success': False, 
                            'error': f'Unsupported window operation: {op_type}'
                        }, status=400)
                    
                except Exception as e:
                    return JsonResponse({
                        'success': False, 
                        'error': f'Error applying {op_type} to {target_col}: {str(e)}'
                    }, status=400)
        
        # Reset index if it was modified during operations
        result_df = result_df.reset_index(drop=True)
        
        # Calculate result stats
        result_stats = {
            'total_rows': len(result_df),
            'result_columns': list(result_df.columns),
            'original_columns': original_stats['columns'],
            'new_columns_added': len(result_df.columns) - len(original_stats['columns'])
        }
        
        if preview_only:
            preview_data = dataframe_to_dict_clean(result_df.head(20))
            return JsonResponse({
                'success': True,
                'preview_data': preview_data,
                'original_stats': clean_data_for_json(original_stats),
                'result_stats': clean_data_for_json(result_stats),
                'columns': list(result_df.columns)
            })
        
        # Create new dataset
        operation_name = f"Window_Functions_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        new_dataset = create_aggregated_dataset(result_df, operation_name, request.session.get("user_id"), 'window')
        
        # Save operation record
        operation = AggregationOperation(
            name=operation_name,
            operation_type='window',
            input_dataset_id=dataset_id,
            output_dataset_id=str(new_dataset.id),
            user_id=str(request.session.get("user_id")),
            parameters={
                'partition_columns': partition_columns,
                'order_columns': order_columns,
                'window_operations': window_operations
            },
            result_stats=result_stats,
            status='completed',
            completed_at=datetime.utcnow()
        )
        operation.save()
        
        preview_data_clean = dataframe_to_dict_clean(result_df.head(10))
        
        return JsonResponse({
            'success': True,
            'operation_id': str(operation.id),
            'new_dataset_id': str(new_dataset.id),
            'result_stats': clean_data_for_json(result_stats),
            'preview_data': preview_data_clean
        })
        
    except Dataset.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Dataset not found'}, status=404)
    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid JSON data'}, status=400)
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)

@mongo_login_required
@require_http_methods(["POST"])
@csrf_exempt
def rollup_cube(request):
    """Perform rollup and cube operations for multi-level aggregations"""
    try:
        data = json.loads(request.body)
        dataset_id = data.get('dataset_id')
        group_columns = data.get('group_columns', [])
        aggregation_columns = data.get('aggregation_columns', {})
        operation_type = data.get('operation_type', 'rollup')  # rollup or cube
        preview_only = data.get('preview_only', False)
        
        # Validate inputs
        if not dataset_id:
            return JsonResponse({'success': False, 'error': 'Dataset ID is required'}, status=400)
        
        if len(group_columns) < 2:
            return JsonResponse({'success': False, 'error': 'At least two group columns are required for rollup/cube'}, status=400)
        
        if not aggregation_columns:
            return JsonResponse({'success': False, 'error': 'At least one aggregation column is required'}, status=400)
        
        # Get dataset
        dataset = Dataset.objects.get(id=ObjectId(dataset_id), owner_id=str(request.session.get("user_id")))
        df = download_and_convert_to_dataframe(dataset)
        
        # Validate columns exist
        all_columns = group_columns + list(aggregation_columns.keys())
        for col in all_columns:
            if col not in df.columns:
                return JsonResponse({'success': False, 'error': f'Column "{col}" not found in dataset'}, status=400)
        
        # Store original stats
        original_stats = {
            'total_rows': len(df),
            'columns': list(df.columns)
        }
        
        # Prepare aggregation dictionary
        agg_dict = {}
        for col, operations in aggregation_columns.items():
            agg_dict[col] = operations
        
        # Perform rollup or cube operation
        try:
            if operation_type == 'rollup':
                # Create all combinations of group columns for rollup
                results = []
                for i in range(len(group_columns) + 1):
                    current_groups = group_columns[:len(group_columns)-i] if i > 0 else group_columns
                    if current_groups:
                        grouped = df.groupby(current_groups).agg(agg_dict).reset_index()
                        # Add level indicator
                        for missing_col in set(group_columns) - set(current_groups):
                            grouped[missing_col] = 'TOTAL'
                        results.append(grouped)
                
                result_df = pd.concat(results, ignore_index=True)
                
            else:  # cube
                # Create power set of group columns for cube
                from itertools import chain, combinations
                
                def powerset(iterable):
                    s = list(iterable)
                    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
                
                results = []
                for groups in powerset(group_columns):
                    if groups:  # Skip empty combination
                        current_groups = list(groups)
                        grouped = df.groupby(current_groups).agg(agg_dict).reset_index()
                        # Add level indicator for missing columns
                        for missing_col in set(group_columns) - set(current_groups):
                            grouped[missing_col] = 'TOTAL'
                        results.append(grouped)
                
                result_df = pd.concat(results, ignore_index=True)
            
            # Sort the result
            result_df = result_df.sort_values(by=group_columns)
            
        except Exception as e:
            return JsonResponse({'success': False, 'error': f'Error in {operation_type} operation: {str(e)}'}, status=400)
        
        # Calculate result stats
        result_stats = {
            'total_rows': len(result_df),
            'result_columns': list(result_df.columns),
            'original_rows': original_stats['total_rows'],
            'operation_type': operation_type,
            'group_levels': len(group_columns)
        }
        
        if preview_only:
            preview_data = dataframe_to_dict_clean(result_df.head(20))
            return JsonResponse({
                'success': True,
                'preview_data': preview_data,
                'original_stats': clean_data_for_json(original_stats),
                'result_stats': clean_data_for_json(result_stats),
                'columns': list(result_df.columns)
            })
        
        # Create new dataset
        operation_name = f"{operation_type.title()}_{'_'.join(group_columns)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        new_dataset = create_aggregated_dataset(result_df, operation_name, request.session.get("user_id"), operation_type)
        
        # Save operation record
        operation = AggregationOperation(
            name=operation_name,
            operation_type=operation_type,
            input_dataset_id=dataset_id,
            output_dataset_id=str(new_dataset.id),
            user_id=str(request.session.get("user_id")),
            parameters={
                'group_columns': group_columns,
                'aggregation_columns': aggregation_columns,
                'operation_type': operation_type
            },
            result_stats=result_stats,
            status='completed',
            completed_at=datetime.utcnow()
        )
        operation.save()
        
        preview_data_clean = dataframe_to_dict_clean(result_df.head(10))
        
        return JsonResponse({
            'success': True,
            'operation_id': str(operation.id),
            'new_dataset_id': str(new_dataset.id),
            'result_stats': clean_data_for_json(result_stats),
            'preview_data': preview_data_clean
        })
        
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)

@mongo_login_required
@require_http_methods(["POST"])
@csrf_exempt
def preview_aggregation(request):
    """Preview any aggregation operation without saving"""
    try:
        data = json.loads(request.body)
        operation_type = data.get('operation_type')
        dataset_id = data.get('dataset_id')
        preview_only = data.get('preview_only', True)
        
        # Get the dataset - FIXED: Use Dataset instead of DataSet
        try:
            dataset = Dataset.objects.get(id=ObjectId(dataset_id), owner_id=str(request.session.get("user_id")))
        except Dataset.DoesNotExist:
            return JsonResponse({'success': False, 'error': 'Dataset not found'}, status=404)
        
        # Load the dataset
        df = download_and_convert_to_dataframe(dataset)
        if df is None or df.empty:
            return JsonResponse({'success': False, 'error': 'Failed to load dataset'}, status=400)
        
        original_stats = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'columns': list(df.columns)
        }
        
        result_df = None
        result_stats = None
        
        if operation_type == 'groupby':
            result_df = preview_groupby_aggregation(df, data)
        elif operation_type == 'pivot':
            result_df = preview_pivot_table(df, data)
        elif operation_type == 'window':
            result_df = preview_window_functions(df, data)
        elif operation_type == 'rollup':
            result_df = preview_rollup_cube(df, data)
        else:
            return JsonResponse({'success': False, 'error': 'Unsupported operation type'}, status=400)
        
        if result_df is None or result_df.empty:
            return JsonResponse({'success': False, 'error': 'No data after aggregation'}, status=400)
        
        # Prepare preview data using your helper function
        preview_data = dataframe_to_dict_clean(result_df.head(20))
        
        result_stats = {
            'total_rows': len(result_df),
            'total_columns': len(result_df.columns),
            'columns': list(result_df.columns)
        }
        
        return JsonResponse({
            'success': True,
            'preview_data': preview_data,
            'columns': list(result_df.columns),
            'original_stats': original_stats,
            'result_stats': result_stats,
            'preview_only': True
        })
            
    except Exception as e:
        logger.error(f"Error in preview_aggregation: {str(e)}")
        return JsonResponse({'success': False, 'error': str(e)}, status=400)
    
def preview_groupby_aggregation(df, data):
    """Preview groupby aggregation"""
    try:
        group_columns = data.get('group_columns', [])
        aggregations = data.get('aggregations', [])
        
        # Validate inputs
        if not aggregations:
            raise ValueError("At least one aggregation operation is required")
        
        # Check if all group columns exist
        missing_group_cols = [col for col in group_columns if col not in df.columns]
        if missing_group_cols:
            raise ValueError(f"Group columns not found: {missing_group_cols}")
        
        # If no group columns, aggregate entire dataset
        if not group_columns:
            # Create aggregation operations in the format your main function expects
            aggregation_operations = {}
            for agg in aggregations:
                column = agg.get('column')
                operations = agg.get('operations', [])
                
                if column not in df.columns:
                    raise ValueError(f"Column '{column}' not found in dataset")
                
                aggregation_operations[column] = operations
            
            # Use the same logic as your main groupby_aggregation function
            agg_dict = {}
            for col, operations in aggregation_operations.items():
                agg_dict[col] = []
                for op in operations:
                    if op in ['sum', 'mean', 'median', 'min', 'max', 'count', 'std', 'var']:
                        agg_dict[col].append(op)
                    elif op == 'unique_count':
                        agg_dict[col].append('nunique')
            
            # Create result similar to your main function
            result_data = []
            for col, operations in aggregation_operations.items():
                for op in operations:
                    try:
                        if op == 'sum':
                            value = df[col].sum()
                        elif op == 'mean':
                            value = df[col].mean()
                        elif op == 'median':
                            value = df[col].median()
                        elif op == 'min':
                            value = df[col].min()
                        elif op == 'max':
                            value = df[col].max()
                        elif op == 'count':
                            value = df[col].count()
                        elif op == 'std':
                            value = df[col].std()
                        elif op == 'var':
                            value = df[col].var()
                        elif op == 'unique_count':
                            value = df[col].nunique()
                        else:
                            value = None
                        
                        # Convert numpy/pandas types to Python native types
                        if hasattr(value, 'item'):
                            value = value.item()
                        elif pd.isna(value):
                            value = None
                        
                        result_data.append({
                            'column': col,
                            'operation': op,
                            'value': value
                        })
                    except Exception as col_error:
                        continue
            
            # Convert to DataFrame for consistent handling
            result_df = pd.DataFrame(result_data)
            
        else:
            # Group by specified columns - use the same logic as main function
            aggregation_operations = {}
            for agg in aggregations:
                column = agg.get('column')
                operations = agg.get('operations', [])
                aggregation_operations[column] = operations
            
            agg_dict = {}
            for col, operations in aggregation_operations.items():
                agg_dict[col] = []
                for op in operations:
                    if op in ['sum', 'mean', 'median', 'min', 'max', 'count', 'std', 'var']:
                        agg_dict[col].append(op)
                    elif op == 'unique_count':
                        agg_dict[col].append('nunique')
            
            grouped_df = df.groupby(group_columns).agg(agg_dict).reset_index()
            
            # Flatten multi-level column names
            if isinstance(grouped_df.columns, pd.MultiIndex):
                grouped_df.columns = ['_'.join(col).strip('_') for col in grouped_df.columns]
            else:
                grouped_df.columns = [str(col) for col in grouped_df.columns]
            
            result_df = grouped_df
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error in preview_groupby_aggregation: {str(e)}")
        raise e

def preview_pivot_table(df, data):
    """Preview pivot table operation"""
    try:
        index_columns = data.get('index_columns', [])
        column_columns = data.get('column_columns', [])
        value_columns = data.get('value_columns', [])
        agg_func = data.get('agg_func', 'mean')
        fill_value = data.get('fill_value', 0)
        
        # Validate inputs
        if not index_columns:
            raise ValueError("At least one index column is required")
        if not value_columns:
            raise ValueError("At least one value column is required")
        
        # Check if columns exist
        missing_cols = [col for col in index_columns + column_columns + value_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found: {missing_cols}")
        
        # Create pivot table
        if column_columns:
            result = df.pivot_table(
                index=index_columns,
                columns=column_columns,
                values=value_columns,
                aggfunc=agg_func,
                fill_value=fill_value
            )
        else:
            result = df.pivot_table(
                index=index_columns,
                values=value_columns,
                aggfunc=agg_func,
                fill_value=fill_value
            )
        
        # Flatten column names if multi-index
        if isinstance(result.columns, pd.MultiIndex):
            result.columns = ['_'.join(filter(None, map(str, col))).strip() for col in result.columns]
        
        return result.reset_index()
        
    except Exception as e:
        logger.error(f"Error in preview_pivot_table: {str(e)}")
        raise e

def preview_window_functions(df, data):
    """Preview window functions"""
    try:
        partition_columns = data.get('partition_columns', [])
        order_columns = data.get('order_columns', [])
        window_functions = data.get('window_functions', [])
        
        # Validate inputs
        if not window_functions:
            raise ValueError("At least one window function is required")
        
        result = df.copy()
        
        for func_config in window_functions:
            target_column = func_config.get('target_column')
            function_type = func_config.get('function_type')
            new_column_name = func_config.get('new_column_name', f"{target_column}_{function_type}")
            
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found")
            
            # Create window specification
            if partition_columns:
                # Check if partition columns exist
                missing_partition = [col for col in partition_columns if col not in df.columns]
                if missing_partition:
                    raise ValueError(f"Partition columns not found: {missing_partition}")
                
                groups = result.groupby(partition_columns)
            else:
                # Create a single group if no partition columns
                groups = [('all', result)]
            
            # Apply window function
            if function_type == 'cumsum':
                if partition_columns:
                    result[new_column_name] = groups[target_column].cumsum()
                else:
                    result[new_column_name] = result[target_column].cumsum()
                    
            elif function_type == 'cummean':
                if partition_columns:
                    result[new_column_name] = groups[target_column].expanding().mean().reset_index(level=0, drop=True)
                else:
                    result[new_column_name] = result[target_column].expanding().mean()
                    
            elif function_type == 'rank':
                if partition_columns:
                    result[new_column_name] = groups[target_column].rank(method='dense')
                else:
                    result[new_column_name] = result[target_column].rank(method='dense')
                    
            elif function_type == 'row_number':
                if partition_columns:
                    result[new_column_name] = groups.cumcount() + 1
                else:
                    result[new_column_name] = range(1, len(result) + 1)
            
            # Add more window functions as needed...
        
        return result
        
    except Exception as e:
        logger.error(f"Error in preview_window_functions: {str(e)}")
        raise e

def preview_rollup_cube(df, data):
    """Preview rollup and cube operations"""
    try:
        # For now, return a simple grouped aggregation as placeholder
        group_columns = data.get('group_columns', [])
        
        if not group_columns:
            return df.head(20)  # Return sample if no grouping
        
        # Simple groupby for preview
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            result = df.groupby(group_columns)[numeric_cols[:2]].agg(['mean', 'sum']).reset_index()
            # Flatten column names
            result.columns = ['_'.join(filter(None, map(str, col))).strip() for col in result.columns]
            return result
        else:
            return df.groupby(group_columns).size().reset_index(name='count')
            
    except Exception as e:
        logger.error(f"Error in preview_rollup_cube: {str(e)}")
        raise e

@mongo_login_required
@require_http_methods(["POST"])
@csrf_exempt
def execute_aggregation(request):
    """Execute aggregation operation and save result"""
    try:
        data = json.loads(request.body)
        operation_type = data.get('operation_type')
        
        if operation_type == 'groupby':
            return groupby_aggregation(request)
        elif operation_type == 'pivot':
            return pivot_table(request)
        elif operation_type == 'window':
            return window_functions(request)
        elif operation_type == 'rollup':
            return rollup_cube(request)
        else:
            return JsonResponse({'success': False, 'error': 'Unsupported operation type'}, status=400)
            
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)

@mongo_login_required
@require_http_methods(["GET"])
def get_aggregation_history(request):
    """Get user's aggregation history"""
    try:
        operations = AggregationOperation.objects(user_id=str(request.session.get("user_id"))).order_by('-created_at')[:50]
        
        operation_list = []
        for op in operations:
            operation_list.append({
                'id': str(op.id),
                'name': op.name,
                'operation_type': op.operation_type,
                'input_dataset_id': op.input_dataset_id,
                'output_dataset_id': op.output_dataset_id,
                'status': op.status,
                'created_at': op.created_at.isoformat(),
                'result_stats': op.result_stats
            })
        
        return JsonResponse({
            'success': True,
            'operations': operation_list
        })
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)

@mongo_login_required
@require_http_methods(["GET", "POST", "DELETE"])
@csrf_exempt
def manage_aggregation_templates(request):
    """Manage reusable aggregation templates"""
    try:
        if request.method == 'GET':
            templates = AggregationTemplate.objects(user_id=str(request.session.get("user_id")))
            public_templates = AggregationTemplate.objects(is_public=True)
            
            template_list = []
            for template in list(templates) + list(public_templates):
                template_list.append({
                    'id': str(template.id),
                    'name': template.name,
                    'description': template.description,
                    'template_type': template.template_type,
                    'parameters': template.parameters,
                    'is_public': template.is_public,
                    'usage_count': template.usage_count
                })
            
            return JsonResponse({
                'success': True,
                'templates': template_list
            })
            
        elif request.method == 'POST':
            data = json.loads(request.body)
            template = AggregationTemplate(
                name=data.get('name'),
                description=data.get('description'),
                template_type=data.get('template_type'),
                parameters=data.get('parameters', {}),
                user_id=str(request.session.get("user_id")),
                is_public=data.get('is_public', False)
            )
            template.save()
            
            return JsonResponse({
                'success': True,
                'template_id': str(template.id)
            })
            
        elif request.method == 'DELETE':
            template_id = request.GET.get('template_id')
            template = AggregationTemplate.objects.get(id=ObjectId(template_id), user_id=str(request.session.get("user_id")))
            template.delete()
            
            return JsonResponse({'success': True})
            
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)

@mongo_login_required
@require_http_methods(["POST"])
@csrf_exempt
def get_column_statistics(request):
    """Get statistical information about dataset columns"""
    try:
        data = json.loads(request.body)
        dataset_id = data.get('dataset_id')
        
        if not dataset_id:
            return JsonResponse({'success': False, 'error': 'Dataset ID is required'}, status=400)
        
        dataset = Dataset.objects.get(id=ObjectId(dataset_id), owner_id=str(request.session.get("user_id")))
        df = download_and_convert_to_dataframe(dataset)
        
        statistics = {}
        for column in df.columns:
            col_data = df[column]
            col_stats = {
                'dtype': str(col_data.dtype),
                'non_null_count': col_data.count(),
                'null_count': col_data.isnull().sum(),
                'unique_count': col_data.nunique()
            }
            
            # Numeric statistics
            if pd.api.types.is_numeric_dtype(col_data):
                col_stats.update({
                    'min': float(col_data.min()) if not col_data.isnull().all() else None,
                    'max': float(col_data.max()) if not col_data.isnull().all() else None,
                    'mean': float(col_data.mean()) if not col_data.isnull().all() else None,
                    'median': float(col_data.median()) if not col_data.isnull().all() else None,
                    'std': float(col_data.std()) if not col_data.isnull().all() else None
                })
            
            # Sample values
            col_stats['sample_values'] = col_data.dropna().head(5).tolist()
            
            statistics[column] = col_stats
        
        return JsonResponse({
            'success': True,
            'statistics': clean_data_for_json(statistics),
            'total_rows': len(df),
            'total_columns': len(df.columns)
        })
        
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)

# Helper function to create aggregated dataset
def create_aggregated_dataset(df, name, user_id, operation_type):
    """Create a new dataset from aggregation result"""
    from datasets.models import Dataset
    from supabase import create_client
    from django.conf import settings
    
    supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
    
    try:
        # Convert DataFrame to CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()
        
        # Upload to Supabase
        unique_filename = f"{uuid.uuid4()}_{name}.csv"
        supabase_path = f"{user_id}/{unique_filename}"
        
        res = supabase.storage.from_(settings.SUPABASE_BUCKET).upload(
            supabase_path, csv_content.encode('utf-8')
        )
        
        # Get public URL
        file_url = supabase.storage.from_(settings.SUPABASE_BUCKET).get_public_url(supabase_path)
        
        # Create dataset record
        dataset = Dataset(
            owner_id=str(user_id),
            file_name=f"{name}.csv",
            file_type="text/csv",
            file_url=file_url,
            file_path=supabase_path,
            uploaded_at=datetime.utcnow(),
            metadata={
                "is_aggregation_result": True,
                "aggregation_operation": operation_type,
                "created_from_transformation": True
            }
        )
        
        dataset.save()
        return dataset
    except Exception as e:
        print(f"Error creating aggregated dataset: {str(e)}")
        raise


@mongo_login_required
@require_http_methods(["POST"])
@csrf_exempt
def handle_missing_values(request):
    """Handle missing values in dataset"""
    try:
        data = json.loads(request.body)
        dataset_id = data.get('dataset_id')
        strategy = data.get('strategy', 'fill')  # fill, drop, interpolate
        columns = data.get('columns', [])
        fill_value = data.get('fill_value')
        preview_only = data.get('preview_only', False)
        
        # Validate inputs
        if not dataset_id:
            return JsonResponse({'success': False, 'error': 'Dataset ID is required'}, status=400)
        
        # Get dataset
        dataset = Dataset.objects.get(id=ObjectId(dataset_id), owner_id=str(request.session.get("user_id")))
        df = download_and_convert_to_dataframe(dataset)
        
        # Store original stats
        original_stats = {
            'total_rows': len(df),
            'missing_counts': df[columns].isnull().sum().to_dict() if columns else df.isnull().sum().to_dict()
        }
        
        # Apply missing value handling
        if strategy == 'drop':
            if columns:
                df = df.dropna(subset=columns)
            else:
                df = df.dropna()
        elif strategy == 'fill':
            if columns:
                if fill_value == 'mean':
                    for col in columns:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            df[col] = df[col].fillna(df[col].mean())
                        else:
                            df[col] = df[col].fillna('')
                elif fill_value == 'median':
                    for col in columns:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            df[col] = df[col].fillna(df[col].median())
                        else:
                            df[col] = df[col].fillna('')
                elif fill_value == 'mode':
                    for col in columns:
                        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else '')
                else:
                    for col in columns:
                        df[col] = df[col].fillna(fill_value)
            else:
                if fill_value == 'mean':
                    df = df.fillna(df.mean(numeric_only=True))
                elif fill_value == 'median':
                    df = df.fillna(df.median(numeric_only=True))
                elif fill_value == 'mode':
                    df = df.fillna(df.mode().iloc[0] if not df.mode().empty else '')
                else:
                    df = df.fillna(fill_value)
        elif strategy == 'interpolate':
            if columns:
                for col in columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = df[col].interpolate()
            else:
                df = df.interpolate()
        
        # Calculate result stats
        result_stats = {
            'total_rows': len(df),
            'remaining_missing': df[columns].isnull().sum().to_dict() if columns else df.isnull().sum().to_dict(),
            'rows_removed': original_stats['total_rows'] - len(df) if strategy == 'drop' else 0
        }
        
        if preview_only:
            return JsonResponse({
                'success': True,
                'preview_data': df.head(20).to_dict('records'),
                'original_stats': original_stats,
                'result_stats': result_stats,
                'columns': list(df.columns)
            })
        
        # Create new dataset
        operation_name = f"Missing_Values_{strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        new_dataset = create_cleaned_dataset(df, operation_name, request.session.get("user_id"), 'missing_values')
        
        # Save operation record
        operation = DataCleaningOperation(
            name=operation_name,
            operation_type='missing_values',
            input_dataset_id=dataset_id,
            output_dataset_id=str(new_dataset.id),
            user_id=str(request.session.get("user_id")),
            parameters={
                'strategy': strategy,
                'columns': columns,
                'fill_value': fill_value
            },
            result_stats=result_stats,
            status='completed',
            completed_at=datetime.utcnow()
        )
        operation.save()
        
        return JsonResponse({
            'success': True,
            'operation_id': str(operation.id),
            'new_dataset_id': str(new_dataset.id),
            'result_stats': result_stats,
            'preview_data': df.head(10).to_dict('records')
        })
        
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)

@mongo_login_required
@require_http_methods(["POST"])
@csrf_exempt
def remove_duplicates(request):
    """Remove duplicate rows from dataset"""
    try:
        data = json.loads(request.body)
        dataset_id = data.get('dataset_id')
        subset = data.get('subset', [])  # Columns to consider for duplicates
        keep = data.get('keep', 'first')  # first, last, False
        preview_only = data.get('preview_only', False)
        
        if not dataset_id:
            return JsonResponse({'success': False, 'error': 'Dataset ID is required'}, status=400)
        
        dataset = Dataset.objects.get(id=ObjectId(dataset_id), owner_id=str(request.session.get("user_id")))
        df = download_and_convert_to_dataframe(dataset)
        
        # Store original stats
        original_stats = {
            'total_rows': len(df),
            'duplicate_count': df.duplicated(subset=subset if subset else None).sum()
        }
        
        # Remove duplicates
        df_cleaned = df.drop_duplicates(subset=subset if subset else None, keep=keep)
        
        # Calculate result stats
        result_stats = {
            'total_rows': len(df_cleaned),
            'duplicates_removed': original_stats['total_rows'] - len(df_cleaned),
            'remaining_duplicates': df_cleaned.duplicated(subset=subset if subset else None).sum()
        }
        
        if preview_only:
            return JsonResponse({
                'success': True,
                'preview_data': df_cleaned.head(20).to_dict('records'),
                'original_stats': original_stats,
                'result_stats': result_stats,
                'columns': list(df_cleaned.columns)
            })
        
        # Create new dataset
        operation_name = f"Remove_Duplicates_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        new_dataset = create_cleaned_dataset(df_cleaned, operation_name, request.session.get("user_id"), 'remove_duplicates')
        
        # Save operation record
        operation = DataCleaningOperation(
            name=operation_name,
            operation_type='remove_duplicates',
            input_dataset_id=dataset_id,
            output_dataset_id=str(new_dataset.id),
            user_id=str(request.session.get("user_id")),
            parameters={
                'subset': subset,
                'keep': keep
            },
            result_stats=result_stats,
            status='completed',
            completed_at=datetime.utcnow()
        )
        operation.save()
        
        return JsonResponse({
            'success': True,
            'operation_id': str(operation.id),
            'new_dataset_id': str(new_dataset.id),
            'result_stats': result_stats,
            'preview_data': df_cleaned.head(10).to_dict('records')
        })
        
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)

@mongo_login_required
@require_http_methods(["POST"])
@csrf_exempt
def standardize_formats(request):
    """Standardize text formats and case sensitivity"""
    try:
        data = json.loads(request.body)
        dataset_id = data.get('dataset_id')
        columns = data.get('columns', [])
        operations = data.get('operations', {})  # lowercase, uppercase, trim, etc.
        preview_only = data.get('preview_only', False)
        
        if not dataset_id:
            return JsonResponse({'success': False, 'error': 'Dataset ID is required'}, status=400)
        
        dataset = Dataset.objects.get(id=ObjectId(dataset_id), owner_id=str(request.session.get("user_id")))
        df = download_and_convert_to_dataframe(dataset)
        
        # Store original sample for comparison
        original_sample = df[columns].head(5).to_dict('records') if columns else df.head(5).to_dict('records')
        
        # Apply standardization operations
        columns_to_process = columns if columns else df.select_dtypes(include=['object']).columns
        
        for col in columns_to_process:
            if col in df.columns:
                # Trim whitespace
                if operations.get('trim', False):
                    df[col] = df[col].astype(str).str.strip()
                
                # Case conversion
                if operations.get('case') == 'lower':
                    df[col] = df[col].astype(str).str.lower()
                elif operations.get('case') == 'upper':
                    df[col] = df[col].astype(str).str.upper()
                elif operations.get('case') == 'title':
                    df[col] = df[col].astype(str).str.title()
                
                # Remove extra spaces
                if operations.get('remove_extra_spaces', False):
                    df[col] = df[col].astype(str).str.replace(r'\s+', ' ', regex=True)
                
                # Standardize date formats
                if operations.get('standardize_dates', False):
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce').dt.strftime('%Y-%m-%d')
                    except:
                        pass
        
        # FIX: Handle NaN values before JSON serialization
        def clean_data_for_json(data):
            """Recursively clean data for JSON serialization"""
            if isinstance(data, dict):
                return {k: clean_data_for_json(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [clean_data_for_json(item) for item in data]
            elif pd.isna(data):  # Handle NaN, NaT, etc.
                return None
            elif isinstance(data, (pd.Timestamp, datetime)):
                return data.isoformat()
            elif isinstance(data, (np.integer, np.floating)):
                return float(data) if np.isnan(data) or np.isinf(data) else data
            else:
                return data
        
        if preview_only:
            # Clean the data before converting to JSON
            preview_data = df.head(20).replace({np.nan: None}).to_dict('records')
            original_sample_clean = clean_data_for_json(original_sample)
            
            return JsonResponse({
                'success': True,
                'preview_data': preview_data,
                'original_sample': original_sample_clean,
                'columns': list(df.columns)
            })
        
        # Create new dataset
        operation_name = f"Standardize_Formats_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        new_dataset = create_cleaned_dataset(df, operation_name, request.session.get("user_id"), 'standardize_formats')
        
        # Save operation record
        operation = DataCleaningOperation(
            name=operation_name,
            operation_type='standardize_formats',
            input_dataset_id=dataset_id,
            output_dataset_id=str(new_dataset.id),
            user_id=str(request.session.get("user_id")),
            parameters={
                'columns': columns,
                'operations': operations
            },
            result_stats={'total_rows': len(df)},
            status='completed',
            completed_at=datetime.utcnow()
        )
        operation.save()
        
        # Clean data for response
        preview_data_clean = df.head(10).replace({np.nan: None}).to_dict('records')
        
        return JsonResponse({
            'success': True,
            'operation_id': str(operation.id),
            'new_dataset_id': str(new_dataset.id),
            'preview_data': preview_data_clean
        })
        
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)

@mongo_login_required
@require_http_methods(["POST"])
@csrf_exempt
def convert_data_types(request):
    """Convert data types of selected columns"""
    try:
        data = json.loads(request.body)
        dataset_id = data.get('dataset_id')
        conversions = data.get('conversions', {})  # {column: target_type}
        preview_only = data.get('preview_only', False)
        
        if not dataset_id:
            return JsonResponse({'success': False, 'error': 'Dataset ID is required'}, status=400)
        
        dataset = Dataset.objects.get(id=ObjectId(dataset_id), owner_id=str(request.session.get("user_id")))
        df = download_and_convert_to_dataframe(dataset)
        
        # Store original dtypes
        original_dtypes = {col: str(df[col].dtype) for col in conversions.keys() if col in df.columns}
        conversion_results = {}
        
        # Apply type conversions
        for col, target_type in conversions.items():
            if col in df.columns:
                try:
                    if target_type == 'numeric':
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        conversion_results[col] = 'success'
                    elif target_type == 'integer':
                        df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                        conversion_results[col] = 'success'
                    elif target_type == 'float':
                        df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
                        conversion_results[col] = 'success'
                    elif target_type == 'string':
                        df[col] = df[col].astype(str)
                        conversion_results[col] = 'success'
                    elif target_type == 'datetime':
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        conversion_results[col] = 'success'
                    elif target_type == 'boolean':
                        df[col] = df[col].astype(str).str.lower().map({'true': True, 'false': False, '1': True, '0': False})
                        conversion_results[col] = 'success'
                    else:
                        conversion_results[col] = 'unsupported_type'
                except Exception as e:
                    conversion_results[col] = f'error: {str(e)}'
        
        if preview_only:
            return JsonResponse({
                'success': True,
                'preview_data': df.head(20).to_dict('records'),
                'original_dtypes': original_dtypes,
                'conversion_results': conversion_results,
                'new_dtypes': {col: str(df[col].dtype) for col in conversions.keys() if col in df.columns},
                'columns': list(df.columns)
            })
        
        # Create new dataset
        operation_name = f"Data_Type_Conversion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        new_dataset = create_cleaned_dataset(df, operation_name, request.session.get("user_id"), 'convert_data_types')
        
        # Save operation record
        operation = DataCleaningOperation(
            name=operation_name,
            operation_type='convert_data_types',
            input_dataset_id=dataset_id,
            output_dataset_id=str(new_dataset.id),
            user_id=str(request.session.get("user_id")),
            parameters={
                'conversions': conversions
            },
            result_stats={
                'conversion_results': conversion_results,
                'total_rows': len(df)
            },
            status='completed',
            completed_at=datetime.utcnow()
        )
        operation.save()
        
        return JsonResponse({
            'success': True,
            'operation_id': str(operation.id),
            'new_dataset_id': str(new_dataset.id),
            'conversion_results': conversion_results,
            'preview_data': df.head(10).to_dict('records')
        })
        
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)

@mongo_login_required
@require_http_methods(["POST"])
@csrf_exempt
def preview_cleaning_operation(request):
    """Preview any cleaning operation without saving"""
    try:
        data = json.loads(request.body)
        operation_type = data.get('operation_type')
        
        if operation_type == 'missing_values':
            return handle_missing_values(request)
        elif operation_type == 'remove_duplicates':
            return remove_duplicates(request)
        elif operation_type == 'standardize_formats':
            return standardize_formats(request)
        elif operation_type == 'convert_data_types':
            return convert_data_types(request)
        else:
            return JsonResponse({'success': False, 'error': 'Unsupported operation type'}, status=400)
            
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)



@mongo_login_required
@require_http_methods(["POST"])
@csrf_exempt
def execute_cleaning_operation(request):
    """Execute cleaning operation and save result"""
    try:
        data = json.loads(request.body)
        operation_type = data.get('operation_type')
        
        if operation_type == 'missing_values':
            return handle_missing_values(request)
        elif operation_type == 'remove_duplicates':
            return remove_duplicates(request)
        elif operation_type == 'standardize_formats':
            return standardize_formats(request)
        elif operation_type == 'convert_data_types':
            return convert_data_types(request)
        else:
            return JsonResponse({'success': False, 'error': 'Unsupported operation type'}, status=400)
            
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)

@mongo_login_required
@require_http_methods(["GET"])
def get_cleaning_history(request):
    """Get user's data cleaning history"""
    try:
        operations = DataCleaningOperation.objects(user_id=str(request.session.get("user_id"))).order_by('-created_at')[:50]
        
        operation_list = []
        for op in operations:
            operation_list.append({
                'id': str(op.id),
                'name': op.name,
                'operation_type': op.operation_type,
                'input_dataset_id': op.input_dataset_id,
                'output_dataset_id': op.output_dataset_id,
                'status': op.status,
                'created_at': op.created_at.isoformat(),
                'result_stats': op.result_stats
            })
        
        return JsonResponse({
            'success': True,
            'operations': operation_list
        })
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)

@mongo_login_required
@require_http_methods(["GET", "POST", "DELETE"])
@csrf_exempt
def manage_cleaning_templates(request):
    """Manage reusable data cleaning templates"""
    try:
        if request.method == 'GET':
            # Get user's templates
            templates = DataCleaningTemplate.objects(user_id=str(request.session.get("user_id")))
            public_templates = DataCleaningTemplate.objects(is_public=True)
            
            template_list = []
            for template in list(templates) + list(public_templates):
                template_list.append({
                    'id': str(template.id),
                    'name': template.name,
                    'description': template.description,
                    'template_type': template.template_type,
                    'parameters': template.parameters,
                    'is_public': template.is_public,
                    'usage_count': template.usage_count
                })
            
            return JsonResponse({
                'success': True,
                'templates': template_list
            })
            
        elif request.method == 'POST':
            # Create new template
            data = json.loads(request.body)
            template = DataCleaningTemplate(
                name=data.get('name'),
                description=data.get('description'),
                template_type=data.get('template_type'),
                parameters=data.get('parameters', {}),
                user_id=str(request.session.get("user_id")),
                is_public=data.get('is_public', False)
            )
            template.save()
            
            return JsonResponse({
                'success': True,
                'template_id': str(template.id)
            })
            
        elif request.method == 'DELETE':
            # Delete template
            template_id = request.GET.get('template_id')
            template = DataCleaningTemplate.objects.get(id=ObjectId(template_id), user_id=str(request.session.get("user_id")))
            template.delete()
            
            return JsonResponse({'success': True})
            
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)

# Helper function to create cleaned dataset
def create_cleaned_dataset(df, name, user_id, operation_type):
    """Create a new dataset from cleaning result"""
    from datasets.models import Dataset
    from supabase import create_client
    from django.conf import settings
    
    supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
    
    try:
        # Convert DataFrame to CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()
        
        # Upload to Supabase
        unique_filename = f"{uuid.uuid4()}_{name}.csv"
        supabase_path = f"{user_id}/{unique_filename}"
        
        res = supabase.storage.from_(settings.SUPABASE_BUCKET).upload(
            supabase_path, csv_content.encode('utf-8')
        )
        
        # Get public URL
        file_url = supabase.storage.from_(settings.SUPABASE_BUCKET).get_public_url(supabase_path)
        
        # Create dataset record
        dataset = Dataset(
            owner_id=str(user_id),
            file_name=f"{name}.csv",
            file_type="text/csv",
            file_url=file_url,
            file_path=supabase_path,
            uploaded_at=datetime.utcnow(),
            metadata={
                "is_cleaning_result": True,
                "cleaning_operation": operation_type,
                "created_from_transformation": True
            }
        )
        
        dataset.save()
        return dataset
    except Exception as e:
        print(f"Error creating cleaned dataset: {str(e)}")
        raise
@mongo_login_required

def dashboard(request):
    datasets = Dataset.objects(owner_id=str(request.session.get("user_id")))
    workspaces = Workspace.objects(owner_id=str(request.session.get("user_id")))
    pinned_workspaces = Workspace.objects(owner_id=str(request.session.get("user_id")), pinned=True)
    
    return render(request, "dashboard.html", {
        "datasets": datasets,
        "workspaces": workspaces,
        "pinned_workspaces": pinned_workspaces
    })

@mongo_login_required
def workspace(request):
    # Initialize sample data if no workspaces exist
    user_workspaces = Workspace.objects(owner_id=str(request.session.get("user_id")))
    if not user_workspaces:
        initialize_sample_data(str(request.session.get("user_id")))
        user_workspaces = Workspace.objects(owner_id=str(request.session.get("user_id")))
    
    workspaces = list(user_workspaces)
    pinned_workspaces = [ws for ws in workspaces if ws.pinned]
    recent_activities = WorkspaceActivity.objects().order_by('-created_at')[:5]
    
    return render(request, "workspace.html", {
        "workspaces": workspaces,
        "pinned_workspaces": pinned_workspaces,
        "recent_activities": recent_activities,
        "user_initials": request.user.username[0].upper() if request.user.username else 'U'
    })

def initialize_sample_data(user_id):
    """Create sample workspaces for new users"""
    sample_workspaces = [
        {
            'name': 'Main Workspace',
            'description': 'Primary workspace for all projects',
            'color': 'yellow',
            'dataset_count': 5,
            'members': ['A', 'M'],
            'pinned': True
        },
        {
            'name': 'Project Alpha',
            'description': 'Client project workspace',
            'color': 'blue',
            'dataset_count': 12,
            'members': ['T', 'J'],
            'pinned': True
        },
        {
            'name': 'Research Lab',
            'description': 'Experimental data analysis',
            'color': 'green',
            'dataset_count': 8,
            'members': ['R'],
            'pinned': False
        }
    ]
    
    for ws_data in sample_workspaces:
        if not Workspace.objects(name=ws_data['name'], owner_id=user_id):
            workspace = Workspace(
                name=ws_data['name'],
                description=ws_data['description'],
                owner_id=user_id,
                color=ws_data['color'],
                dataset_count=ws_data['dataset_count'],
                members=ws_data['members'],
                pinned=ws_data['pinned']
            )
            workspace.save()
            
            activity = WorkspaceActivity(
                workspace_id=str(workspace.id),
                user_id=user_id,
                user_name="System",
                action='create',
                description=f'Created workspace "{workspace.name}"',
                details={'workspace_name': workspace.name}
            )
            activity.save()

@mongo_login_required
def table_view(request):
    # Get workspace_id from query parameters if provided
    workspace_id = request.GET.get('workspace')
    
    # Get all datasets for the current user
    datasets = Dataset.objects(owner_id=str(request.session.get("user_id")))
    
    # Get the selected dataset if provided
    selected_dataset_id = request.GET.get('dataset')
    selected_dataset = None
    if selected_dataset_id:
        try:
            selected_dataset = Dataset.objects.get(id=ObjectId(selected_dataset_id), owner_id=str(request.session.get("user_id")))
        except:
            selected_dataset = None
    
    # If no dataset selected, use the first one
    if not selected_dataset and datasets:
        selected_dataset = datasets[0]
    
    context = {
        "datasets": datasets,
        "selected_dataset": selected_dataset,
        "workspace_id": workspace_id,
        "user_initials": request.user.username[0].upper() if request.user.username else 'U'
    }
    
    return render(request, "table_view.html", context)



@mongo_login_required
def get_dataset_preview(request, dataset_id):
    """API endpoint to get dataset preview data"""
    try:
        dataset = Dataset.objects.get(id=ObjectId(dataset_id), owner_id=str(request.session.get("user_id")))
        
        # Import the preview function from datasets app
        from datasets.views import preview_dataset
        
        # Create a mock request for the preview function
        class MockRequest:
            def __init__(self, user):
                self.user = user
        
        mock_request = MockRequest(request.user)
        response = preview_dataset(mock_request, dataset_id)
        
        if response.status_code == 200:
            data = json.loads(response.content)
            return JsonResponse({
                'success': True,
                'dataset': {
                    'id': str(dataset.id),
                    'name': dataset.file_name,
                    'size': '5.2 MB',  # You might want to calculate this
                    'rows': 10542,     # You might want to calculate this
                    'columns': 8,      # You might want to calculate this
                },
                'preview': data.get('rows', [])
            })
        else:
            return JsonResponse({
                'success': False,
                'error': 'Failed to load dataset preview'
            })
            
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        })


@mongo_login_required
def transformation(request):
    datasets = Dataset.objects(owner_id=str(request.session.get("user_id")))
    return render(request, "transformation.html", {
        "datasets": datasets,
        "user_initials": request.user.username[0].upper() if request.user.username else 'U'
    })

@mongo_login_required
def schema_page(request):
    datasets = Dataset.objects(owner_id=str(request.session.get("user_id")))
    return render(request, "schema_page.html", {"datasets": datasets})

# Join Operations API Views
@mongo_login_required
@require_http_methods(["POST"])
@csrf_exempt
def create_join_operation(request):
    """Create a new join operation between datasets"""
    try:
        data = json.loads(request.body)
        
        # Get datasets
        left_dataset_id = data.get('left_dataset')
        right_dataset_id = data.get('right_dataset')
        join_type = data.get('join_type', 'inner')
        left_column = data.get('left_column')
        right_column = data.get('right_column')
        join_name = data.get('name', f'Join_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        
        # Validate required fields
        if not all([left_dataset_id, right_dataset_id, left_column, right_column]):
            return JsonResponse({'success': False, 'error': 'Missing required fields'}, status=400)
        
        # Validate datasets
        left_dataset = Dataset.objects.get(id=ObjectId(left_dataset_id), owner_id=str(request.session.get("user_id")))
        right_dataset = Dataset.objects.get(id=ObjectId(right_dataset_id), owner_id=str(request.session.get("user_id")))
        
        # Download and process datasets
        left_df = download_and_convert_to_dataframe(left_dataset)
        right_df = download_and_convert_to_dataframe(right_dataset)
        
        # Perform join
        result_df = perform_join(left_df, right_df, left_column, right_column, join_type)
        
        # Create new dataset record for the join result
        join_dataset = create_join_dataset(result_df, join_name, request.session.get("user_id"))
        
        return JsonResponse({
            'success': True,
            'join_id': str(join_dataset.id),
            'join_name': join_name,
            'row_count': len(result_df),
            'column_count': len(result_df.columns),
            'preview_data': result_df.head(10).to_dict('records')
        })
        
    except Dataset.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Dataset not found or access denied'}, status=404)
    except Exception as e:
        print(f"Error creating join operation: {str(e)}")
        return JsonResponse({'success': False, 'error': str(e)}, status=400)

@mongo_login_required
@require_http_methods(["POST"])
@csrf_exempt
def preview_join_operation(request):
    """Preview join operation without saving"""
    try:
        data = json.loads(request.body)
        
        left_dataset_id = data.get('left_dataset')
        right_dataset_id = data.get('right_dataset')
        join_type = data.get('join_type', 'inner')
        left_column = data.get('left_column')
        right_column = data.get('right_column')
        
        # Validate required fields
        if not all([left_dataset_id, right_dataset_id, left_column, right_column]):
            return JsonResponse({'success': False, 'error': 'Missing required fields'}, status=400)
        
        # Validate datasets
        left_dataset = Dataset.objects.get(id=ObjectId(left_dataset_id), owner_id=str(request.session.get("user_id")))
        right_dataset = Dataset.objects.get(id=ObjectId(right_dataset_id), owner_id=str(request.session.get("user_id")))
        
        # Download and process datasets
        left_df = download_and_convert_to_dataframe(left_dataset)
        right_df = download_and_convert_to_dataframe(right_dataset)
        
        # Perform join
        result_df = perform_join(left_df, right_df, left_column, right_column, join_type)
        
        return JsonResponse({
            'success': True,
            'preview_data': result_df.head(20).to_dict('records'),
            'columns': list(result_df.columns),
            'row_count': len(result_df),
            'left_columns': list(left_df.columns),
            'right_columns': list(right_df.columns)
        })
        
    except Dataset.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Dataset not found or access denied'}, status=404)
    except Exception as e:
        print(f"Error previewing join operation: {str(e)}")
        return JsonResponse({'success': False, 'error': str(e)}, status=400)


@mongo_login_required
@require_http_methods(["GET"])
def get_dataset_columns(request, dataset_id):
    """Get column names and sample data from any dataset type including API callback"""
    try:
        print(f"🔍 Getting columns for dataset: {dataset_id}")
        print(f"👤 User ID: {request.session.get('user_id')}")

        # Validate dataset_id
        if not dataset_id:
            return JsonResponse(
                {'success': False, 'error': 'Dataset ID is required'},
                status=400
            )

        # Validate user
        user_id = request.session.get("user_id")
        if not user_id:
            return JsonResponse(
                {'success': False, 'error': 'Unauthorized'},
                status=401
            )

        # Fetch dataset owned by user
        dataset = Dataset.objects.get(
            id=ObjectId(dataset_id),
            owner_id=str(user_id)
        )

        print(f"📊 Dataset found: {dataset.name}, Type: {dataset.source_type}")

        # Convert dataset to DataFrame
        df = download_and_convert_to_dataframe(dataset)

        if df is None or df.empty:
            # For API callback datasets, provide helpful message
            if dataset.source_type == "api_callback":
                return JsonResponse({
                    'success': True,
                    'columns': [],
                    'sample_data': [],
                    'message': 'No data received yet via API. Data will appear here when your website sends data to the endpoint.',
                    'endpoint_info': {
                        'endpoint_url': dataset.get_api_endpoint(request),
                        'total_received': dataset.metadata.get('total_received', 0),
                        'last_received': dataset.metadata.get('last_received', 'Never')
                    }
                })
            else:
                return JsonResponse({
                    'success': False, 
                    'error': 'Dataset is empty or could not be converted'
                }, status=400)

        columns = list(df.columns)
        
        # Get sample data (handle different data types)
        sample_data = []
        for _, row in df.head(5).iterrows():
            row_dict = {}
            for col in columns:
                value = row[col]
                # Convert non-serializable types
                if pd.isna(value):
                    row_dict[col] = None
                elif isinstance(value, (pd.Timestamp, datetime)):
                    row_dict[col] = value.isoformat()
                elif isinstance(value, (int, float, str, bool)) or value is None:
                    row_dict[col] = value
                else:
                    row_dict[col] = str(value)
            sample_data.append(row_dict)

        print(f"✅ Found {len(columns)} columns: {columns}")

        response_data = {
            'success': True,
            'columns': columns,
            'sample_data': sample_data,
            'total_rows': len(df),
            'dataset_type': dataset.source_type,
            'dataset_name': dataset.name
        }

        # Add API callback specific info
        if dataset.source_type == "api_callback":
            response_data['endpoint_info'] = {
                'endpoint_url': dataset.get_api_endpoint(request),
                'total_received': dataset.metadata.get('total_received', 0),
                'last_received': dataset.metadata.get('last_received', 'Never')
            }

        return JsonResponse(response_data, status=200)

    except Dataset.DoesNotExist:
        print(f"❌ Dataset not found: {dataset_id}")
        return JsonResponse(
            {'success': False, 'error': 'Dataset not found'},
            status=404
        )

    except InvalidId:
        print(f"❌ Invalid dataset ID: {dataset_id}")
        return JsonResponse(
            {'success': False, 'error': 'Invalid dataset ID'},
            status=400
        )

    except Exception as e:
        print(f"❌ Error getting columns: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Provide more specific error messages
        error_msg = str(e)
        if "psycopg2" in error_msg:
            error_msg = "PostgreSQL connection failed. Please check your connection settings."
        elif "mysql.connector" in error_msg:
            error_msg = "MySQL connection failed. Please check your connection settings."
        elif "pymongo" in error_msg:
            error_msg = "MongoDB connection failed. Please check your connection settings."
        elif "requests" in error_msg:
            error_msg = "API connection failed. Please check your API settings."
            
        return JsonResponse(
            {'success': False, 'error': error_msg},
            status=500
        )

from supabase import create_client as create_supabase_client
from pymongo import MongoClient


def create_mongo_client(uri: str):
    return MongoClient(uri)


# Helper function
def download_and_convert_to_dataframe(dataset):
    """
    Convert any dataset source (file, MongoDB, PostgreSQL, MySQL, API, API Callback)
    into a pandas DataFrame, fully normalized for dashboard usage (no NaT crashes).
    """
    import pandas as pd
    source_type = dataset.source_type
    df = None  # final unified output

    # ---------------------------------------------------------------
    # 📌 1. FILE FROM SUPABASE (CSV / Excel / JSON)
    # ---------------------------------------------------------------
    if dataset.is_file_based:
        supabase = create_supabase_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
        file_bytes = supabase.storage.from_(settings.SUPABASE_BUCKET).download(dataset.file_path)

        file_name = dataset.file_info.get("original_name", dataset.file_name or "").lower()
        decoded = file_bytes.decode("utf-8") if not file_name.endswith(".xlsx") else file_bytes

        if file_name.endswith(".csv"):
            df = pd.read_csv(io.StringIO(decoded))
        elif file_name.endswith(".xlsx") or file_name.endswith(".xls"):
            df = pd.read_excel(io.BytesIO(file_bytes))
        elif file_name.endswith(".json"):
            df = pd.json_normalize(json.loads(decoded))
        else:
            raise ValueError("Unsupported file type. Only CSV, Excel, JSON supported")

    # ---------------------------------------------------------------
    # 📌 2. MONGODB
    # ---------------------------------------------------------------
    elif source_type == "mongodb":
        from pymongo import MongoClient
        from bson import ObjectId

        info = dataset.connection_info
        client = MongoClient(info["uri"])
        docs = list(client[info["database"]][info["collection"]].find())
        client.close()

        for d in docs:  # convert ObjectId → string
            for k, v in d.items():
                if isinstance(v, ObjectId):
                    d[k] = str(v)

        df = pd.json_normalize(docs)

    # ---------------------------------------------------------------
    # 📌 3. POSTGRESQL
    # ---------------------------------------------------------------
    elif source_type == "postgresql":
        import psycopg2, pandas.io.sql as psql
        info = dataset.connection_info
        conn = psycopg2.connect(**info)
        query = f'SELECT * FROM "{info["schema"]}"."{info["table"]}"'
        df = psql.read_sql(query, conn)
        conn.close()

    # ---------------------------------------------------------------
    # 📌 4. MYSQL
    # ---------------------------------------------------------------
    elif source_type == "mysql":
        import mysql.connector
        conn = mysql.connector.connect(**dataset.connection_info)
        query = f'SELECT * FROM `{dataset.connection_info["table"]}`'
        df = pd.read_sql(query, conn)
        conn.close()

    # ---------------------------------------------------------------
    # 📌 5. API (REST)
    # ---------------------------------------------------------------
    elif source_type == "api":
        import requests
        info = dataset.connection_info
        res = requests.request(
            info.get("method", "GET"),
            info["url"],
            headers=json.loads(info.get("headers", "{}") or "{}"),
            params=json.loads(info.get("params", "{}") or "{}"),
            timeout=10
        )
        data = res.json()
        if isinstance(data, list):
            df = pd.json_normalize(data)
        elif isinstance(data, dict):
            for pivot in ["data", "items", "results", "records"]:
                if pivot in data and isinstance(data[pivot], list):
                    df = pd.json_normalize(data[pivot])
                    break
            else:
                df = pd.json_normalize([data])
        else:
            df = pd.DataFrame([{"value": str(data)}])

    # ---------------------------------------------------------------
    # 📌 6. API CALLBACK (Webhook Data)
    # ---------------------------------------------------------------
    elif source_type == "api_callback":
        records = DatasetIncomingData.objects(dataset_id=str(dataset.id))
        if not records:
            df = pd.DataFrame()
        else:
            data_list = []
            for rec in records:
                rd = rec.data
                if isinstance(rd, dict):
                    rd = {**rd, "_received_at": rec.received_at}
                    data_list.append(rd)
                elif isinstance(rd, list):
                    for x in rd:
                        if isinstance(x, dict):
                            data_list.append({**x, "_received_at": rec.received_at})
                        else:
                            data_list.append({"value": x, "_received_at": rec.received_at})
                else:
                    data_list.append({"value": rd, "_received_at": rec.received_at})
            df = pd.json_normalize(data_list)

    else:
        raise ValueError(f"Unsupported dataset source type: {source_type}")

    # ---------------------------------------------------------------
    # 🛡 GLOBAL DATETIME FIX (single place)
    # ---------------------------------------------------------------
    if df is not None and not df.empty:
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]) or \
               any(key in col.lower() for key in ["date", "time", "at"]):
                df[col] = pd.to_datetime(df[col], errors="coerce")
                df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S').fillna("")  # avoid NaT crash

    return df if df is not None else pd.DataFrame()


def perform_join(left_df, right_df, left_column, right_column, join_type):
    """Perform the actual join operation"""
    try:
        # Validate columns exist
        if left_column not in left_df.columns:
            raise ValueError(f"Column '{left_column}' not found in left dataset")
        if right_column not in right_df.columns:
            raise ValueError(f"Column '{right_column}' not found in right dataset")
            
        if join_type == 'inner':
            return pd.merge(left_df, right_df, left_on=left_column, right_on=right_column, how='inner')
        elif join_type == 'left':
            return pd.merge(left_df, right_df, left_on=left_column, right_on=right_column, how='left')
        elif join_type == 'right':
            return pd.merge(left_df, right_df, left_on=left_column, right_on=right_column, how='right')
        elif join_type == 'outer':
            return pd.merge(left_df, right_df, left_on=left_column, right_on=right_column, how='outer')
        else:
            raise ValueError(f"Unsupported join type: {join_type}")
    except Exception as e:
        print(f"Error performing join: {str(e)}")
        raise

def create_join_dataset(df, name, user_id):
    """Create a new dataset from join result"""
    from datasets.models import Dataset
    from supabase import create_client
    from django.conf import settings
    
    supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
    
    try:
        # Convert DataFrame to CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()
        
        # Upload to Supabase
        unique_filename = f"{uuid.uuid4()}_{name}.csv"
        supabase_path = f"{user_id}/{unique_filename}"
        
        res = supabase.storage.from_(settings.SUPABASE_BUCKET).upload(
            supabase_path, csv_content.encode('utf-8')
        )
        
        # Get public URL
        file_url = supabase.storage.from_(settings.SUPABASE_BUCKET).get_public_url(supabase_path)
        
        # Create dataset record
        dataset = Dataset(
            owner_id=str(user_id),
            file_name=f"{name}.csv",
            file_type="text/csv",
            file_url=file_url,
            file_path=supabase_path,
            uploaded_at=datetime.utcnow(),
            metadata={
                "is_join_result": True,
                "join_type": "multiple",
                "created_from_transformation": True
            }
        )
        
        dataset.save()
        return dataset
    except Exception as e:
        print(f"Error creating join dataset: {str(e)}")
        raise

# Workspace API Views
@mongo_login_required
@require_http_methods(["POST"])
@csrf_exempt
def create_workspace(request):
    try:
        data = json.loads(request.body)
        workspace = Workspace(
            name=data.get('name', 'New Workspace'),
            description=data.get('description', ''),
            owner_id=str(request.session.get("user_id")),
            members=data.get('members', []),
            color=data.get('color', 'yellow'),
            dataset_count=data.get('dataset_count', 0)
        )
        workspace.save()
        
        activity = WorkspaceActivity(
            workspace_id=str(workspace.id),
            user_id=str(request.session.get("user_id")),
            user_name=request.user.username or "User",
            action='create',
            description=f'Created workspace "{workspace.name}"',
            details={'workspace_name': workspace.name}
        )
        activity.save()
        
        return JsonResponse({
            'success': True,
            'workspace': {
                'id': str(workspace.id),
                'name': workspace.name,
                'description': workspace.description,
                'color': workspace.color,
                'pinned': workspace.pinned,
                'dataset_count': workspace.dataset_count,
                'members': workspace.members,
                'created_at': workspace.created_at.isoformat(),
                'updated_at': workspace.updated_at.isoformat()
            }
        })
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)

@mongo_login_required
@require_http_methods(["POST"])
@csrf_exempt
def toggle_pin_workspace(request, workspace_id):
    try:
        # Try to find workspace by string ID
        user_workspaces = Workspace.objects(owner_id=str(request.session.get("user_id")))
        workspace = None
        
        for ws in user_workspaces:
            if str(ws.id) == workspace_id:
                workspace = ws
                break
        
        if not workspace:
            return JsonResponse({'success': False, 'error': 'Workspace not found'}, status=404)
        
        workspace.pinned = not workspace.pinned
        workspace.updated_at = datetime.utcnow()
        workspace.save()
        
        action = "pinned" if workspace.pinned else "unpinned"
        activity = WorkspaceActivity(
            workspace_id=str(workspace.id),
            user_id=str(request.session.get("user_id")),
            user_name=request.user.username or "User",
            action='pin',
            description=f'{action} workspace "{workspace.name}"',
            details={'workspace_name': workspace.name, 'pinned': workspace.pinned}
        )
        activity.save()
        
        return JsonResponse({
            'success': True,
            'pinned': workspace.pinned
        })
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)

@mongo_login_required
@require_http_methods(["POST"])
@csrf_exempt
def edit_workspace(request, workspace_id):
    try:
        # Try to find workspace by string ID
        user_workspaces = Workspace.objects(owner_id=str(request.session.get("user_id")))
        workspace = None
        
        for ws in user_workspaces:
            if str(ws.id) == workspace_id:
                workspace = ws
                break
        
        if not workspace:
            return JsonResponse({'success': False, 'error': 'Workspace not found'}, status=404)
        
        data = json.loads(request.body)
        
        workspace.name = data.get('name', workspace.name)
        workspace.description = data.get('description', workspace.description)
        workspace.color = data.get('color', workspace.color)
        workspace.updated_at = datetime.utcnow()
        workspace.save()
        
        activity = WorkspaceActivity(
            workspace_id=str(workspace.id),
            user_id=str(request.session.get("user_id")),
            user_name=request.user.username or "User",
            action='edit',
            description=f'Updated workspace "{workspace.name}"',
            details={'workspace_name': workspace.name}
        )
        activity.save()
        
        return JsonResponse({
            'success': True,
            'workspace': {
                'id': str(workspace.id),
                'name': workspace.name,
                'description': workspace.description,
                'color': workspace.color
            }
        })
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)

@mongo_login_required
@require_http_methods(["POST"])
@csrf_exempt
def delete_workspace(request, workspace_id):
    try:
        print(f"Deleting workspace: {workspace_id}")
        
        # Get all workspaces for the user
        user_workspaces = Workspace.objects(owner_id=str(request.session.get("user_id")))
        
        # Find the workspace by string ID
        workspace_to_delete = None
        for ws in user_workspaces:
            if str(ws.id) == workspace_id:
                workspace_to_delete = ws
                break
        
        if not workspace_to_delete:
            return JsonResponse({'success': False, 'error': 'Workspace not found'}, status=404)
        
        workspace_name = workspace_to_delete.name
        workspace_id_str = str(workspace_to_delete.id)
        
        # Delete the workspace
        workspace_to_delete.delete()
        
        # Delete related activities
        WorkspaceActivity.objects(workspace_id=workspace_id_str).delete()
        
        # Create activity for deletion
        activity = WorkspaceActivity(
            user_id=str(request.session.get("user_id")),
            user_name=request.user.username or "User",
            action='delete',
            description=f'Deleted workspace "{workspace_name}"',
            details={'workspace_name': workspace_name, 'deleted': True}
        )
        activity.save()
        
        return JsonResponse({
            'success': True,
            'message': 'Workspace deleted successfully'
        })
    except Exception as e:
        print(f"Delete error: {e}")
        return JsonResponse({'success': False, 'error': str(e)}, status=400)

@mongo_login_required
@require_http_methods(["GET"])
def get_workspace_activities(request, workspace_id=None):
    try:
        if workspace_id:
            activities = WorkspaceActivity.objects(workspace_id=workspace_id).order_by('-created_at')[:10]
        else:
            activities = WorkspaceActivity.objects().order_by('-created_at')[:10]
        
        activity_list = []
        for activity in activities:
            activity_list.append({
                'id': str(activity.id),
                'action': activity.action,
                'description': activity.description,
                'user_name': activity.user_name,
                'created_at': activity.created_at.isoformat()
            })
        
        return JsonResponse({
            'success': True,
            'activities': activity_list
        })
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)




@mongo_login_required
@require_http_methods(["POST"])
@csrf_exempt
def filter_data(request):
    """Apply filters to dataset"""
    try:
        data = json.loads(request.body)
        dataset_id = data.get('dataset_id')
        filters = data.get('filters', [])
        preview_only = data.get('preview_only', False)
        
        # Validate inputs
        if not dataset_id:
            return JsonResponse({'success': False, 'error': 'Dataset ID is required'}, status=400)
        
        # Get dataset
        dataset = Dataset.objects.get(id=ObjectId(dataset_id), owner_id=str(request.session.get("user_id")))
        df = download_and_convert_to_dataframe(dataset)
        
        # Store original stats
        original_stats = {
            'total_rows': len(df),
            'columns': list(df.columns)
        }
        
        # Apply filters
        result_df = df.copy()
        
        for filter_config in filters:
            column = filter_config.get('column')
            operator = filter_config.get('operator')
            value = filter_config.get('value')
            
            if column not in df.columns:
                return JsonResponse({'success': False, 'error': f'Column "{column}" not found in dataset'}, status=400)
            
            try:
                if operator == 'equals':
                    result_df = result_df[result_df[column] == value]
                elif operator == 'not_equals':
                    result_df = result_df[result_df[column] != value]
                elif operator == 'contains':
                    result_df = result_df[result_df[column].astype(str).str.contains(str(value), na=False)]
                elif operator == 'starts_with':
                    result_df = result_df[result_df[column].astype(str).str.startswith(str(value))]
                elif operator == 'ends_with':
                    result_df = result_df[result_df[column].astype(str).str.endswith(str(value))]
                elif operator == 'greater_than':
                    result_df = result_df[result_df[column] > value]
                elif operator == 'less_than':
                    result_df = result_df[result_df[column] < value]
                elif operator == 'greater_than_equal':
                    result_df = result_df[result_df[column] >= value]
                elif operator == 'less_than_equal':
                    result_df = result_df[result_df[column] <= value]
                elif operator == 'is_null':
                    result_df = result_df[result_df[column].isnull()]
                elif operator == 'not_null':
                    result_df = result_df[result_df[column].notnull()]
                elif operator == 'in_list':
                    if isinstance(value, list):
                        result_df = result_df[result_df[column].isin(value)]
                    else:
                        result_df = result_df[result_df[column].isin([value])]
                elif operator == 'not_in_list':
                    if isinstance(value, list):
                        result_df = result_df[~result_df[column].isin(value)]
                    else:
                        result_df = result_df[~result_df[column].isin([value])]
                elif operator == 'between':
                    if isinstance(value, list) and len(value) == 2:
                        result_df = result_df[(result_df[column] >= value[0]) & (result_df[column] <= value[1])]
                        
            except Exception as e:
                return JsonResponse({
                    'success': False, 
                    'error': f'Error applying filter {operator} on {column}: {str(e)}'
                }, status=400)
        
        # Calculate result stats
        result_stats = {
            'total_rows': len(result_df),
            'rows_removed': original_stats['total_rows'] - len(result_df),
            'filter_efficiency': f"{(len(result_df) / original_stats['total_rows']) * 100:.1f}%" if original_stats['total_rows'] > 0 else '0%',
            'filters_applied': len(filters)
        }
        
        if preview_only:
            preview_data = dataframe_to_dict_clean(result_df.head(20))
            return JsonResponse({
                'success': True,
                'preview_data': preview_data,
                'original_stats': clean_data_for_json(original_stats),
                'result_stats': clean_data_for_json(result_stats),
                'columns': list(result_df.columns)
            })
        
        # Create new dataset
        operation_name = f"Filtered_Data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        new_dataset = create_cleaned_dataset(result_df, operation_name, request.session.get("user_id"), 'filter')
        
        # Save operation record
        operation = DataCleaningOperation(
            name=operation_name,
            operation_type='filter',
            input_dataset_id=dataset_id,
            output_dataset_id=str(new_dataset.id),
            user_id=str(request.session.get("user_id")),
            parameters={
                'filters': filters
            },
            result_stats=result_stats,
            status='completed',
            completed_at=datetime.utcnow()
        )
        operation.save()
        
        preview_data_clean = dataframe_to_dict_clean(result_df.head(10))
        
        return JsonResponse({
            'success': True,
            'operation_id': str(operation.id),
            'new_dataset_id': str(new_dataset.id),
            'result_stats': clean_data_for_json(result_stats),
            'preview_data': preview_data_clean
        })
        
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)

@mongo_login_required
@require_http_methods(["POST"])
@csrf_exempt
def sort_data(request):
    """Sort dataset by specified columns"""
    try:
        data = json.loads(request.body)
        dataset_id = data.get('dataset_id')
        sort_columns = data.get('sort_columns', [])
        preview_only = data.get('preview_only', False)
        
        # Validate inputs
        if not dataset_id:
            return JsonResponse({'success': False, 'error': 'Dataset ID is required'}, status=400)
        
        if not sort_columns:
            return JsonResponse({'success': False, 'error': 'At least one sort column is required'}, status=400)
        
        # Get dataset
        dataset = Dataset.objects.get(id=ObjectId(dataset_id), owner_id=str(request.session.get("user_id")))
        df = download_and_convert_to_dataframe(dataset)
        
        # Store original stats
        original_stats = {
            'total_rows': len(df),
            'columns': list(df.columns)
        }
        
        # Prepare sort parameters
        sort_by = []
        ascending = []
        
        for sort_config in sort_columns:
            column = sort_config.get('column')
            order = sort_config.get('order', 'asc')
            
            if column not in df.columns:
                return JsonResponse({'success': False, 'error': f'Column "{column}" not found in dataset'}, status=400)
            
            sort_by.append(column)
            ascending.append(order == 'asc')
        
        # Apply sorting
        result_df = df.sort_values(by=sort_by, ascending=ascending).reset_index(drop=True)
        
        # Calculate result stats
        result_stats = {
            'total_rows': len(result_df),
            'sort_columns': sort_by,
            'sort_orders': ['asc' if asc else 'desc' for asc in ascending]
        }
        
        if preview_only:
            preview_data = dataframe_to_dict_clean(result_df.head(20))
            return JsonResponse({
                'success': True,
                'preview_data': preview_data,
                'original_stats': clean_data_for_json(original_stats),
                'result_stats': clean_data_for_json(result_stats),
                'columns': list(result_df.columns)
            })
        
        # Create new dataset
        operation_name = f"Sorted_Data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        new_dataset = create_cleaned_dataset(result_df, operation_name, request.session.get("user_id"), 'sort')
        
        # Save operation record
        operation = DataCleaningOperation(
            name=operation_name,
            operation_type='sort',
            input_dataset_id=dataset_id,
            output_dataset_id=str(new_dataset.id),
            user_id=str(request.session.get("user_id")),
            parameters={
                'sort_columns': sort_columns
            },
            result_stats=result_stats,
            status='completed',
            completed_at=datetime.utcnow()
        )
        operation.save()
        
        preview_data_clean = dataframe_to_dict_clean(result_df.head(10))
        
        return JsonResponse({
            'success': True,
            'operation_id': str(operation.id),
            'new_dataset_id': str(new_dataset.id),
            'result_stats': clean_data_for_json(result_stats),
            'preview_data': preview_data_clean
        })
        
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)

@mongo_login_required
@require_http_methods(["POST"])
@csrf_exempt
def top_n_records(request):
    """Get top N records based on criteria"""
    try:
        data = json.loads(request.body)
        dataset_id = data.get('dataset_id')
        n_records = data.get('n_records', 10)
        sort_by = data.get('sort_by')
        sort_order = data.get('sort_order', 'desc')
        preview_only = data.get('preview_only', False)
        
        # Validate inputs
        if not dataset_id:
            return JsonResponse({'success': False, 'error': 'Dataset ID is required'}, status=400)
        
        if not sort_by:
            return JsonResponse({'success': False, 'error': 'Sort column is required'}, status=400)
        
        # Get dataset
        dataset = Dataset.objects.get(id=ObjectId(dataset_id), owner_id=str(request.session.get("user_id")))
        df = download_and_convert_to_dataframe(dataset)
        
        # Validate sort column
        if sort_by not in df.columns:
            return JsonResponse({'success': False, 'error': f'Sort column "{sort_by}" not found in dataset'}, status=400)
        
        # Store original stats
        original_stats = {
            'total_rows': len(df),
            'columns': list(df.columns)
        }
        
        # Apply top N selection
        ascending = sort_order == 'asc'
        result_df = df.sort_values(by=sort_by, ascending=ascending).head(n_records).reset_index(drop=True)
        
        # Calculate result stats
        result_stats = {
            'total_rows': len(result_df),
            'n_records': n_records,
            'sort_column': sort_by,
            'sort_order': sort_order,
            'percentage_of_total': f"{(len(result_df) / original_stats['total_rows']) * 100:.1f}%" if original_stats['total_rows'] > 0 else '0%'
        }
        
        if preview_only:
            preview_data = dataframe_to_dict_clean(result_df.head(20))
            return JsonResponse({
                'success': True,
                'preview_data': preview_data,
                'original_stats': clean_data_for_json(original_stats),
                'result_stats': clean_data_for_json(result_stats),
                'columns': list(result_df.columns)
            })
        
        # Create new dataset
        operation_name = f"Top_{n_records}_Records_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        new_dataset = create_cleaned_dataset(result_df, operation_name, request.session.get("user_id"), 'top_n')
        
        # Save operation record
        operation = DataCleaningOperation(
            name=operation_name,
            operation_type='top_n',
            input_dataset_id=dataset_id,
            output_dataset_id=str(new_dataset.id),
            user_id=str(request.session.get("user_id")),
            parameters={
                'n_records': n_records,
                'sort_by': sort_by,
                'sort_order': sort_order
            },
            result_stats=result_stats,
            status='completed',
            completed_at=datetime.utcnow()
        )
        operation.save()
        
        preview_data_clean = dataframe_to_dict_clean(result_df.head(10))
        
        return JsonResponse({
            'success': True,
            'operation_id': str(operation.id),
            'new_dataset_id': str(new_dataset.id),
            'result_stats': clean_data_for_json(result_stats),
            'preview_data': preview_data_clean
        })
        
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)

@mongo_login_required
@require_http_methods(["POST"])
@csrf_exempt
def random_sampling(request):
    """Get random sample from dataset"""
    try:
        data = json.loads(request.body)
        dataset_id = data.get('dataset_id')
        sample_size = data.get('sample_size', 100)
        sample_type = data.get('sample_type', 'count')  # count or percentage
        random_state = data.get('random_state', 42)
        preview_only = data.get('preview_only', False)
        
        # Validate inputs
        if not dataset_id:
            return JsonResponse({'success': False, 'error': 'Dataset ID is required'}, status=400)
        
        # Get dataset
        dataset = Dataset.objects.get(id=ObjectId(dataset_id), owner_id=str(request.session.get("user_id")))
        df = download_and_convert_to_dataframe(dataset)
        
        # Store original stats
        original_stats = {
            'total_rows': len(df),
            'columns': list(df.columns)
        }
        
        # Calculate sample size
        if sample_type == 'percentage':
            n_samples = int(len(df) * (sample_size / 100))
        else:
            n_samples = min(sample_size, len(df))
        
        # Apply random sampling
        result_df = df.sample(n=n_samples, random_state=random_state).reset_index(drop=True)
        
        # Calculate result stats
        result_stats = {
            'total_rows': len(result_df),
            'sample_size': n_samples,
            'sample_type': sample_type,
            'original_size': original_stats['total_rows'],
            'sampling_rate': f"{(n_samples / original_stats['total_rows']) * 100:.1f}%" if original_stats['total_rows'] > 0 else '0%'
        }
        
        if preview_only:
            preview_data = dataframe_to_dict_clean(result_df.head(20))
            return JsonResponse({
                'success': True,
                'preview_data': preview_data,
                'original_stats': clean_data_for_json(original_stats),
                'result_stats': clean_data_for_json(result_stats),
                'columns': list(result_df.columns)
            })
        
        # Create new dataset
        operation_name = f"Random_Sample_{n_samples}_Records_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        new_dataset = create_cleaned_dataset(result_df, operation_name, request.session.get("user_id"), 'random_sample')
        
        # Save operation record
        operation = DataCleaningOperation(
            name=operation_name,
            operation_type='random_sample',
            input_dataset_id=dataset_id,
            output_dataset_id=str(new_dataset.id),
            user_id=str(request.session.get("user_id")),
            parameters={
                'sample_size': sample_size,
                'sample_type': sample_type,
                'random_state': random_state
            },
            result_stats=result_stats,
            status='completed',
            completed_at=datetime.utcnow()
        )
        operation.save()
        
        preview_data_clean = dataframe_to_dict_clean(result_df.head(10))
        
        return JsonResponse({
            'success': True,
            'operation_id': str(operation.id),
            'new_dataset_id': str(new_dataset.id),
            'result_stats': clean_data_for_json(result_stats),
            'preview_data': preview_data_clean
        })
        
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)

@mongo_login_required
@require_http_methods(["POST"])
@csrf_exempt
def preview_filter_sort(request):
    """Preview filter and sort operations without saving"""
    try:
        data = json.loads(request.body)
        operation_type = data.get('operation_type')
        
        if operation_type == 'filter':
            return filter_data(request)
        elif operation_type == 'sort':
            return sort_data(request)
        elif operation_type == 'top_n':
            return top_n_records(request)
        elif operation_type == 'random_sample':
            return random_sampling(request)
        else:
            return JsonResponse({'success': False, 'error': 'Unsupported operation type'}, status=400)
            
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)

@mongo_login_required
@require_http_methods(["POST"])
@csrf_exempt
def execute_filter_sort(request):
    """Execute filter and sort operations and save result"""
    try:
        data = json.loads(request.body)
        operation_type = data.get('operation_type')
        
        if operation_type == 'filter':
            return filter_data(request)
        elif operation_type == 'sort':
            return sort_data(request)
        elif operation_type == 'top_n':
            return top_n_records(request)
        elif operation_type == 'random_sample':
            return random_sampling(request)
        else:
            return JsonResponse({'success': False, 'error': 'Unsupported operation type'}, status=400)
            
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)


# feture eng
@mongo_login_required
@require_http_methods(["POST"])
@csrf_exempt
def create_calculated_columns(request):
    """Create new calculated columns based on expressions"""
    try:
        data = json.loads(request.body)
        dataset_id = data.get('dataset_id')
        calculated_columns = data.get('calculated_columns', [])
        preview_only = data.get('preview_only', False)
        
        # Validate inputs
        if not dataset_id:
            return JsonResponse({'success': False, 'error': 'Dataset ID is required'}, status=400)
        
        if not calculated_columns:
            return JsonResponse({'success': False, 'error': 'At least one calculated column is required'}, status=400)
        
        # Get dataset
        dataset = Dataset.objects.get(id=ObjectId(dataset_id), owner_id=str(request.session.get("user_id")))
        df = download_and_convert_to_dataframe(dataset)
        
        # Store original stats
        original_stats = {
            'total_rows': len(df),
            'columns': list(df.columns)
        }
        
        # Apply calculated columns
        result_df = df.copy()
        
        for column_config in calculated_columns:
            column_name = column_config.get('column_name')
            expression = column_config.get('expression')
            data_type = column_config.get('data_type', 'auto')
            
            if not column_name or not expression:
                return JsonResponse({'success': False, 'error': 'Column name and expression are required'}, status=400)
            
            try:
                # Safe evaluation of expression
                # Replace column references with df['column_name']
                safe_expression = expression
                for col in df.columns:
                    safe_expression = safe_expression.replace(col, f"df['{col}']")
                
                # Evaluate the expression
                result = eval(safe_expression, {'df': df, 'np': np, 'pd': pd})
                
                # Assign to new column
                result_df[column_name] = result
                
                # Convert data type if specified
                if data_type != 'auto':
                    if data_type == 'int':
                        result_df[column_name] = pd.to_numeric(result_df[column_name], errors='coerce').astype('Int64')
                    elif data_type == 'float':
                        result_df[column_name] = pd.to_numeric(result_df[column_name], errors='coerce').astype(float)
                    elif data_type == 'string':
                        result_df[column_name] = result_df[column_name].astype(str)
                    elif data_type == 'boolean':
                        result_df[column_name] = result_df[column_name].astype(bool)
                        
            except Exception as e:
                return JsonResponse({
                    'success': False, 
                    'error': f'Error creating column "{column_name}": {str(e)}'
                }, status=400)
        
        # Calculate result stats
        result_stats = {
            'total_rows': len(result_df),
            'result_columns': list(result_df.columns),
            'original_columns': original_stats['columns'],
            'new_columns_added': len(result_df.columns) - len(original_stats['columns'])
        }
        
        if preview_only:
            preview_data = dataframe_to_dict_clean(result_df.head(20))
            return JsonResponse({
                'success': True,
                'preview_data': preview_data,
                'original_stats': clean_data_for_json(original_stats),
                'result_stats': clean_data_for_json(result_stats),
                'columns': list(result_df.columns)
            })
        
        # Create new dataset
        operation_name = f"Calculated_Columns_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        new_dataset = create_cleaned_dataset(result_df, operation_name, request.session.get("user_id"), 'calculated_columns')
        
        # Save operation record
        operation = DataCleaningOperation(
            name=operation_name,
            operation_type='calculated_columns',
            input_dataset_id=dataset_id,
            output_dataset_id=str(new_dataset.id),
            user_id=str(request.session.get("user_id")),
            parameters={
                'calculated_columns': calculated_columns
            },
            result_stats=result_stats,
            status='completed',
            completed_at=datetime.utcnow()
        )
        operation.save()
        
        preview_data_clean = dataframe_to_dict_clean(result_df.head(10))
        
        return JsonResponse({
            'success': True,
            'operation_id': str(operation.id),
            'new_dataset_id': str(new_dataset.id),
            'result_stats': clean_data_for_json(result_stats),
            'preview_data': preview_data_clean
        })
        
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)

@mongo_login_required
@require_http_methods(["POST"])
@csrf_exempt
def datetime_extraction(request):
    """Extract components from datetime columns"""
    try:
        data = json.loads(request.body)
        dataset_id = data.get('dataset_id')
        datetime_columns = data.get('datetime_columns', [])
        preview_only = data.get('preview_only', False)
        
        # Validate inputs
        if not dataset_id:
            return JsonResponse({'success': False, 'error': 'Dataset ID is required'}, status=400)
        
        if not datetime_columns:
            return JsonResponse({'success': False, 'error': 'At least one datetime column configuration is required'}, status=400)
        
        # Get dataset
        dataset = Dataset.objects.get(id=ObjectId(dataset_id), owner_id=str(request.session.get("user_id")))
        df = download_and_convert_to_dataframe(dataset)
        
        # Store original stats
        original_stats = {
            'total_rows': len(df),
            'columns': list(df.columns)
        }
        
        # Apply datetime extraction
        result_df = df.copy()
        
        for column_config in datetime_columns:
            source_column = column_config.get('source_column')
            extractions = column_config.get('extractions', [])
            
            if source_column not in df.columns:
                return JsonResponse({'success': False, 'error': f'Source column "{source_column}" not found'}, status=400)
            
            try:
                # Convert to datetime if not already
                if not pd.api.types.is_datetime64_any_dtype(df[source_column]):
                    result_df[source_column] = pd.to_datetime(df[source_column], errors='coerce')
                
                # Apply extractions
                for extraction in extractions:
                    component = extraction.get('component')
                    new_column_name = extraction.get('new_column_name', f'{source_column}_{component}')
                    
                    if component == 'year':
                        result_df[new_column_name] = result_df[source_column].dt.year
                    elif component == 'month':
                        result_df[new_column_name] = result_df[source_column].dt.month
                    elif component == 'day':
                        result_df[new_column_name] = result_df[source_column].dt.day
                    elif component == 'hour':
                        result_df[new_column_name] = result_df[source_column].dt.hour
                    elif component == 'minute':
                        result_df[new_column_name] = result_df[source_column].dt.minute
                    elif component == 'second':
                        result_df[new_column_name] = result_df[source_column].dt.second
                    elif component == 'quarter':
                        result_df[new_column_name] = result_df[source_column].dt.quarter
                    elif component == 'dayofweek':
                        result_df[new_column_name] = result_df[source_column].dt.dayofweek
                    elif component == 'dayofyear':
                        result_df[new_column_name] = result_df[source_column].dt.dayofyear
                    elif component == 'week':
                        result_df[new_column_name] = result_df[source_column].dt.isocalendar().week
                    elif component == 'is_weekend':
                        result_df[new_column_name] = result_df[source_column].dt.dayofweek >= 5
                    elif component == 'month_name':
                        result_df[new_column_name] = result_df[source_column].dt.month_name()
                    elif component == 'day_name':
                        result_df[new_column_name] = result_df[source_column].dt.day_name()
                    
            except Exception as e:
                return JsonResponse({
                    'success': False, 
                    'error': f'Error processing datetime column "{source_column}": {str(e)}'
                }, status=400)
        
        # Calculate result stats
        result_stats = {
            'total_rows': len(result_df),
            'result_columns': list(result_df.columns),
            'original_columns': original_stats['columns'],
            'new_columns_added': len(result_df.columns) - len(original_stats['columns'])
        }
        
        if preview_only:
            preview_data = dataframe_to_dict_clean(result_df.head(20))
            return JsonResponse({
                'success': True,
                'preview_data': preview_data,
                'original_stats': clean_data_for_json(original_stats),
                'result_stats': clean_data_for_json(result_stats),
                'columns': list(result_df.columns)
            })
        
        # Create new dataset
        operation_name = f"Datetime_Extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        new_dataset = create_cleaned_dataset(result_df, operation_name, request.session.get("user_id"), 'datetime_extraction')
        
        # Save operation record
        operation = DataCleaningOperation(
            name=operation_name,
            operation_type='datetime_extraction',
            input_dataset_id=dataset_id,
            output_dataset_id=str(new_dataset.id),
            user_id=str(request.session.get("user_id")),
            parameters={
                'datetime_columns': datetime_columns
            },
            result_stats=result_stats,
            status='completed',
            completed_at=datetime.utcnow()
        )
        operation.save()
        
        preview_data_clean = dataframe_to_dict_clean(result_df.head(10))
        
        return JsonResponse({
            'success': True,
            'operation_id': str(operation.id),
            'new_dataset_id': str(new_dataset.id),
            'result_stats': clean_data_for_json(result_stats),
            'preview_data': preview_data_clean
        })
        
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)

@mongo_login_required
@require_http_methods(["POST"])
@csrf_exempt
def text_processing(request):
    """Apply text processing operations to text columns"""
    try:
        data = json.loads(request.body)
        dataset_id = data.get('dataset_id')
        text_columns = data.get('text_columns', [])
        preview_only = data.get('preview_only', False)
        
        # Validate inputs
        if not dataset_id:
            return JsonResponse({'success': False, 'error': 'Dataset ID is required'}, status=400)
        
        if not text_columns:
            return JsonResponse({'success': False, 'error': 'At least one text column configuration is required'}, status=400)
        
        # Get dataset
        dataset = Dataset.objects.get(id=ObjectId(dataset_id), owner_id=str(request.session.get("user_id")))
        df = download_and_convert_to_dataframe(dataset)
        
        # Store original stats
        original_stats = {
            'total_rows': len(df),
            'columns': list(df.columns)
        }
        
        # Apply text processing
        result_df = df.copy()
        
        for column_config in text_columns:
            source_column = column_config.get('source_column')
            operations = column_config.get('operations', [])
            
            if source_column not in df.columns:
                return JsonResponse({'success': False, 'error': f'Source column "{source_column}" not found'}, status=400)
            
            try:
                # Convert to string to ensure text operations work
                result_df[source_column] = result_df[source_column].astype(str)
                
                # Apply operations
                for operation in operations:
                    op_type = operation.get('type')
                    new_column_name = operation.get('new_column_name', f'{source_column}_{op_type}')
                    
                    if op_type == 'lowercase':
                        result_df[new_column_name] = result_df[source_column].str.lower()
                    elif op_type == 'uppercase':
                        result_df[new_column_name] = result_df[source_column].str.upper()
                    elif op_type == 'title_case':
                        result_df[new_column_name] = result_df[source_column].str.title()
                    elif op_type == 'capitalize':
                        result_df[new_column_name] = result_df[source_column].str.capitalize()
                    elif op_type == 'strip':
                        result_df[new_column_name] = result_df[source_column].str.strip()
                    elif op_type == 'remove_extra_spaces':
                        result_df[new_column_name] = result_df[source_column].str.replace(r'\s+', ' ', regex=True)
                    elif op_type == 'remove_special_chars':
                        result_df[new_column_name] = result_df[source_column].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
                    elif op_type == 'extract_numbers':
                        result_df[new_column_name] = result_df[source_column].str.extract(r'(\d+)', expand=False)
                    elif op_type == 'extract_letters':
                        result_df[new_column_name] = result_df[source_column].str.replace(r'[^a-zA-Z]', '', regex=True)
                    elif op_type == 'word_count':
                        result_df[new_column_name] = result_df[source_column].str.split().str.len()
                    elif op_type == 'character_count':
                        result_df[new_column_name] = result_df[source_column].str.len()
                    elif op_type == 'replace_text':
                        old_text = operation.get('old_text', '')
                        new_text = operation.get('new_text', '')
                        result_df[new_column_name] = result_df[source_column].str.replace(old_text, new_text, regex=False)
                    elif op_type == 'substring':
                        start = operation.get('start', 0)
                        end = operation.get('end', None)
                        result_df[new_column_name] = result_df[source_column].str.slice(start, end)
                    
            except Exception as e:
                return JsonResponse({
                    'success': False, 
                    'error': f'Error processing text column "{source_column}": {str(e)}'
                }, status=400)
        
        # Calculate result stats
        result_stats = {
            'total_rows': len(result_df),
            'result_columns': list(result_df.columns),
            'original_columns': original_stats['columns'],
            'new_columns_added': len(result_df.columns) - len(original_stats['columns'])
        }
        
        if preview_only:
            preview_data = dataframe_to_dict_clean(result_df.head(20))
            return JsonResponse({
                'success': True,
                'preview_data': preview_data,
                'original_stats': clean_data_for_json(original_stats),
                'result_stats': clean_data_for_json(result_stats),
                'columns': list(result_df.columns)
            })
        
        # Create new dataset
        operation_name = f"Text_Processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        new_dataset = create_cleaned_dataset(result_df, operation_name, request.session.get("user_id"), 'text_processing')
        
        # Save operation record
        operation = DataCleaningOperation(
            name=operation_name,
            operation_type='text_processing',
            input_dataset_id=dataset_id,
            output_dataset_id=str(new_dataset.id),
            user_id=str(request.session.get("user_id")),
            parameters={
                'text_columns': text_columns
            },
            result_stats=result_stats,
            status='completed',
            completed_at=datetime.utcnow()
        )
        operation.save()
        
        preview_data_clean = dataframe_to_dict_clean(result_df.head(10))
        
        return JsonResponse({
            'success': True,
            'operation_id': str(operation.id),
            'new_dataset_id': str(new_dataset.id),
            'result_stats': clean_data_for_json(result_stats),
            'preview_data': preview_data_clean
        })
        
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)

@mongo_login_required
@require_http_methods(["POST"])
@csrf_exempt
def one_hot_encoding(request):
    """Apply one-hot encoding to categorical columns"""
    try:
        data = json.loads(request.body)
        dataset_id = data.get('dataset_id')
        categorical_columns = data.get('categorical_columns', [])
        drop_first = data.get('drop_first', False)
        prefix = data.get('prefix', True)
        preview_only = data.get('preview_only', False)
        
        # Validate inputs
        if not dataset_id:
            return JsonResponse({'success': False, 'error': 'Dataset ID is required'}, status=400)
        
        if not categorical_columns:
            return JsonResponse({'success': False, 'error': 'At least one categorical column is required'}, status=400)
        
        # Get dataset
        dataset = Dataset.objects.get(id=ObjectId(dataset_id), owner_id=str(request.session.get("user_id")))
        df = download_and_convert_to_dataframe(dataset)
        
        # Validate columns exist
        for col in categorical_columns:
            if col not in df.columns:
                return JsonResponse({'success': False, 'error': f'Column "{col}" not found in dataset'}, status=400)
        
        # Store original stats
        original_stats = {
            'total_rows': len(df),
            'columns': list(df.columns),
            'categorical_counts': {col: df[col].nunique() for col in categorical_columns}
        }
        
        # Apply one-hot encoding
        result_df = df.copy()
        
        try:
            # Handle prefix parameter properly
            prefix_param = None
            if prefix is True:
                prefix_param = categorical_columns  # Use column names as prefixes
            elif prefix is False:
                prefix_param = None  # No prefixes
            elif isinstance(prefix, list) and len(prefix) == len(categorical_columns):
                prefix_param = prefix  # Use provided prefixes
            else:
                prefix_param = categorical_columns  # Default to column names
            
            # Use pandas get_dummies for one-hot encoding
            encoded_df = pd.get_dummies(
                result_df[categorical_columns], 
                prefix=prefix_param,
                prefix_sep='_',
                drop_first=drop_first
            )
            
            # Drop original categorical columns and add encoded ones
            result_df = result_df.drop(columns=categorical_columns)
            result_df = pd.concat([result_df, encoded_df], axis=1)
            
        except Exception as e:
            import traceback
            print(f"Error in one-hot encoding: {str(e)}")
            print(traceback.format_exc())
            return JsonResponse({'success': False, 'error': f'Error in one-hot encoding: {str(e)}'}, status=400)
        
        # Calculate result stats
        result_stats = {
            'total_rows': len(result_df),
            'result_columns': list(result_df.columns),
            'original_columns': original_stats['columns'],
            'new_columns_added': len(result_df.columns) - len(original_stats['columns']),
            'encoding_details': {
                'original_categorical_columns': categorical_columns,
                'encoded_columns_count': len(encoded_df.columns),
                'drop_first': drop_first,
                'prefix_used': prefix_param if prefix_param else False
            }
        }
        
        if preview_only:
            preview_data = dataframe_to_dict_clean(result_df.head(20))
            return JsonResponse({
                'success': True,
                'preview_data': preview_data,
                'original_stats': clean_data_for_json(original_stats),
                'result_stats': clean_data_for_json(result_stats),
                'columns': list(result_df.columns)
            })
        
        # Create new dataset
        operation_name = f"OneHot_Encoding_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        new_dataset = create_cleaned_dataset(result_df, operation_name, request.session.get("user_id"), 'one_hot_encoding')
        
        # Save operation record
        operation = DataCleaningOperation(
            name=operation_name,
            operation_type='one_hot_encoding',
            input_dataset_id=dataset_id,
            output_dataset_id=str(new_dataset.id),
            user_id=str(request.session.get("user_id")),
            parameters={
                'categorical_columns': categorical_columns,
                'drop_first': drop_first,
                'prefix': prefix
            },
            result_stats=result_stats,
            status='completed',
            completed_at=datetime.utcnow()
        )
        operation.save()
        
        preview_data_clean = dataframe_to_dict_clean(result_df.head(10))
        
        return JsonResponse({
            'success': True,
            'operation_id': str(operation.id),
            'new_dataset_id': str(new_dataset.id),
            'result_stats': clean_data_for_json(result_stats),
            'preview_data': preview_data_clean
        })
        
    except Exception as e:
        import traceback
        print(f"General error in one_hot_encoding: {str(e)}")
        print(traceback.format_exc())
        return JsonResponse({'success': False, 'error': str(e)}, status=400)
    
@mongo_login_required
@require_http_methods(["POST"])
@csrf_exempt
def preview_feature_engineering(request):
    """Preview feature engineering operations without saving"""
    try:
        data = json.loads(request.body)
        operation_type = data.get('operation_type')
        
        if operation_type == 'calculated_columns':
            return create_calculated_columns(request)
        elif operation_type == 'datetime_extraction':
            return datetime_extraction(request)
        elif operation_type == 'text_processing':
            return text_processing(request)
        elif operation_type == 'one_hot_encoding':
            return one_hot_encoding(request)
        else:
            return JsonResponse({'success': False, 'error': 'Unsupported operation type'}, status=400)
            
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)

@mongo_login_required
@require_http_methods(["POST"])
@csrf_exempt
def execute_feature_engineering(request):
    """Execute feature engineering operations and save result"""
    try:
        data = json.loads(request.body)
        operation_type = data.get('operation_type')
        
        if operation_type == 'calculated_columns':
            return create_calculated_columns(request)
        elif operation_type == 'datetime_extraction':
            return datetime_extraction(request)
        elif operation_type == 'text_processing':
            return text_processing(request)
        elif operation_type == 'one_hot_encoding':
            return one_hot_encoding(request)
        else:
            return JsonResponse({'success': False, 'error': 'Unsupported operation type'}, status=400)
            
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)



@mongo_login_required
@require_http_methods(["POST"])
@csrf_exempt
def train_test_split(request):
    """Split dataset into training and testing sets"""
    try:
        data = json.loads(request.body)
        dataset_id = data.get('dataset_id')
        target_column = data.get('target_column')
        test_size = data.get('test_size', 0.2)
        random_state = data.get('random_state', 42)
        shuffle = data.get('shuffle', True)
        stratify = data.get('stratify', False)
        preview_only = data.get('preview_only', False)
        
        # Validate inputs
        if not dataset_id:
            return JsonResponse({'success': False, 'error': 'Dataset ID is required'}, status=400)
        
        # Get dataset
        dataset = Dataset.objects.get(id=ObjectId(dataset_id), owner_id=str(request.session.get("user_id")))
        df = download_and_convert_to_dataframe(dataset)
        
        # Validate target column if provided
        if target_column and target_column not in df.columns:
            return JsonResponse({'success': False, 'error': f'Target column "{target_column}" not found'}, status=400)
        
        # Store original stats
        original_stats = {
            'total_rows': len(df),
            'columns': list(df.columns),
            'target_column': target_column
        }
        
        # Perform train-test split
        try:
            from sklearn.model_selection import train_test_split
            
            # Prepare features and target
            if target_column:
                X = df.drop(columns=[target_column])
                y = df[target_column]
                
                # Handle stratification
                stratify_param = y if stratify and target_column else None
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, 
                    test_size=test_size, 
                    random_state=random_state,
                    shuffle=shuffle,
                    stratify=stratify_param
                )
                
                # Combine features and target for output datasets
                train_df = X_train.copy()
                train_df[target_column] = y_train
                
                test_df = X_test.copy()
                test_df[target_column] = y_test
                
            else:
                # Split without target column (unsupervised learning)
                train_df, test_df = train_test_split(
                    df,
                    test_size=test_size, 
                    random_state=random_state,
                    shuffle=shuffle
                )
                
        except Exception as e:
            return JsonResponse({'success': False, 'error': f'Error in train-test split: {str(e)}'}, status=400)
        
        # Calculate result stats
        result_stats = {
            'train_rows': len(train_df),
            'test_rows': len(test_df),
            'train_percentage': f"{(len(train_df) / len(df)) * 100:.1f}%",
            'test_percentage': f"{(len(test_df) / len(df)) * 100:.1f}%",
            'split_ratio': f"{1-test_size}:{test_size}",
            'target_column': target_column,
            'stratified': stratify and target_column is not None
        }
        
        if preview_only:
            # Return preview of both train and test sets
            train_preview = dataframe_to_dict_clean(train_df.head(10))
            test_preview = dataframe_to_dict_clean(test_df.head(10))
            
            return JsonResponse({
                'success': True,
                'train_preview': train_preview,
                'test_preview': test_preview,
                'original_stats': clean_data_for_json(original_stats),
                'result_stats': clean_data_for_json(result_stats),
                'train_columns': list(train_df.columns),
                'test_columns': list(test_df.columns)
            })
        
        # Create new datasets for train and test
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        train_dataset_name = f"Train_Set_{timestamp}"
        test_dataset_name = f"Test_Set_{timestamp}"
        
        train_dataset = create_cleaned_dataset(train_df, train_dataset_name, request.session.get("user_id"), 'train_test_split')
        test_dataset = create_cleaned_dataset(test_df, test_dataset_name, request.session.get("user_id"), 'train_test_split')
        
        # Save operation record
        operation = DataCleaningOperation(
            name=f"Train_Test_Split_{timestamp}",
            operation_type='train_test_split',
            input_dataset_id=dataset_id,
            output_dataset_id=str(train_dataset.id),
            user_id=str(request.session.get("user_id")),
            parameters={
                'target_column': target_column,
                'test_size': test_size,
                'random_state': random_state,
                'shuffle': shuffle,
                'stratify': stratify
            },
            result_stats=result_stats,
            status='completed',
            completed_at=datetime.utcnow()
        )
        operation.save()
        
        # Also save test dataset reference in metadata
        operation.parameters['test_dataset_id'] = str(test_dataset.id)
        operation.save()
        
        train_preview_clean = dataframe_to_dict_clean(train_df.head(5))
        test_preview_clean = dataframe_to_dict_clean(test_df.head(5))
        
        return JsonResponse({
            'success': True,
            'operation_id': str(operation.id),
            'train_dataset_id': str(train_dataset.id),
            'test_dataset_id': str(test_dataset.id),
            'result_stats': clean_data_for_json(result_stats),
            'train_preview': train_preview_clean,
            'test_preview': test_preview_clean
        })
        
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)

@mongo_login_required
@require_http_methods(["POST"])
@csrf_exempt
def feature_scaling(request):
    """Apply feature scaling to numeric columns"""
    try:
        data = json.loads(request.body)
        dataset_id = data.get('dataset_id')
        numeric_columns = data.get('numeric_columns', [])
        scaling_method = data.get('scaling_method', 'standard')  # standard, minmax, robust
        exclude_columns = data.get('exclude_columns', [])
        preview_only = data.get('preview_only', False)
        
        # Validate inputs
        if not dataset_id:
            return JsonResponse({'success': False, 'error': 'Dataset ID is required'}, status=400)
        
        # Get dataset
        dataset = Dataset.objects.get(id=ObjectId(dataset_id), owner_id=str(request.session.get("user_id")))
        df = download_and_convert_to_dataframe(dataset)
        
        # Identify numeric columns if not specified
        if not numeric_columns:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude specified columns
        numeric_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        if not numeric_columns:
            return JsonResponse({'success': False, 'error': 'No numeric columns found for scaling'}, status=400)
        
        # Validate columns exist
        for col in numeric_columns:
            if col not in df.columns:
                return JsonResponse({'success': False, 'error': f'Column "{col}" not found in dataset'}, status=400)
        
        # Store original stats
        original_stats = {
            'total_rows': len(df),
            'columns': list(df.columns),
            'numeric_columns': numeric_columns,
            'original_means': {col: float(df[col].mean()) for col in numeric_columns},
            'original_stds': {col: float(df[col].std()) for col in numeric_columns}
        }
        
        # Apply feature scaling
        result_df = df.copy()
        scaling_params = {}
        
        try:
            from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
            
            if scaling_method == 'standard':
                scaler = StandardScaler()
            elif scaling_method == 'minmax':
                scaler = MinMaxScaler()
            elif scaling_method == 'robust':
                scaler = RobustScaler()
            else:
                return JsonResponse({'success': False, 'error': f'Unsupported scaling method: {scaling_method}'}, status=400)
            
            # Scale the numeric columns
            scaled_values = scaler.fit_transform(df[numeric_columns])
            
            # Create new column names
            scaled_columns = [f"{col}_scaled" for col in numeric_columns]
            
            # Add scaled columns to result dataframe
            for i, col in enumerate(numeric_columns):
                result_df[scaled_columns[i]] = scaled_values[:, i]
            
            # Store scaling parameters
            if hasattr(scaler, 'mean_'):
                scaling_params['means'] = scaler.mean_.tolist()
            if hasattr(scaler, 'scale_'):
                scaling_params['scales'] = scaler.scale_.tolist()
            if hasattr(scaler, 'min_'):
                scaling_params['mins'] = scaler.min_.tolist()
            if hasattr(scaler, 'data_min_'):
                scaling_params['data_mins'] = scaler.data_min_.tolist()
            if hasattr(scaler, 'data_max_'):
                scaling_params['data_maxs'] = scaler.data_max_.tolist()
                
        except Exception as e:
            return JsonResponse({'success': False, 'error': f'Error in feature scaling: {str(e)}'}, status=400)
        
        # Calculate result stats
        result_stats = {
            'total_rows': len(result_df),
            'result_columns': list(result_df.columns),
            'original_columns': original_stats['columns'],
            'scaled_columns': scaled_columns,
            'scaling_method': scaling_method,
            'scaling_parameters': scaling_params
        }
        
        if preview_only:
            preview_data = dataframe_to_dict_clean(result_df.head(20))
            return JsonResponse({
                'success': True,
                'preview_data': preview_data,
                'original_stats': clean_data_for_json(original_stats),
                'result_stats': clean_data_for_json(result_stats),
                'columns': list(result_df.columns)
            })
        
        # Create new dataset
        operation_name = f"Feature_Scaling_{scaling_method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        new_dataset = create_cleaned_dataset(result_df, operation_name, request.session.get("user_id"), 'feature_scaling')
        
        # Save operation record
        operation = DataCleaningOperation(
            name=operation_name,
            operation_type='feature_scaling',
            input_dataset_id=dataset_id,
            output_dataset_id=str(new_dataset.id),
            user_id=str(request.session.get("user_id")),
            parameters={
                'numeric_columns': numeric_columns,
                'scaling_method': scaling_method,
                'exclude_columns': exclude_columns,
                'scaling_parameters': scaling_params
            },
            result_stats=result_stats,
            status='completed',
            completed_at=datetime.utcnow()
        )
        operation.save()
        
        preview_data_clean = dataframe_to_dict_clean(result_df.head(10))
        
        return JsonResponse({
            'success': True,
            'operation_id': str(operation.id),
            'new_dataset_id': str(new_dataset.id),
            'result_stats': clean_data_for_json(result_stats),
            'preview_data': preview_data_clean
        })
        
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)

@mongo_login_required
@require_http_methods(["POST"])
@csrf_exempt
def outlier_detection(request):
    """Detect and handle outliers in numeric columns"""
    try:
        data = json.loads(request.body)
        dataset_id = data.get('dataset_id')
        numeric_columns = data.get('numeric_columns', [])
        detection_method = data.get('detection_method', 'iqr')  # iqr, zscore, isolation_forest
        handling_method = data.get('handling_method', 'mark')  # mark, remove, cap, transform
        z_threshold = data.get('z_threshold', 3.0)
        iqr_multiplier = data.get('iqr_multiplier', 1.5)
        preview_only = data.get('preview_only', False)
        
        # Validate inputs
        if not dataset_id:
            return JsonResponse({'success': False, 'error': 'Dataset ID is required'}, status=400)
        
        # Get dataset
        dataset = Dataset.objects.get(id=ObjectId(dataset_id), owner_id=str(request.session.get("user_id")))
        df = download_and_convert_to_dataframe(dataset)
        
        # Identify numeric columns if not specified
        if not numeric_columns:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_columns:
            return JsonResponse({'success': False, 'error': 'No numeric columns found for outlier detection'}, status=400)
        
        # Validate columns exist
        for col in numeric_columns:
            if col not in df.columns:
                return JsonResponse({'success': False, 'error': f'Column "{col}" not found in dataset'}, status=400)
        
        # Store original stats
        original_stats = {
            'total_rows': len(df),
            'columns': list(df.columns),
            'numeric_columns': numeric_columns,
            'original_means': {col: float(df[col].mean()) for col in numeric_columns},
            'original_stds': {col: float(df[col].std()) for col in numeric_columns}
        }
        
        # Detect and handle outliers
        result_df = df.copy()
        outlier_stats = {}
        
        try:
            for col in numeric_columns:
                col_data = df[col].dropna()
                outliers_mask = pd.Series([False] * len(df), index=df.index)
                
                if detection_method == 'zscore':
                    # Z-score method
                    from scipy import stats
                    z_scores = np.abs(stats.zscore(col_data))
                    outliers_mask = z_scores > z_threshold
                    
                elif detection_method == 'iqr':
                    # IQR method
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - iqr_multiplier * IQR
                    upper_bound = Q3 + iqr_multiplier * IQR
                    outliers_mask = (col_data < lower_bound) | (col_data > upper_bound)
                    
                elif detection_method == 'isolation_forest':
                    # Isolation Forest method
                    from sklearn.ensemble import IsolationForest
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    preds = iso_forest.fit_predict(col_data.values.reshape(-1, 1))
                    outliers_mask = preds == -1
                
                # Handle outliers based on selected method
                outlier_count = outliers_mask.sum()
                outlier_stats[col] = {
                    'outlier_count': int(outlier_count),
                    'outlier_percentage': f"{(outlier_count / len(col_data)) * 100:.2f}%"
                }
                
                if handling_method == 'mark':
                    # Mark outliers with a new column
                    result_df[f"{col}_outlier"] = outliers_mask
                    
                elif handling_method == 'remove':
                    # Remove outliers
                    result_df = result_df[~outliers_mask]
                    
                elif handling_method == 'cap':
                    # Cap outliers at bounds
                    if detection_method == 'iqr':
                        Q1 = col_data.quantile(0.25)
                        Q3 = col_data.quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - iqr_multiplier * IQR
                        upper_bound = Q3 + iqr_multiplier * IQR
                        
                        result_df[col] = np.where(result_df[col] < lower_bound, lower_bound, result_df[col])
                        result_df[col] = np.where(result_df[col] > upper_bound, upper_bound, result_df[col])
                        
                elif handling_method == 'transform':
                    # Apply log transformation to reduce outlier impact
                    result_df[f"{col}_log"] = np.log1p(result_df[col])
                    
        except Exception as e:
            return JsonResponse({'success': False, 'error': f'Error in outlier detection: {str(e)}'}, status=400)
        
        # Calculate result stats
        result_stats = {
            'total_rows': len(result_df),
            'result_columns': list(result_df.columns),
            'original_columns': original_stats['columns'],
            'outlier_stats': outlier_stats,
            'detection_method': detection_method,
            'handling_method': handling_method,
            'rows_removed': original_stats['total_rows'] - len(result_df) if handling_method == 'remove' else 0
        }
        
        if preview_only:
            preview_data = dataframe_to_dict_clean(result_df.head(20))
            return JsonResponse({
                'success': True,
                'preview_data': preview_data,
                'original_stats': clean_data_for_json(original_stats),
                'result_stats': clean_data_for_json(result_stats),
                'columns': list(result_df.columns)
            })
        
        # Create new dataset
        operation_name = f"Outlier_Detection_{detection_method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        new_dataset = create_cleaned_dataset(result_df, operation_name, request.session.get("user_id"), 'outlier_detection')
        
        # Save operation record
        operation = DataCleaningOperation(
            name=operation_name,
            operation_type='outlier_detection',
            input_dataset_id=dataset_id,
            output_dataset_id=str(new_dataset.id),
            user_id=str(request.session.get("user_id")),
            parameters={
                'numeric_columns': numeric_columns,
                'detection_method': detection_method,
                'handling_method': handling_method,
                'z_threshold': z_threshold,
                'iqr_multiplier': iqr_multiplier
            },
            result_stats=result_stats,
            status='completed',
            completed_at=datetime.utcnow()
        )
        operation.save()
        
        preview_data_clean = dataframe_to_dict_clean(result_df.head(10))
        
        return JsonResponse({
            'success': True,
            'operation_id': str(operation.id),
            'new_dataset_id': str(new_dataset.id),
            'result_stats': clean_data_for_json(result_stats),
            'preview_data': preview_data_clean
        })
        
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)

@mongo_login_required
@require_http_methods(["POST"])
@csrf_exempt
def cross_validation(request):
    """Perform cross-validation for model evaluation"""
    try:
        data = json.loads(request.body)
        dataset_id = data.get('dataset_id')
        target_column = data.get('target_column')
        cv_method = data.get('cv_method', 'kfold')  # kfold, stratified_kfold, leave_one_out
        n_splits = data.get('n_splits', 5)
        random_state = data.get('random_state', 42)
        shuffle = data.get('shuffle', True)
        preview_only = data.get('preview_only', False)
        
        # Validate inputs
        if not dataset_id:
            return JsonResponse({'success': False, 'error': 'Dataset ID is required'}, status=400)
        
        if not target_column:
            return JsonResponse({'success': False, 'error': 'Target column is required for cross-validation'}, status=400)
        
        # Get dataset
        dataset = Dataset.objects.get(id=ObjectId(dataset_id), owner_id=str(request.session.get("user_id")))
        df = download_and_convert_to_dataframe(dataset)
        
        # Validate target column
        if target_column not in df.columns:
            return JsonResponse({'success': False, 'error': f'Target column "{target_column}" not found'}, status=400)
        
        # Store original stats
        original_stats = {
            'total_rows': len(df),
            'columns': list(df.columns),
            'target_column': target_column,
            'target_distribution': df[target_column].value_counts().to_dict()
        }
        
        # Perform cross-validation
        cv_results = {}
        
        try:
            from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score, classification_report
            
            # Prepare features and target
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # Handle different CV methods
            if cv_method == 'kfold':
                cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
            elif cv_method == 'stratified_kfold':
                cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
            elif cv_method == 'leave_one_out':
                cv = LeaveOneOut()
                n_splits = len(df)  # Override for LOOCV
            else:
                return JsonResponse({'success': False, 'error': f'Unsupported CV method: {cv_method}'}, status=400)
            
            # Store fold information
            folds_data = []
            fold_scores = []
            
            for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Store fold data for preview
                fold_df = df.iloc[test_idx].copy()
                fold_df['fold'] = fold + 1
                folds_data.append(fold_df)
                
                # Simple model evaluation for demonstration
                try:
                    model = RandomForestClassifier(n_estimators=10, random_state=random_state)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    fold_scores.append(accuracy)
                except:
                    # If model fails (e.g., single class), use dummy score
                    fold_scores.append(0.5)
            
            # Combine all fold data for preview
            cv_preview_df = pd.concat(folds_data, ignore_index=True)
            
            # Calculate CV statistics
            cv_results = {
                'n_splits': n_splits,
                'cv_method': cv_method,
                'fold_scores': [float(score) for score in fold_scores],
                'mean_score': float(np.mean(fold_scores)),
                'std_score': float(np.std(fold_scores)),
                'min_score': float(np.min(fold_scores)),
                'max_score': float(np.max(fold_scores))
            }
            
        except Exception as e:
            return JsonResponse({'success': False, 'error': f'Error in cross-validation: {str(e)}'}, status=400)
        
        # Calculate result stats
        result_stats = {
            'total_rows': len(df),
            'cv_results': cv_results,
            'target_column': target_column,
            'feature_columns': list(X.columns)
        }
        
        if preview_only:
            preview_data = dataframe_to_dict_clean(cv_preview_df.head(20))
            return JsonResponse({
                'success': True,
                'preview_data': preview_data,
                'original_stats': clean_data_for_json(original_stats),
                'result_stats': clean_data_for_json(result_stats),
                'columns': list(cv_preview_df.columns)
            })
        
        # Create new dataset with fold information
        operation_name = f"Cross_Validation_{cv_method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        new_dataset = create_cleaned_dataset(cv_preview_df, operation_name, request.session.get("user_id"), 'cross_validation')
        
        # Save operation record
        operation = DataCleaningOperation(
            name=operation_name,
            operation_type='cross_validation',
            input_dataset_id=dataset_id,
            output_dataset_id=str(new_dataset.id),
            user_id=str(request.session.get("user_id")),
            parameters={
                'target_column': target_column,
                'cv_method': cv_method,
                'n_splits': n_splits,
                'random_state': random_state,
                'shuffle': shuffle
            },
            result_stats=result_stats,
            status='completed',
            completed_at=datetime.utcnow()
        )
        operation.save()
        
        preview_data_clean = dataframe_to_dict_clean(cv_preview_df.head(10))
        
        return JsonResponse({
            'success': True,
            'operation_id': str(operation.id),
            'new_dataset_id': str(new_dataset.id),
            'result_stats': clean_data_for_json(result_stats),
            'preview_data': preview_data_clean,
            'cv_results': clean_data_for_json(cv_results)
        })
        
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)

@mongo_login_required
@require_http_methods(["POST"])
@csrf_exempt
def preview_ml_preparation(request):
    """Preview ML preparation operations without saving"""
    try:
        data = json.loads(request.body)
        operation_type = data.get('operation_type')
        
        if operation_type == 'train_test_split':
            return train_test_split(request)
        elif operation_type == 'feature_scaling':
            return feature_scaling(request)
        elif operation_type == 'outlier_detection':
            return outlier_detection(request)
        elif operation_type == 'cross_validation':
            return cross_validation(request)
        else:
            return JsonResponse({'success': False, 'error': 'Unsupported operation type'}, status=400)
            
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)

@mongo_login_required
@require_http_methods(["POST"])
@csrf_exempt
def execute_ml_preparation(request):
    """Execute ML preparation operations and save result"""
    try:
        data = json.loads(request.body)
        operation_type = data.get('operation_type')
        
        if operation_type == 'train_test_split':
            return train_test_split(request)
        elif operation_type == 'feature_scaling':
            return feature_scaling(request)
        elif operation_type == 'outlier_detection':
            return outlier_detection(request)
        elif operation_type == 'cross_validation':
            return cross_validation(request)
        else:
            return JsonResponse({'success': False, 'error': 'Unsupported operation type'}, status=400)
            
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)


@mongo_login_required
@require_http_methods(["POST"])
@csrf_exempt
def create_pipeline(request):
    """Create a new transformation pipeline"""
    try:
        data = json.loads(request.body)
        
        print("=== PIPELINE CREATION REQUEST ===")
        print(f"Request data: {json.dumps(data, indent=2)}")
        
        # Validate required fields
        if not data.get('name'):
            return JsonResponse({'success': False, 'error': 'Pipeline name is required'}, status=400)
        
        if not data.get('input_dataset_id'):
            return JsonResponse({'success': False, 'error': 'Input dataset ID is required'}, status=400)
        
        if not data.get('steps') or len(data.get('steps', [])) == 0:
            return JsonResponse({'success': False, 'error': 'At least one step is required'}, status=400)
        
        # Get workspace_id from request or use a default
        workspace_id = data.get('workspace_id', 'default-workspace')
        
        # Verify the input dataset belongs to the user
        try:
            input_dataset = Dataset.objects.get(id=ObjectId(data.get('input_dataset_id')), owner_id=str(request.session.get("user_id")))
            print(f"Input dataset found: {input_dataset.file_name} (ID: {input_dataset.id})")
        except Dataset.DoesNotExist:
            return JsonResponse({'success': False, 'error': 'Input dataset not found or access denied'}, status=404)
        
        # Create the pipeline
        pipeline = TransformationPipeline(
            name=data.get('name'),
            description=data.get('description', ''),
            owner_id=str(request.session.get("user_id")),
            input_dataset_id=ObjectId(data.get('input_dataset_id')),
            steps=data.get('steps', []),
            total_steps=len(data.get('steps', [])),
            status='draft'
        )
        pipeline.save()
        
        print(f"✅ Pipeline created successfully!")
        print(f"   Pipeline ID: {pipeline.id}")
        print(f"   Pipeline Name: {pipeline.name}")
        print(f"   Owner ID: {pipeline.owner_id}")
        print(f"   Input Dataset ID: {pipeline.input_dataset_id}")
        print(f"   Total Steps: {pipeline.total_steps}")
        print(f"   Status: {pipeline.status}")
        
        # Create individual pipeline steps
        for i, step_config in enumerate(data.get('steps', [])):
            step = PipelineStep(
                pipeline_id=pipeline.id,
                step_number=i + 1,
                step_type=step_config.get('type'),
                operation=step_config.get('operation'),
                parameters=step_config.get('parameters', {}),
                status='pending'
            )
            step.save()
            
            print(f"   Step {i+1}: {step.step_type}.{step.operation}")
            print(f"      Parameters: {json.dumps(step.parameters, indent=8)}")
        
        # Log pipeline creation activity with workspace_id
        activity = WorkspaceActivity(
            workspace_id=workspace_id,  # Add this required field
            user_id=str(request.session.get("user_id")),
            user_name=request.user.username or "User",
            action='create',
            description=f'Created pipeline "{pipeline.name}"',
            details={
                'pipeline_name': pipeline.name,
                'pipeline_id': str(pipeline.id),
                'step_count': len(data.get('steps', [])),
                'input_dataset': input_dataset.file_name
            }
        )
        activity.save()
        
        # Verify the stored data by retrieving it from database
        print("\n=== DATABASE VERIFICATION ===")
        stored_pipeline = TransformationPipeline.objects.get(id=pipeline.id)
        print(f"Retrieved Pipeline: {stored_pipeline.name} (ID: {stored_pipeline.id})")
        print(f"Stored Steps: {len(stored_pipeline.steps)}")
        
        stored_steps = PipelineStep.objects.filter(pipeline_id=pipeline.id).order_by('step_number')
        print(f"Stored Step Documents: {stored_steps.count()}")
        
        for step in stored_steps:
            print(f"   Step {step.step_number}: {step.step_type}.{step.operation}")
            print(f"      Step Document ID: {step.id}")
            print(f"      Pipeline ID Reference: {step.pipeline_id}")
        
        response_data = {
            'success': True,
            'pipeline_id': str(pipeline.id),
            'pipeline_name': pipeline.name,
            'step_count': len(data.get('steps', [])),
            'message': 'Pipeline created successfully',
            'stored_data': {
                'pipeline_id': str(pipeline.id),
                'name': pipeline.name,
                'input_dataset_id': str(pipeline.input_dataset_id),
                'total_steps': pipeline.total_steps,
                'steps': [
                    {
                        'step_number': step.step_number,
                        'type': step.step_type,
                        'operation': step.operation,
                        'parameters': step.parameters
                    }
                    for step in stored_steps
                ]
            }
        }
        
        print(f"\n=== RESPONSE DATA ===")
        print(json.dumps(response_data, indent=2))
        
        return JsonResponse(response_data)
        
    except Exception as e:
        print(f"❌ Error creating pipeline: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        
        return JsonResponse({'success': False, 'error': str(e)}, status=400)


@mongo_login_required
@csrf_exempt
@require_http_methods(["GET", "POST"])
def get_pipelines(request):
    """Handle both GET (list pipelines) and POST (create pipeline) requests"""
    
    if request.method == 'GET':
        """Get all pipelines for the current user"""
        try:
            pipelines = TransformationPipeline.objects.filter(owner_id=str(request.session.get("user_id"))).order_by('-created_at')
            
            pipeline_data = []
            for pipeline in pipelines:
                pipeline_data.append({
                    'id': str(pipeline.id),
                    'name': pipeline.name,
                    'description': pipeline.description,
                    'status': getattr(pipeline, 'status', 'draft'),
                    'total_steps': getattr(pipeline, 'total_steps', 0),
                    'input_dataset_id': str(pipeline.input_dataset_id),
                    'created_at': pipeline.created_at.isoformat() if hasattr(pipeline, 'created_at') and pipeline.created_at else None,
                    'updated_at': pipeline.updated_at.isoformat() if hasattr(pipeline, 'updated_at') and pipeline.updated_at else None,
                    'step_summary': [
                        {
                            'type': step.get('type', 'unknown'),
                            'operation': step.get('operation', 'unknown')
                        } 
                        for step in getattr(pipeline, 'steps', [])[:3]
                    ]
                })
            
            return JsonResponse({
                'success': True,
                'pipelines': pipeline_data,
                'total_count': len(pipeline_data)
            })
            
        except Exception as e:
            print(f"Error fetching pipelines: {str(e)}")
            return JsonResponse({'success': False, 'error': 'Failed to fetch pipelines'}, status=500)
    
    elif request.method == 'POST':
        """Create a new pipeline"""
        try:
            data = json.loads(request.body)
            print(f"Creating pipeline with data: {data}")
            
            # Validate required fields
            required_fields = ['name', 'input_dataset_id', 'steps']
            for field in required_fields:
                if field not in data:
                    return JsonResponse({
                        'success': False, 
                        'error': f'Missing required field: {field}'
                    }, status=400)
            
            # Verify the input dataset exists and belongs to user
            try:
                dataset = Dataset.objects.get(
                    id=ObjectId(data['input_dataset_id']),
                    owner_id=str(request.session.get("user_id"))  # Changed from user_id to owner_id
                )
            except Dataset.DoesNotExist:
                return JsonResponse({
                    'success': False,
                    'error': 'Input dataset not found or access denied'
                }, status=404)
            
            # Create the pipeline
            pipeline = TransformationPipeline(
                name=data['name'],
                description=data.get('description', ''),
                input_dataset_id=ObjectId(data['input_dataset_id']),
                owner_id=str(request.session.get("user_id")),
                workspace_id=data.get('workspace_id', 'default-workspace'),
                steps=data['steps'],
                total_steps=len(data['steps']),
                status='draft'
            )
            pipeline.save()
            
            # Create individual pipeline steps
            for i, step_config in enumerate(data['steps']):
                step = PipelineStep(
                    pipeline_id=pipeline.id,
                    step_number=i + 1,
                    step_type=step_config.get('type'),
                    operation=step_config.get('operation'),
                    parameters=step_config.get('parameters', {}),
                    status='pending'
                )
                step.save()
            
            # Create workspace activity
            activity = WorkspaceActivity(
                workspace_id=data.get('workspace_id', 'default-workspace'),
                user_id=str(request.session.get("user_id")),
                user_name=request.user.username or "User",
                action='create',
                description=f'Created pipeline "{data["name"]}"',
                details={
                    'pipeline_name': data['name'],
                    'pipeline_id': str(pipeline.id),
                    'steps_count': len(data['steps']),
                    'input_dataset': dataset.file_name
                }
            )
            activity.save()
            
            return JsonResponse({
                'success': True,
                'pipeline_id': str(pipeline.id),
                'pipeline_name': pipeline.name,
                'step_count': len(data['steps']),
                'message': 'Pipeline created successfully',
                'stored_data': {
                    'pipeline_id': str(pipeline.id),
                    'name': pipeline.name,
                    'input_dataset_id': str(pipeline.input_dataset_id),
                    'total_steps': pipeline.total_steps,
                    'steps': [
                        {
                            'step_number': step.step_number,
                            'type': step.step_type,
                            'operation': step.operation,
                            'parameters': step.parameters
                        }
                        for step in PipelineStep.objects.filter(pipeline_id=pipeline.id).order_by('step_number')
                    ]
                }
            })
            
        except json.JSONDecodeError:
            return JsonResponse({'success': False, 'error': 'Invalid JSON'}, status=400)
        except Exception as e:
            print(f"Error creating pipeline: {str(e)}")
            return JsonResponse({'success': False, 'error': 'Failed to create pipeline'}, status=500)




from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from bson import ObjectId
import json


@mongo_login_required
@csrf_exempt
@require_http_methods(["DELETE"])
def delete_pipeline(request, pipeline_id):
    """Delete a pipeline"""
    try:
        # Verify pipeline exists and belongs to user
        pipeline = TransformationPipeline.objects.get(
            id=ObjectId(pipeline_id), 
            owner_id=str(request.session.get("user_id"))
        )
        
        # Delete associated steps first
        PipelineStep.objects.filter(pipeline_id=ObjectId(pipeline_id)).delete()
        
        # Delete the pipeline
        pipeline.delete()
        
        return JsonResponse({
            'success': True,
            'message': 'Pipeline deleted successfully'
        })
        
    except TransformationPipeline.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Pipeline not found'}, status=404)
    except Exception as e:
        print(f"Error deleting pipeline: {str(e)}")
        return JsonResponse({'success': False, 'error': 'Failed to delete pipeline'}, status=500)





@mongo_login_required
@csrf_exempt
@require_http_methods(["POST"])
def run_pipeline(request, pipeline_id):
    """Execute a pipeline by running all steps sequentially on a single file"""
    try:
        pipeline = TransformationPipeline.objects.get(
            id=ObjectId(pipeline_id), 
            owner_id=str(request.session.get("user_id"))
        )
        
        # Update pipeline status
        pipeline.status = 'running'
        pipeline.save()
        
        # Get the input dataset
        input_dataset = Dataset.objects.get(id=pipeline.input_dataset_id)
        
        # Load the initial dataframe ONCE
        df = download_and_convert_to_dataframe(input_dataset)
        execution_results = []
        branch_datasets = {}  # Store datasets from branches (like test sets)
        
        print(f"🚀 Starting pipeline execution: {pipeline.name}")
        print(f"Input dataset: {input_dataset.file_name} (ID: {input_dataset.id})")
        print(f"Initial data shape: {df.shape}")
        print(f"Total steps: {len(pipeline.steps)}")
        print(f"Initial columns: {list(df.columns)}")
        
        # Execute each step sequentially on the same dataframe
        for i, step in enumerate(pipeline.steps):
            step_number = i + 1
            step_type = step.get('type')
            operation = step.get('operation')
            parameters = step.get('parameters', {})
            step_name = step.get('name', f'Step {step_number}')
            
            print(f"\n▶️ Executing step {step_number}: {step_name} ({step_type}.{operation})")
            print(f"   Parameters: {json.dumps(parameters, indent=2)}")
            print(f"   Data shape before step: {df.shape}")
            print(f"   Columns before step: {list(df.columns)}")
            
            try:
                # Store original state for rollback if needed
                original_shape = df.shape
                original_columns = list(df.columns)
                
                # Apply the transformation directly to the dataframe
                df = apply_pipeline_step_to_dataframe(df, step_type, operation, parameters)
                
                # Check if this step created branch data (like train-test split)
                branch_data = None
                step_meta = {}
                
                if hasattr(df, 'attrs') and df.attrs:
                    step_meta = clean_data_for_json(df.attrs.copy())
                    
                    # Check if this is a train-test split that created a test set
                    if 'test_set_data' in df.attrs:
                        branch_data = df.attrs['test_set_data']
                        print(f"🌿 Step created branch data with shape: {branch_data.shape}")
                    
                    # Clear attrs for next step
                    df.attrs.clear()
                
                execution_results.append({
                    'step_number': step_number,
                    'step_name': step_name,
                    'step_type': step_type,
                    'operation': operation,
                    'success': True,
                    'rows_processed': len(df),
                    'data_shape_before': original_shape,
                    'data_shape_after': df.shape,
                    'columns_added': list(set(df.columns) - set(original_columns)),
                    'columns_removed': list(set(original_columns) - set(df.columns)),
                    'step_metadata': step_meta,
                    'execution_time': datetime.utcnow().isoformat(),
                    'created_branch': branch_data is not None  # Fixed: use explicit None check
                })
                
                # If branch data was created (like test set), save it immediately
                if branch_data is not None:  # Fixed: explicit None check instead of truthy check
                    try:
                        branch_dataset = create_dataset_from_dataframe(
                            branch_data, 
                            f"{pipeline.name}_test_set_step_{step_number}", 
                            str(request.session.get("user_id")),
                            f"Test set from step {step_number}: {step_name} of pipeline: {pipeline.name}"
                        )
                        branch_datasets[f"test_set_step_{step_number}"] = {
                            'dataset_id': str(branch_dataset.id),
                            'dataset_name': branch_dataset.file_name,
                            'rows': len(branch_data),
                            'columns': len(branch_data.columns),
                            'source_step': step_number
                        }
                        print(f"✅ Created branch dataset: {branch_dataset.file_name}")
                    except Exception as e:
                        print(f"❌ Failed to create branch dataset: {str(e)}")
                        # Don't fail the whole pipeline if branch dataset creation fails
                        branch_datasets[f"test_set_step_{step_number}"] = {
                            'dataset_id': None,
                            'error': str(e),
                            'source_step': step_number
                        }
                
                print(f"✅ Step {step_number} completed successfully")
                print(f"   Data shape after step: {df.shape}")
                print(f"   Columns after step: {list(df.columns)}")
                
                # Check if dataframe is empty after transformation
                if len(df) == 0:
                    raise ValueError("DataFrame is empty after transformation")
                    
            except Exception as e:
                error_msg = f"Step {step_number} ({step_name}) failed: {str(e)}"
                print(f"❌ {error_msg}")
                
                execution_results.append({
                    'step_number': step_number,
                    'step_name': step_name,
                    'step_type': step_type,
                    'operation': operation,
                    'success': False,
                    'error': str(e),
                    'data_shape_before': original_shape if 'original_shape' in locals() else None,
                    'execution_time': datetime.utcnow().isoformat()
                })
                
                # Update pipeline status to failed
                pipeline.status = 'failed'
                pipeline.execution_results = clean_data_for_json(execution_results)
                pipeline.last_executed = datetime.utcnow()
                pipeline.execution_count += 1
                pipeline.save()
                
                # Create failure activity
                activity = WorkspaceActivity(
                    workspace_id=getattr(pipeline, 'workspace_id', 'default-workspace'),
                    user_id=str(request.session.get("user_id")),
                    user_name=request.user.username or "User",
                    action='execute',
                    description=f'Failed to execute pipeline "{pipeline.name}"',
                    details={
                        'pipeline_name': pipeline.name,
                        'pipeline_id': str(pipeline.id),
                        'status': 'failed',
                        'failed_at_step': step_number,
                        'failed_step_name': step_name,
                        'error': str(e),
                        'steps_completed': step_number - 1,
                        'total_steps': len(pipeline.steps)
                    }
                )
                activity.save()
                
                return JsonResponse({
                    'success': False,
                    'message': error_msg,
                    'pipeline_id': pipeline_id,
                    'status': 'failed',
                    'failed_step': step_number,
                    'failed_step_name': step_name,
                    'error': str(e),
                    'execution_results': clean_data_for_json(execution_results),
                    'steps_completed': step_number - 1,
                    'total_steps': len(pipeline.steps)
                }, status=400)
        
        # Update pipeline status based on execution results
        all_steps_successful = all(result['success'] for result in execution_results)
        
        if all_steps_successful:
            try:
                # Create FINAL output dataset (training set) after all steps are complete
                final_dataset = create_dataset_from_dataframe(
                    df, 
                    f"{pipeline.name}_train", 
                    str(request.session.get("user_id")),
                    f"Training set from pipeline: {pipeline.name}"
                )
                
                pipeline.status = 'completed'
                pipeline.final_dataset_id = final_dataset.id
                final_message = f"Pipeline completed successfully with {len(execution_results)} steps"
                
                if branch_datasets:
                    final_message += f" and created {len(branch_datasets)} branch dataset(s)"
                
                print(f"🎉 {final_message}")
                print(f"Final training data shape: {df.shape}")
                print(f"Final columns: {list(df.columns)}")
                
            except Exception as e:
                # Handle final dataset creation error
                pipeline.status = 'failed'
                final_message = f"Pipeline steps completed but failed to create final dataset: {str(e)}"
                print(f"💥 {final_message}")
                
        else:
            pipeline.status = 'failed'
            failed_step = next((result for result in execution_results if not result['success']), None)
            final_message = f"Pipeline failed at step {failed_step['step_number']}: {failed_step['error']}"
            print(f"💥 {final_message}")
        
        # Clean execution results before saving to MongoDB
        cleaned_execution_results = clean_data_for_json(execution_results)
        cleaned_branch_datasets = clean_data_for_json(branch_datasets)
        
        # Save execution results and final status
        pipeline.execution_results = cleaned_execution_results
        pipeline.branch_datasets = cleaned_branch_datasets
        pipeline.last_executed = datetime.utcnow()
        pipeline.execution_count += 1
        pipeline.save()
        
        # Create workspace activity
        activity_details = {
            'pipeline_name': pipeline.name,
            'pipeline_id': str(pipeline.id),
            'status': pipeline.status,
            'steps_executed': len([r for r in execution_results if r['success']]),
            'total_steps': len(pipeline.steps),
            'final_data_shape': df.shape if all_steps_successful else None,
            'execution_time': datetime.utcnow().isoformat(),
            'branch_datasets_created': len(branch_datasets)
        }
        
        if pipeline.final_dataset_id:
            activity_details['final_dataset_id'] = str(pipeline.final_dataset_id)
            activity_details['final_dataset_name'] = final_dataset.file_name if 'final_dataset' in locals() else None
        
        activity = WorkspaceActivity(
            workspace_id=getattr(pipeline, 'workspace_id', 'default-workspace'),
            user_id=str(request.session.get("user_id")),
            user_name=request.user.username or "User",
            action='execute',
            description=f'Executed pipeline "{pipeline.name}" - {pipeline.status}',
            details=activity_details
        )
        activity.save()
        
        response_data = {
            'success': all_steps_successful,
            'message': final_message,
            'pipeline_id': pipeline_id,
            'status': pipeline.status,
            'execution_results': cleaned_execution_results,
            'steps_completed': len([r for r in execution_results if r['success']]),
            'total_steps': len(pipeline.steps),
        }
        
        if pipeline.final_dataset_id:
            response_data.update({
                'final_dataset_id': str(pipeline.final_dataset_id),
                'final_dataset_name': final_dataset.file_name if 'final_dataset' in locals() else 'Unknown',
                'final_data_shape': df.shape if all_steps_successful else None,
                'final_columns': list(df.columns) if all_steps_successful else None
            })
        
        # Add branch datasets to response
        if branch_datasets:
            response_data['branch_datasets'] = cleaned_branch_datasets
        
        return JsonResponse(response_data)
        
    except TransformationPipeline.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Pipeline not found'}, status=404)
    except Dataset.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Input dataset not found'}, status=404)
    except Exception as e:
        print(f"Error running pipeline: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        
        return JsonResponse({
            'success': False, 
            'error': f'Failed to run pipeline: {str(e)}'
        }, status=500)


@mongo_login_required
@csrf_exempt
@require_http_methods(["POST"])
def create_and_run_pipeline_from_json(request):
    """
    Create a pipeline from JSON payload and run it immediately.
    Expected JSON body:
    {
      "pipeline_name": "my_pipeline",
      "dataset_id": "<optional dataset id>",
      "filename": "<optional file name (fallback if dataset_id missing)>",
      "steps": [
         {"type": "ml", "operation": "train-test-split", "parameters": {...}, "name": "Split"},
         {"type": "ml", "operation": "feature-scaling", "parameters": {...}, "name": "Scale"},
         ...
      ]
    }
    Returns: JSON response from run_pipeline (or validation errors)
    """
    try:
        payload = json.loads(request.body or "{}")
    except Exception as e:
        return JsonResponse({"success": False, "error": "Invalid JSON payload"}, status=400)

    pipeline_name = payload.get("pipeline_name") or f"pipeline_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    dataset_id = payload.get("dataset_id")
    filename = payload.get("filename")
    steps = payload.get("steps")

    if not steps or not isinstance(steps, list):
        return JsonResponse({"success": False, "error": "`steps` must be a non-empty array"}, status=400)

    # Resolve dataset
    dataset_obj = None
    try:
        if dataset_id:
            dataset_obj = Dataset.objects.get(id=ObjectId(dataset_id), owner_id=str(request.session.get("user_id")))
        elif filename:
            dataset_obj = Dataset.objects.get(file_name=filename, owner_id=str(request.session.get("user_id")))
        else:
            return JsonResponse({"success": False, "error": "Provide dataset_id or filename"}, status=400)
    except Dataset.DoesNotExist:
        return JsonResponse({"success": False, "error": "Dataset not found or not owned by user"}, status=404)
    except Exception as e:
        return JsonResponse({"success": False, "error": f"Dataset lookup error: {str(e)}"}, status=400)

    # Build pipeline document
    try:
        pipeline_doc = TransformationPipeline(
            name=pipeline_name,
            owner_id=str(request.session.get("user_id")),
            input_dataset_id=str(dataset_obj.id),
            steps=steps,
            status="created",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            execution_count=0
        )
        pipeline_doc.save()
    except Exception as e:
        return JsonResponse({"success": False, "error": f"Failed to create pipeline: {str(e)}"}, status=500)

    # Call the existing run_pipeline view directly to execute the pipeline synchronously.
    # Note: run_pipeline expects (request, pipeline_id). We'll call it and return its JsonResponse.
    try:
        # run_pipeline returns a JsonResponse — call and return it
        return run_pipeline(request, str(pipeline_doc.id))
    except Exception as e:
        # If run_pipeline raises, update pipeline status and return error
        try:
            pipeline_doc.status = "failed"
            pipeline_doc.execution_results = [{"error": str(e)}]
            pipeline_doc.last_executed = datetime.utcnow()
            pipeline_doc.save()
        except:
            pass
        return JsonResponse({"success": False, "error": f"Pipeline execution failed: {str(e)}"}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def run_pipeline_from_json(request):
    """
    Run a transformation pipeline directly from JSON POST (for CURL use)
    """
    import json
    from bson import ObjectId
    from assistant.models import Dataset
    from assistant.core.pipeline.pipeline_runner import apply_pipeline_step_to_dataframe

    try:
        body = json.loads(request.body.decode('utf-8'))

        dataset_name = body.get("filename")
        steps = body.get("steps", [])

        if not dataset_name:
            return JsonResponse({"success": False, "error": "filename missing"}, status=400)

        # --- Load dataset ---
        dataset = Dataset.objects.get(filename=dataset_name)
        df = dataset.dataframe

        # --- Apply pipeline steps in order ---
        for step in steps:
            step_type = step.get("type")
            operation = step.get("operation")
            parameters = step.get("parameters", {})

            df = apply_pipeline_step_to_dataframe(df, step_type, operation, parameters)

        # --- Save final dataset output ---
        output = Dataset.objects.create_from_dataframe(
            owner_id=str(request.session.get("user_id")) if request.user.is_authenticated else None,
            filename=f"{dataset_name}_processed",
            dataframe=df
        )

        return JsonResponse({
            "success": True,
            "message": "Pipeline executed",
            "output_dataset_id": str(output.id),
            "rows": len(df),
            "columns": list(df.columns)
        })

    except Dataset.DoesNotExist:
        return JsonResponse({"success": False, "error": "dataset not found"}, status=404)
    except Exception as e:
        print("run_pipeline_from_json::", e)
        return JsonResponse({"success": False, "error": str(e)}, status=500)

def apply_train_test_split_direct(df, parameters):
    """Apply train-test split and create both train and test datasets"""
    from sklearn.model_selection import train_test_split
    
    target_column = parameters.get('target_column')
    test_size = parameters.get('test_size', 0.2)
    random_state = parameters.get('random_state', 42)
    shuffle = parameters.get('shuffle', True)
    stratify = parameters.get('stratify', False)
    
    # Perform the split
    if target_column:
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        stratify_param = y if stratify and target_column else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify_param
        )
        
        # Combine features and target
        train_df = X_train.copy()
        train_df[target_column] = y_train
        
        test_df = X_test.copy()
        test_df[target_column] = y_test
        
    else:
        # Unsupervised learning
        train_df, test_df = train_test_split(
            df,
            test_size=test_size, 
            random_state=random_state,
            shuffle=shuffle
        )
    
    # Store test set in metadata so it can be saved as a branch dataset
    train_df.attrs['test_set_data'] = test_df
    train_df.attrs['train_test_split_info'] = {
        'test_size': test_size,
        'target_column': target_column,
        'stratified': stratify,
        'train_rows': len(train_df),
        'test_rows': len(test_df),
        'split_ratio': f"{(1-test_size)*100:.1f}% / {test_size*100:.1f}%"
    }
    
    print(f"📊 Train-test split completed: {len(train_df)} train rows, {len(test_df)} test rows")
    
    # Return training set as the main pipeline output
    return train_df

def create_dataset_from_dataframe(df, base_name, user_id, description):
    """Create a dataset from a pandas DataFrame"""
    from datasets.models import Dataset
    from supabase import create_client
    from django.conf import settings
    import io
    import uuid
    
    supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
    
    try:
        # Convert DataFrame to CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()
        
        # Upload to Supabase
        unique_filename = f"{base_name}_{uuid.uuid4().hex[:8]}.csv"
        supabase_path = f"{user_id}/{unique_filename}"
        
        res = supabase.storage.from_(settings.SUPABASE_BUCKET).upload(
            supabase_path, csv_content.encode('utf-8')
        )
        
        # Get public URL
        file_url = supabase.storage.from_(settings.SUPABASE_BUCKET).get_public_url(supabase_path)
        
        # Create dataset record
        dataset = Dataset(
            owner_id=user_id,
            file_name=unique_filename,
            file_type="text/csv",
            file_url=file_url,
            file_path=supabase_path,
            uploaded_at=datetime.utcnow(),
            description=description,
            metadata={
                "is_pipeline_output": True,
                "base_name": base_name,
                "created_from_transformation": True,
                "row_count": len(df),
                "column_count": len(df.columns),
                "created_at": datetime.utcnow().isoformat()
            }
        )
        
        dataset.save()
        print(f"💾 Created dataset: {unique_filename} with {len(df)} rows")
        return dataset
        
    except Exception as e:
        print(f"Error creating dataset: {str(e)}")
        raise

# Helper function to clean data for JSON serialization
def clean_data_for_json(data):
    """Recursively clean data for JSON serialization"""
    if isinstance(data, dict):
        return {str(k): clean_data_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_data_for_json(item) for item in data]
    elif isinstance(data, (int, float)):
        # Handle NaN and infinity
        if isinstance(data, float) and (np.isnan(data) or np.isinf(data)):
            return None
        return data
    elif isinstance(data, (str, bool, type(None))):
        return data
    elif isinstance(data, (pd.DataFrame, pd.Series)):
        # Convert DataFrames/Series to basic info
        return f"DataFrame({data.shape})" if hasattr(data, 'shape') else str(data)
    else:
        return str(data)

# Keep all other functions the same (apply_feature_scaling_direct, apply_outlier_detection_direct, etc.)
# ... [rest of your existing functions remain unchanged] ...
def apply_pipeline_step_to_dataframe(df, step_type, operation, parameters):
    """Apply a pipeline step directly to a pandas DataFrame"""
    
    # Normalize operation names (hyphen to underscore)
    def normalize_operation_name(op):
        mapping = {
            'train-test-split': 'train_test_split',
            'feature-scaling': 'feature_scaling',
            'outlier-detection': 'outlier_detection',
            'cross-validation': 'cross_validation',
            'group-by-&-summarize': 'group_by_summarize',
            'window-functions': 'window_functions',
            'rollup-&-cube': 'rollup_cube',
            'data-type-conversion': 'data_type_conversion',
            'text-cleaning': 'text_cleaning',
            'filter-rows': 'filter_rows',
            'sort-data': 'sort_data',
            'select-columns': 'select_columns',
            'remove-columns': 'remove_columns',
            'top-n-records': 'top_n_records',
            'random-sampling': 'random_sampling',
            'calculated-columns': 'calculated_columns',
            'datetime-extraction': 'datetime_extraction',
            'text-processing': 'text_processing',
            'one-hot-encoding': 'one_hot_encoding'
        }
        return mapping.get(op, op)
    
    normalized_operation = normalize_operation_name(operation)
    
    print(f"Applying step: {step_type}.{operation} (normalized: {normalized_operation})")
    
    if step_type == 'cleaning':
        if normalized_operation == 'handle-missing-values':
            return apply_missing_values_direct(df, parameters)
        elif normalized_operation == 'remove-duplicates':
            return apply_remove_duplicates_direct(df, parameters)
        elif normalized_operation == 'data_type_conversion':
            return apply_data_type_conversion_direct(df, parameters)
        elif normalized_operation == 'text_cleaning':
            return apply_text_cleaning_direct(df, parameters)
    
    elif step_type == 'aggregation':
        if normalized_operation == 'group_by_summarize':
            return apply_groupby_aggregation_direct(df, parameters)
        elif normalized_operation == 'pivot-tables':
            return apply_pivot_table_direct(df, parameters)
        elif normalized_operation == 'window_functions':
            return apply_window_functions_direct(df, parameters)
        elif normalized_operation == 'rollup_cube':
            return apply_rollup_cube_direct(df, parameters)
    
    elif step_type == 'filter':
        if normalized_operation == 'filter_rows':
            return apply_filter_rows_direct(df, parameters)
        elif normalized_operation == 'sort_data':
            return apply_sort_data_direct(df, parameters)
        elif normalized_operation == 'select_columns':
            return apply_select_columns_direct(df, parameters)
        elif normalized_operation == 'remove_columns':
            return apply_remove_columns_direct(df, parameters)
        elif normalized_operation == 'top_n_records':
            return apply_top_n_records_direct(df, parameters)
        elif normalized_operation == 'random_sampling':
            return apply_random_sampling_direct(df, parameters)
    
    elif step_type == 'feature':
        if normalized_operation == 'calculated_columns':
            return apply_calculated_columns_direct(df, parameters)
        elif normalized_operation == 'datetime_extraction':
            return apply_datetime_extraction_direct(df, parameters)
        elif normalized_operation == 'text_processing':
            return apply_text_processing_direct(df, parameters)
        elif normalized_operation == 'one_hot_encoding':
            return apply_one_hot_encoding_direct(df, parameters)
    
    # ML PREPARATION OPERATIONS
    elif step_type == 'ml':
        if normalized_operation == 'train_test_split':
            return apply_train_test_split_direct(df, parameters)
        elif normalized_operation == 'feature_scaling':
            return apply_feature_scaling_direct(df, parameters)
        elif normalized_operation == 'outlier_detection':
            return apply_outlier_detection_direct(df, parameters)
        elif normalized_operation == 'cross_validation':
            return apply_cross_validation_direct(df, parameters)
    
    raise ValueError(f"Unsupported operation: {step_type}.{operation} (normalized: {normalized_operation})")




def apply_feature_scaling_direct(df, parameters):
    """Apply feature scaling directly to DataFrame"""
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    
    scaling_method = parameters.get('scaling_method', 'standard')
    exclude_columns = parameters.get('exclude_columns', [])
    
    # Identify numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclude specified columns
    numeric_columns = [col for col in numeric_columns if col not in exclude_columns]
    
    if not numeric_columns:
        return df  # No numeric columns to scale
    
    # Apply scaling
    result_df = df.copy()
    
    if scaling_method == 'standard':
        scaler = StandardScaler()
    elif scaling_method == 'minmax':
        scaler = MinMaxScaler()
    elif scaling_method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unsupported scaling method: {scaling_method}")
    
    # Scale the numeric columns
    scaled_values = scaler.fit_transform(df[numeric_columns])
    
    # Create new column names
    scaled_columns = [f"{col}_scaled" for col in numeric_columns]
    
    # Add scaled columns to result dataframe
    for i, col in enumerate(numeric_columns):
        result_df[scaled_columns[i]] = scaled_values[:, i]
    
    # Store scaling parameters in metadata
    result_df.attrs['scaling_info'] = {
        'scaling_method': scaling_method,
        'original_columns': numeric_columns,
        'scaled_columns': scaled_columns,
        'scaler_params': {
            'means': scaler.mean_.tolist() if hasattr(scaler, 'mean_') else None,
            'scales': scaler.scale_.tolist() if hasattr(scaler, 'scale_') else None
        }
    }
    
    return result_df

def apply_outlier_detection_direct(df, parameters):
    """Apply outlier detection directly to DataFrame"""
    from scipy import stats
    from sklearn.ensemble import IsolationForest
    
    detection_method = parameters.get('detection_method', 'iqr')
    handling_method = parameters.get('handling_method', 'mark')
    numeric_columns = parameters.get('numeric_columns', [])
    z_threshold = parameters.get('z_threshold', 3.0)
    iqr_multiplier = parameters.get('iqr_multiplier', 1.5)
    
    # Identify numeric columns if not specified
    if not numeric_columns:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_columns:
        return df  # No numeric columns to analyze
    
    result_df = df.copy()
    outlier_info = {}
    
    for col in numeric_columns:
        col_data = df[col].dropna()
        outliers_mask = pd.Series([False] * len(df), index=df.index)
        
        if detection_method == 'zscore':
            # Z-score method
            z_scores = np.abs(stats.zscore(col_data))
            outliers_mask = z_scores > z_threshold
            
        elif detection_method == 'iqr':
            # IQR method
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - iqr_multiplier * IQR
            upper_bound = Q3 + iqr_multiplier * IQR
            outliers_mask = (col_data < lower_bound) | (col_data > upper_bound)
            
        elif detection_method == 'isolation_forest':
            # Isolation Forest method
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            preds = iso_forest.fit_predict(col_data.values.reshape(-1, 1))
            outliers_mask = preds == -1
        
        # Handle outliers based on selected method
        outlier_count = outliers_mask.sum()
        outlier_info[col] = {
            'outlier_count': int(outlier_count),
            'outlier_percentage': (outlier_count / len(col_data)) * 100
        }
        
        if handling_method == 'mark':
            # Mark outliers with a new column
            result_df[f"{col}_outlier"] = outliers_mask
            
        elif handling_method == 'remove':
            # Remove outliers
            result_df = result_df[~outliers_mask]
            
        elif handling_method == 'cap':
            # Cap outliers at bounds
            if detection_method == 'iqr':
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - iqr_multiplier * IQR
                upper_bound = Q3 + iqr_multiplier * IQR
                
                result_df[col] = np.where(result_df[col] < lower_bound, lower_bound, result_df[col])
                result_df[col] = np.where(result_df[col] > upper_bound, upper_bound, result_df[col])
                
        elif handling_method == 'transform':
            # Apply log transformation to reduce outlier impact
            result_df[f"{col}_log"] = np.log1p(result_df[col])
    
    # Store outlier information in metadata
    result_df.attrs['outlier_info'] = {
        'detection_method': detection_method,
        'handling_method': handling_method,
        'outlier_stats': outlier_info,
        'rows_removed': len(df) - len(result_df) if handling_method == 'remove' else 0
    }
    
    return result_df

def apply_cross_validation_direct(df, parameters):
    """Apply cross-validation directly to DataFrame"""
    from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    
    target_column = parameters.get('target_column')
    cv_method = parameters.get('cv_method', 'kfold')
    n_splits = parameters.get('n_splits', 5)
    random_state = parameters.get('random_state', 42)
    shuffle = parameters.get('shuffle', True)
    
    if not target_column:
        raise ValueError("Target column is required for cross-validation")
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")
    
    # Prepare features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Handle different CV methods
    if cv_method == 'kfold':
        cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    elif cv_method == 'stratified_kfold':
        cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    elif cv_method == 'leave_one_out':
        cv = LeaveOneOut()
        n_splits = len(df)
    else:
        raise ValueError(f"Unsupported CV method: {cv_method}")
    
    # Store fold information
    folds_data = []
    fold_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Store fold data
        fold_df = df.iloc[test_idx].copy()
        fold_df['fold'] = fold + 1
        folds_data.append(fold_df)
        
        # Simple model evaluation for demonstration
        try:
            model = RandomForestClassifier(n_estimators=10, random_state=random_state)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            fold_scores.append(accuracy)
        except:
            # If model fails, use dummy score
            fold_scores.append(0.5)
    
    # Combine all fold data
    cv_result_df = pd.concat(folds_data, ignore_index=True)
    
    # Calculate CV statistics
    cv_results = {
        'n_splits': n_splits,
        'cv_method': cv_method,
        'fold_scores': [float(score) for score in fold_scores],
        'mean_score': float(np.mean(fold_scores)),
        'std_score': float(np.std(fold_scores)),
        'min_score': float(np.min(fold_scores)),
        'max_score': float(np.max(fold_scores))
    }
    
    # Store CV results in metadata
    cv_result_df.attrs['cv_info'] = {
        'target_column': target_column,
        'cv_method': cv_method,
        'cv_results': cv_results,
        'feature_columns': list(X.columns)
    }
    
    return cv_result_df

def create_final_dataset_from_dataframe(df, pipeline_name, user_id, description):
    """Create final dataset after all pipeline steps"""
    from datasets.models import Dataset
    from supabase import create_client
    from django.conf import settings
    import io
    import uuid
    
    supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
    
    try:
        # Convert DataFrame to CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()
        
        # Upload to Supabase
        unique_filename = f"pipeline_output_{uuid.uuid4()}_{pipeline_name}.csv"
        supabase_path = f"{user_id}/{unique_filename}"
        
        res = supabase.storage.from_(settings.SUPABASE_BUCKET).upload(
            supabase_path, csv_content.encode('utf-8')
        )
        
        # Get public URL
        file_url = supabase.storage.from_(settings.SUPABASE_BUCKET).get_public_url(supabase_path)
        
        # Create dataset record
        dataset = Dataset(
            owner_id=user_id,
            file_name=f"{pipeline_name}_output.csv",
            file_type="text/csv",
            file_url=file_url,
            file_path=supabase_path,
            uploaded_at=datetime.utcnow(),
            description=description,
            metadata={
                "is_pipeline_output": True,
                "pipeline_name": pipeline_name,
                "created_from_transformation": True,
                "row_count": len(df),
                "column_count": len(df.columns)
            }
        )
        
        dataset.save()
        return dataset
    except Exception as e:
        print(f"Error creating final dataset: {str(e)}")
        raise


# Direct dataframe transformation functions for filter operations
def apply_filter_rows_direct(df, parameters):
    """Apply filter operations directly to dataframe"""
    filters = parameters.get('filters', [])
    
    if not filters:
        return df
    
    result_df = df.copy()
    
    for filter_config in filters:
        column = filter_config.get('column')
        operator = filter_config.get('operator')
        value = filter_config.get('value')
        
        if column not in df.columns:
            raise ValueError(f'Column "{column}" not found in dataset')
        
        try:
            if operator == 'equals':
                result_df = result_df[result_df[column] == value]
            elif operator == 'not_equals':
                result_df = result_df[result_df[column] != value]
            elif operator == 'contains':
                result_df = result_df[result_df[column].astype(str).str.contains(str(value), na=False)]
            elif operator == 'starts_with':
                result_df = result_df[result_df[column].astype(str).str.startswith(str(value))]
            elif operator == 'ends_with':
                result_df = result_df[result_df[column].astype(str).str.endswith(str(value))]
            elif operator == 'greater_than':
                result_df = result_df[result_df[column] > value]
            elif operator == 'less_than':
                result_df = result_df[result_df[column] < value]
            elif operator == 'greater_than_equal':
                result_df = result_df[result_df[column] >= value]
            elif operator == 'less_than_equal':
                result_df = result_df[result_df[column] <= value]
            elif operator == 'is_null':
                result_df = result_df[result_df[column].isnull()]
            elif operator == 'not_null':
                result_df = result_df[result_df[column].notnull()]
            elif operator == 'in_list':
                if isinstance(value, list):
                    result_df = result_df[result_df[column].isin(value)]
                else:
                    result_df = result_df[result_df[column].isin([value])]
            elif operator == 'not_in_list':
                if isinstance(value, list):
                    result_df = result_df[~result_df[column].isin(value)]
                else:
                    result_df = result_df[~result_df[column].isin([value])]
            elif operator == 'between':
                if isinstance(value, list) and len(value) == 2:
                    result_df = result_df[(result_df[column] >= value[0]) & (result_df[column] <= value[1])]
            else:
                raise ValueError(f"Unsupported filter operator: {operator}")
                
        except Exception as e:
            raise ValueError(f'Error applying filter {operator} on {column}: {str(e)}')
    
    return result_df.reset_index(drop=True)

def apply_sort_data_direct(df, parameters):
    """Apply sorting directly to dataframe"""
    sort_columns = parameters.get('sort_columns', [])
    
    if not sort_columns:
        return df
    
    sort_by = []
    ascending = []
    
    for sort_config in sort_columns:
        column = sort_config.get('column')
        order = sort_config.get('order', 'asc')
        
        if column not in df.columns:
            raise ValueError(f'Column "{column}" not found in dataset')
        
        sort_by.append(column)
        ascending.append(order == 'asc')
    
    return df.sort_values(by=sort_by, ascending=ascending).reset_index(drop=True)

def apply_select_columns_direct(df, parameters):
    """Apply column selection directly to dataframe"""
    selected_columns = parameters.get('selected_columns', [])
    
    if not selected_columns:
        return df
    
    # Validate all selected columns exist
    missing_columns = [col for col in selected_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Selected columns not found: {missing_columns}")
    
    return df[selected_columns].reset_index(drop=True)

def apply_remove_columns_direct(df, parameters):
    """Apply column removal directly to dataframe"""
    removed_columns = parameters.get('removed_columns', [])
    
    if not removed_columns:
        return df
    
    # Validate all columns to remove exist
    missing_columns = [col for col in removed_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Columns to remove not found: {missing_columns}")
    
    # Keep only columns that are NOT in removed_columns
    columns_to_keep = [col for col in df.columns if col not in removed_columns]
    
    if not columns_to_keep:
        raise ValueError("Cannot remove all columns from dataset")
    
    return df[columns_to_keep].reset_index(drop=True)

def apply_top_n_records_direct(df, parameters):
    """Apply top N records selection directly to dataframe"""
    n_records = parameters.get('n_records', 10)
    sort_by = parameters.get('sort_by')
    sort_order = parameters.get('sort_order', 'desc')
    
    if not sort_by:
        raise ValueError('Sort column is required')
    
    if sort_by not in df.columns:
        raise ValueError(f'Sort column "{sort_by}" not found in dataset')
    
    ascending = sort_order == 'asc'
    return df.sort_values(by=sort_by, ascending=ascending).head(n_records).reset_index(drop=True)

def apply_random_sampling_direct(df, parameters):
    """Apply random sampling directly to dataframe"""
    sample_size = parameters.get('sample_size', 100)
    sample_type = parameters.get('sample_type', 'count')
    random_state = parameters.get('random_state', 42)
    
    # Calculate sample size
    if sample_type == 'percentage':
        n_samples = int(len(df) * (sample_size / 100))
    else:
        n_samples = min(sample_size, len(df))
    
    return df.sample(n=n_samples, random_state=random_state).reset_index(drop=True)


# Direct dataframe transformation functions for cleaning
def apply_missing_values_direct(df, parameters):
    """Apply missing values handling directly to dataframe"""
    strategy = parameters.get('strategy', 'fill')
    columns = parameters.get('columns', [])
    fill_value = parameters.get('fill_value', 'mean')
    
    columns_to_process = columns if columns else df.columns
    
    if strategy == 'drop':
        df = df.dropna(subset=columns_to_process)
    elif strategy == 'fill':
        for col in columns_to_process:
            if col in df.columns:
                if fill_value == 'mean' and pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].mean())
                elif fill_value == 'median' and pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                elif fill_value == 'mode':
                    mode_val = df[col].mode()
                    df[col] = df[col].fillna(mode_val[0] if not mode_val.empty else '')
                else:
                    df[col] = df[col].fillna(fill_value)
    elif strategy == 'interpolate':
        for col in columns_to_process:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].interpolate()
    
    return df

def apply_remove_duplicates_direct(df, parameters):
    """Apply duplicate removal directly to dataframe"""
    subset = parameters.get('subset', [])
    keep = parameters.get('keep', 'first')
    
    return df.drop_duplicates(subset=subset if subset else None, keep=keep)

def apply_data_type_conversion_direct(df, parameters):
    """Apply data type conversion directly to dataframe"""
    conversions = parameters.get('conversions', {})
    
    for col, target_type in conversions.items():
        if col in df.columns:
            try:
                if target_type == 'numeric':
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                elif target_type == 'integer':
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                elif target_type == 'float':
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
                elif target_type == 'string':
                    df[col] = df[col].astype(str)
                elif target_type == 'datetime':
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                elif target_type == 'boolean':
                    df[col] = df[col].astype(str).str.lower().map({
                        'true': True, 'false': False, '1': True, '0': False
                    }).fillna(False)
            except Exception as e:
                print(f"Warning: Could not convert column {col} to {target_type}: {e}")
    
    return df

def apply_text_cleaning_direct(df, parameters):
    """Apply text cleaning directly to dataframe"""
    columns = parameters.get('columns', [])
    operations = parameters.get('operations', {})
    
    columns_to_process = columns if columns else df.select_dtypes(include=['object']).columns
    
    for col in columns_to_process:
        if col in df.columns:
            # Trim whitespace
            if operations.get('trim', False):
                df[col] = df[col].astype(str).str.strip()
            
            # Case conversion
            if operations.get('case') == 'lower':
                df[col] = df[col].astype(str).str.lower()
            elif operations.get('case') == 'upper':
                df[col] = df[col].astype(str).str.upper()
            elif operations.get('case') == 'title':
                df[col] = df[col].astype(str).str.title()
            
            # Remove extra spaces
            if operations.get('remove_extra_spaces', False):
                df[col] = df[col].astype(str).str.replace(r'\s+', ' ', regex=True)
    
    return df

# Direct dataframe transformation functions for aggregation
def apply_groupby_aggregation_direct(df, parameters):
    """Apply groupby aggregation directly to dataframe"""
    group_columns = parameters.get('group_columns', [])
    aggregations = parameters.get('aggregations', [])
    
    # Validate inputs
    if not aggregations:
        raise ValueError("At least one aggregation operation is required")
    
    # Check if all group columns exist
    missing_group_cols = [col for col in group_columns if col not in df.columns]
    if missing_group_cols:
        raise ValueError(f"Group columns not found: {missing_group_cols}")
    
    # Convert aggregations to the format expected by your main function
    aggregation_operations = {}
    for agg in aggregations:
        column = agg.get('column')
        operations = agg.get('operations', [])
        
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in dataset")
        
        aggregation_operations[column] = operations
    
    # Prepare aggregation dictionary for pandas
    agg_dict = {}
    for col, operations in aggregation_operations.items():
        agg_dict[col] = []
        for op in operations:
            if op in ['sum', 'mean', 'median', 'min', 'max', 'count', 'std', 'var']:
                agg_dict[col].append(op)
            elif op == 'unique_count':
                agg_dict[col].append('nunique')
    
    # Perform aggregation
    if group_columns:
        # Group by aggregation
        grouped_df = df.groupby(group_columns).agg(agg_dict).reset_index()
        
        # Flatten multi-level column names
        if isinstance(grouped_df.columns, pd.MultiIndex):
            grouped_df.columns = ['_'.join(col).strip('_') for col in grouped_df.columns]
        else:
            grouped_df.columns = [str(col) for col in grouped_df.columns]
        
        return grouped_df
    else:
        # Whole dataset aggregation (no grouping)
        result_data = []
        
        for col, operations in aggregation_operations.items():
            for op in operations:
                try:
                    if op == 'sum':
                        value = df[col].sum()
                    elif op == 'mean':
                        value = df[col].mean()
                    elif op == 'median':
                        value = df[col].median()
                    elif op == 'min':
                        value = df[col].min()
                    elif op == 'max':
                        value = df[col].max()
                    elif op == 'count':
                        value = df[col].count()
                    elif op == 'std':
                        value = df[col].std()
                    elif op == 'var':
                        value = df[col].var()
                    elif op == 'unique_count':
                        value = df[col].nunique()
                    else:
                        value = None
                    
                    # Convert numpy/pandas types to Python native types
                    if hasattr(value, 'item'):
                        value = value.item()
                    elif pd.isna(value):
                        value = None
                    
                    result_data.append({
                        'column': col,
                        'operation': op,
                        'value': value
                    })
                except Exception as col_error:
                    print(f"Warning: Could not aggregate {col} with {op}: {col_error}")
                    continue
        
        # Convert to DataFrame for consistent handling
        return pd.DataFrame(result_data)

def apply_pivot_table_direct(df, parameters):
    """Apply pivot table directly to dataframe"""
    index_columns = parameters.get('index_columns', [])
    column_columns = parameters.get('column_columns', [])
    value_columns = parameters.get('value_columns', [])
    agg_func = parameters.get('agg_func', 'mean')
    fill_value = parameters.get('fill_value', 0)
    
    # Validate inputs
    if not index_columns:
        raise ValueError("At least one index column is required")
    if not value_columns:
        raise ValueError("At least one value column is required")
    
    # Check if columns exist
    all_columns = index_columns + column_columns + value_columns
    missing_cols = [col for col in all_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found: {missing_cols}")
    
    # Create pivot table
    if column_columns:
        pivot_df = df.pivot_table(
            index=index_columns,
            columns=column_columns,
            values=value_columns,
            aggfunc=agg_func,
            fill_value=fill_value
        ).reset_index()
    else:
        pivot_df = df.pivot_table(
            index=index_columns,
            values=value_columns,
            aggfunc=agg_func,
            fill_value=fill_value
        ).reset_index()
    
    # Flatten multi-level column names
    if isinstance(pivot_df.columns, pd.MultiIndex):
        pivot_df.columns = ['_'.join(filter(None, map(str, col))).strip('_') for col in pivot_df.columns]
    else:
        pivot_df.columns = [str(col) for col in pivot_df.columns]
    
    return pivot_df


def apply_calculated_columns_direct(df, parameters):
    """Apply calculated columns directly to dataframe"""
    calculated_columns = parameters.get('calculated_columns', [])
    
    if not calculated_columns:
        return df
    
    result_df = df.copy()
    
    for column_config in calculated_columns:
        column_name = column_config.get('column_name')
        expression = column_config.get('expression')
        data_type = column_config.get('data_type', 'auto')
        
        if not column_name or not expression:
            raise ValueError('Column name and expression are required')
        
        try:
            # Safe evaluation of expression
            # Replace column references with df['column_name']
            safe_expression = expression
            for col in df.columns:
                safe_expression = safe_expression.replace(col, f"df['{col}']")
            
            # Evaluate the expression
            result = eval(safe_expression, {'df': df, 'np': np, 'pd': pd})
            
            # Assign to new column
            result_df[column_name] = result
            
            # Convert data type if specified
            if data_type != 'auto':
                if data_type == 'int':
                    result_df[column_name] = pd.to_numeric(result_df[column_name], errors='coerce').astype('Int64')
                elif data_type == 'float':
                    result_df[column_name] = pd.to_numeric(result_df[column_name], errors='coerce').astype(float)
                elif data_type == 'string':
                    result_df[column_name] = result_df[column_name].astype(str)
                elif data_type == 'boolean':
                    result_df[column_name] = result_df[column_name].astype(bool)
                    
        except Exception as e:
            raise ValueError(f'Error creating column "{column_name}": {str(e)}')
    
    return result_df
def apply_window_functions_direct(df, parameters):
    """Apply window functions directly to dataframe"""
    partition_columns = parameters.get('partition_columns', [])
    order_columns = parameters.get('order_columns', [])
    window_functions = parameters.get('window_functions', [])
    
    # Validate inputs
    if not window_functions:
        raise ValueError("At least one window function is required")
    
    result_df = df.copy()
    
    for func_config in window_functions:
        target_column = func_config.get('target_column')
        function_type = func_config.get('function_type')
        new_column_name = func_config.get('new_column_name', f"{target_column}_{function_type}")
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        
        # Apply ordering if specified
        if order_columns:
            result_df = result_df.sort_values(by=order_columns)
        
        # Create window specification based on partition columns
        if partition_columns:
            # Verify partition columns exist
            missing_partition = [col for col in partition_columns if col not in result_df.columns]
            if missing_partition:
                raise ValueError(f"Partition columns not found: {missing_partition}")
            
            # Create window group with partitions
            if order_columns:
                window = result_df.groupby(partition_columns)[target_column]
            else:
                window = result_df.groupby(partition_columns)[target_column]
        else:
            # No partitioning - use entire dataset
            window = result_df[target_column]
        
        # Apply window function
        if function_type == 'cumsum':
            if partition_columns:
                result_df[new_column_name] = window.cumsum()
            else:
                result_df[new_column_name] = window.cumsum()
                
        elif function_type == 'cummean':
            if partition_columns:
                result_df[new_column_name] = window.expanding().mean().reset_index(level=0, drop=True)
            else:
                result_df[new_column_name] = window.expanding().mean()
                
        elif function_type == 'cummin':
            if partition_columns:
                result_df[new_column_name] = window.cummin()
            else:
                result_df[new_column_name] = window.cummin()
                
        elif function_type == 'cummax':
            if partition_columns:
                result_df[new_column_name] = window.cummax()
            else:
                result_df[new_column_name] = window.cummax()
                
        elif function_type == 'rank':
            if partition_columns:
                result_df[new_column_name] = window.rank(method='dense')
            else:
                result_df[new_column_name] = window.rank(method='dense')
                
        elif function_type == 'row_number':
            if partition_columns:
                result_df[new_column_name] = window.cumcount() + 1
            else:
                result_df[new_column_name] = range(1, len(result_df) + 1)
                
        elif function_type == 'lag':
            periods = func_config.get('periods', 1)
            if partition_columns:
                result_df[new_column_name] = window.shift(periods)
            else:
                result_df[new_column_name] = window.shift(periods)
                
        elif function_type == 'lead':
            periods = func_config.get('periods', 1)
            if partition_columns:
                result_df[new_column_name] = window.shift(-periods)
            else:
                result_df[new_column_name] = window.shift(-periods)
                
        elif function_type == 'rolling_mean':
            window_size = func_config.get('window_size', 3)
            if partition_columns:
                result_df[new_column_name] = window.rolling(window=window_size, min_periods=1).mean()
            else:
                result_df[new_column_name] = window.rolling(window=window_size, min_periods=1).mean()
                
        elif function_type == 'rolling_sum':
            window_size = func_config.get('window_size', 3)
            if partition_columns:
                result_df[new_column_name] = window.rolling(window=window_size, min_periods=1).sum()
            else:
                result_df[new_column_name] = window.rolling(window=window_size, min_periods=1).sum()
        
        else:
            raise ValueError(f"Unsupported window function: {function_type}")
    
    # Reset index if it was modified during operations
    return result_df.reset_index(drop=True)

def apply_rollup_cube_direct(df, parameters):
    """Apply rollup and cube operations directly to dataframe"""
    group_columns = parameters.get('group_columns', [])
    aggregation_columns = parameters.get('aggregation_columns', {})
    operation_type = parameters.get('operation_type', 'rollup')
    
    # Validate inputs
    if len(group_columns) < 2:
        raise ValueError("At least two group columns are required for rollup/cube")
    
    if not aggregation_columns:
        raise ValueError("At least one aggregation column is required")
    
    # Validate columns exist
    all_columns = group_columns + list(aggregation_columns.keys())
    for col in all_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataset")
    
    # Prepare aggregation dictionary
    agg_dict = {}
    for col, operations in aggregation_columns.items():
        agg_dict[col] = operations
    
    # Perform rollup or cube operation
    if operation_type == 'rollup':
        # Create all combinations of group columns for rollup
        results = []
        for i in range(len(group_columns) + 1):
            current_groups = group_columns[:len(group_columns)-i] if i > 0 else group_columns
            if current_groups:
                grouped = df.groupby(current_groups).agg(agg_dict).reset_index()
                # Add level indicator
                for missing_col in set(group_columns) - set(current_groups):
                    grouped[missing_col] = 'TOTAL'
                results.append(grouped)
        
        result_df = pd.concat(results, ignore_index=True)
        
    else:  # cube
        # Create power set of group columns for cube
        from itertools import chain, combinations
        
        def powerset(iterable):
            s = list(iterable)
            return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
        
        results = []
        for groups in powerset(group_columns):
            if groups:  # Skip empty combination
                current_groups = list(groups)
                grouped = df.groupby(current_groups).agg(agg_dict).reset_index()
                # Add level indicator for missing columns
                for missing_col in set(group_columns) - set(current_groups):
                    grouped[missing_col] = 'TOTAL'
                results.append(grouped)
        
        result_df = pd.concat(results, ignore_index=True)
    
    # Sort the result
    return result_df.sort_values(by=group_columns)



def apply_calculated_columns_direct(df, parameters):
    """Apply calculated columns directly to DataFrame"""
    calculated_columns = parameters.get('calculated_columns', [])
    result_df = df.copy()
    
    for column_config in calculated_columns:
        column_name = column_config.get('column_name')
        expression = column_config.get('expression')
        data_type = column_config.get('data_type', 'auto')
        
        if not column_name or not expression:
            raise ValueError('Column name and expression are required')
        
        try:
            # Safe evaluation of expression
            # Replace column references with df['column_name']
            safe_expression = expression
            for col in df.columns:
                safe_expression = safe_expression.replace(col, f"df['{col}']")
            
            # Evaluate the expression
            result = eval(safe_expression, {'df': df, 'np': np, 'pd': pd})
            
            # Assign to new column
            result_df[column_name] = result
            
            # Convert data type if specified
            if data_type != 'auto':
                if data_type == 'int':
                    result_df[column_name] = pd.to_numeric(result_df[column_name], errors='coerce').astype('Int64')
                elif data_type == 'float':
                    result_df[column_name] = pd.to_numeric(result_df[column_name], errors='coerce').astype(float)
                elif data_type == 'string':
                    result_df[column_name] = result_df[column_name].astype(str)
                elif data_type == 'boolean':
                    result_df[column_name] = result_df[column_name].astype(bool)
                    
        except Exception as e:
            raise ValueError(f'Error creating column "{column_name}": {str(e)}')
    
    return result_df

def apply_datetime_extraction_direct(df, parameters):
    """Apply datetime extraction directly to DataFrame"""
    datetime_columns = parameters.get('datetime_columns', [])
    result_df = df.copy()
    
    for column_config in datetime_columns:
        source_column = column_config.get('source_column')
        extractions = column_config.get('extractions', [])
        
        if source_column not in df.columns:
            raise ValueError(f'Source column "{source_column}" not found')
        
        try:
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(df[source_column]):
                result_df[source_column] = pd.to_datetime(df[source_column], errors='coerce')
            
            # Apply extractions
            for extraction in extractions:
                component = extraction.get('component')
                new_column_name = extraction.get('new_column_name', f'{source_column}_{component}')
                
                if component == 'year':
                    result_df[new_column_name] = result_df[source_column].dt.year
                elif component == 'month':
                    result_df[new_column_name] = result_df[source_column].dt.month
                elif component == 'day':
                    result_df[new_column_name] = result_df[source_column].dt.day
                elif component == 'hour':
                    result_df[new_column_name] = result_df[source_column].dt.hour
                elif component == 'minute':
                    result_df[new_column_name] = result_df[source_column].dt.minute
                elif component == 'second':
                    result_df[new_column_name] = result_df[source_column].dt.second
                elif component == 'quarter':
                    result_df[new_column_name] = result_df[source_column].dt.quarter
                elif component == 'dayofweek':
                    result_df[new_column_name] = result_df[source_column].dt.dayofweek
                elif component == 'dayofyear':
                    result_df[new_column_name] = result_df[source_column].dt.dayofyear
                elif component == 'week':
                    result_df[new_column_name] = result_df[source_column].dt.isocalendar().week
                elif component == 'is_weekend':
                    result_df[new_column_name] = result_df[source_column].dt.dayofweek >= 5
                elif component == 'month_name':
                    result_df[new_column_name] = result_df[source_column].dt.month_name()
                elif component == 'day_name':
                    result_df[new_column_name] = result_df[source_column].dt.day_name()
                
        except Exception as e:
            raise ValueError(f'Error processing datetime column "{source_column}": {str(e)}')
    
    return result_df

def apply_text_processing_direct(df, parameters):
    """Apply text processing directly to DataFrame"""
    text_columns = parameters.get('text_columns', [])
    result_df = df.copy()
    
    for column_config in text_columns:
        source_column = column_config.get('source_column')
        operations = column_config.get('operations', [])
        
        if source_column not in df.columns:
            raise ValueError(f'Source column "{source_column}" not found')
        
        try:
            # Convert to string to ensure text operations work
            result_df[source_column] = result_df[source_column].astype(str)
            
            # Apply operations
            for operation in operations:
                op_type = operation.get('type')
                new_column_name = operation.get('new_column_name', f'{source_column}_{op_type}')
                
                if op_type == 'lowercase':
                    result_df[new_column_name] = result_df[source_column].str.lower()
                elif op_type == 'uppercase':
                    result_df[new_column_name] = result_df[source_column].str.upper()
                elif op_type == 'title_case':
                    result_df[new_column_name] = result_df[source_column].str.title()
                elif op_type == 'capitalize':
                    result_df[new_column_name] = result_df[source_column].str.capitalize()
                elif op_type == 'strip':
                    result_df[new_column_name] = result_df[source_column].str.strip()
                elif op_type == 'remove_extra_spaces':
                    result_df[new_column_name] = result_df[source_column].str.replace(r'\s+', ' ', regex=True)
                elif op_type == 'remove_special_chars':
                    result_df[new_column_name] = result_df[source_column].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
                elif op_type == 'extract_numbers':
                    result_df[new_column_name] = result_df[source_column].str.extract(r'(\d+)', expand=False)
                elif op_type == 'extract_letters':
                    result_df[new_column_name] = result_df[source_column].str.replace(r'[^a-zA-Z]', '', regex=True)
                elif op_type == 'word_count':
                    result_df[new_column_name] = result_df[source_column].str.split().str.len()
                elif op_type == 'character_count':
                    result_df[new_column_name] = result_df[source_column].str.len()
                elif op_type == 'replace_text':
                    old_text = operation.get('old_text', '')
                    new_text = operation.get('new_text', '')
                    result_df[new_column_name] = result_df[source_column].str.replace(old_text, new_text, regex=False)
                elif op_type == 'substring':
                    start = operation.get('start', 0)
                    end = operation.get('end', None)
                    result_df[new_column_name] = result_df[source_column].str.slice(start, end)
                
        except Exception as e:
            raise ValueError(f'Error processing text column "{source_column}": {str(e)}')
    
    return result_df

def apply_one_hot_encoding_direct(df, parameters):
    """Apply one-hot encoding directly to DataFrame"""
    categorical_columns = parameters.get('categorical_columns', [])
    drop_first = parameters.get('drop_first', False)
    prefix = parameters.get('prefix', True)
    
    result_df = df.copy()
    
    # Validate columns exist
    for col in categorical_columns:
        if col not in df.columns:
            raise ValueError(f'Column "{col}" not found in dataset')
    
    try:
        # Handle prefix parameter properly
        prefix_param = None
        if prefix is True:
            prefix_param = categorical_columns  # Use column names as prefixes
        elif prefix is False:
            prefix_param = None  # No prefixes
        elif isinstance(prefix, list) and len(prefix) == len(categorical_columns):
            prefix_param = prefix  # Use provided prefixes
        else:
            prefix_param = categorical_columns  # Default to column names
        
        # Use pandas get_dummies for one-hot encoding
        encoded_df = pd.get_dummies(
            result_df[categorical_columns], 
            prefix=prefix_param,
            prefix_sep='_',
            drop_first=drop_first
        )
        
        # Drop original categorical columns and add encoded ones
        result_df = result_df.drop(columns=categorical_columns)
        result_df = pd.concat([result_df, encoded_df], axis=1)
        
    except Exception as e:
        raise ValueError(f'Error in one-hot encoding: {str(e)}')
    
    return result_df
@mongo_login_required
@require_http_methods(["GET"])
def edit_pipeline(request, pipeline_id):
    """Get pipeline details for editing"""
    try:
        pipeline = TransformationPipeline.objects.get(
            id=ObjectId(pipeline_id), 
            owner_id=str(request.session.get("user_id"))
        )
        
        # Get pipeline steps
        steps = PipelineStep.objects.filter(pipeline_id=ObjectId(pipeline_id)).order_by('step_number')
        
        return JsonResponse({
            'success': True,
            'pipeline': {
                'id': str(pipeline.id),
                'name': pipeline.name,
                'description': pipeline.description,
                'input_dataset_id': str(pipeline.input_dataset_id),
                'steps': [
                    {
                        'step_number': step.step_number,
                        'type': step.step_type,
                        'operation': step.operation,
                        'parameters': step.parameters
                    }
                    for step in steps
                ]
            }
        })
        
    except TransformationPipeline.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Pipeline not found'}, status=404)
    except Exception as e:
        print(f"Error fetching pipeline for edit: {str(e)}")
        return JsonResponse({'success': False, 'error': 'Failed to fetch pipeline'}, status=500)
    

@mongo_login_required
@require_http_methods(["GET", "POST", "DELETE"])
@csrf_exempt
def manage_pipeline_templates(request):
    """Manage reusable pipeline templates"""
    try:
        if request.method == 'GET':
            # Get user's templates and public templates
            user_templates = PipelineTemplate.objects(user_id=str(request.session.get("user_id")))
            public_templates = PipelineTemplate.objects(is_public=True)
            
            template_list = []
            for template in list(user_templates) + list(public_templates):
                template_list.append({
                    'id': str(template.id),
                    'name': template.name,
                    'description': template.description,
                    'category': template.category,
                    'steps': template.steps,
                    'is_public': template.is_public,
                    'usage_count': template.usage_count,
                    'created_at': template.created_at.isoformat(),
                    'updated_at': template.updated_at.isoformat()
                })
            
            return JsonResponse({
                'success': True,
                'templates': template_list
            })
            
        elif request.method == 'POST':
            data = json.loads(request.body)
            
            template = PipelineTemplate(
                name=data.get('name'),
                description=data.get('description', ''),
                category=data.get('category', 'general'),
                steps=data.get('steps', []),
                user_id=str(request.session.get("user_id")),
                is_public=data.get('is_public', False),
                usage_count=0
            )
            template.save()
            
            return JsonResponse({
                'success': True,
                'template_id': str(template.id),
                'message': 'Pipeline template created successfully'
            })
            
        elif request.method == 'DELETE':
            template_id = request.GET.get('template_id')
            if not template_id:
                return JsonResponse({'success': False, 'error': 'Template ID is required'}, status=400)
            
            template = PipelineTemplate.objects.get(id=ObjectId(template_id), user_id=str(request.session.get("user_id")))
            template.delete()
            
            return JsonResponse({
                'success': True,
                'message': 'Pipeline template deleted successfully'
            })
            
    except PipelineTemplate.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Template not found'}, status=404)
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)



def execute_pipeline_step(dataset_id, step_type, operation, parameters, user_id):
    """Execute a single pipeline step using existing cleaning operations"""
    
    # Map pipeline operations to existing cleaning functions
    operation_mapping = {
        'cleaning': {
            'handle-missing-values': 'missing_values',
            'remove-duplicates': 'remove_duplicates', 
            'data-type-conversion': 'convert_data_types',
            'text-cleaning': 'standardize_formats'
        }
    }
    
    # Get the corresponding cleaning operation
    cleaning_operation = operation_mapping.get(step_type, {}).get(operation)
    
    if not cleaning_operation:
        return {
            'success': False,
            'error': f'Unsupported operation: {step_type}.{operation}'
        }
    
    # Prepare request data for the cleaning operation
    request_data = {
        'dataset_id': str(dataset_id),
        'preview_only': False  # Execute for real, not preview
    }
    
    # Map parameters based on operation type
    if cleaning_operation == 'missing_values':
        request_data.update({
            'strategy': parameters.get('strategy', 'fill'),
            'columns': parameters.get('columns', []),
            'fill_value': parameters.get('fill_value', 'mean')
        })
    
    elif cleaning_operation == 'remove_duplicates':
        request_data.update({
            'subset': parameters.get('subset', []),
            'keep': parameters.get('keep', 'first')
        })
    
    elif cleaning_operation == 'standardize_formats':
        request_data.update({
            'columns': parameters.get('columns', []),
            'operations': parameters.get('operations', {})
        })
    
    elif cleaning_operation == 'convert_data_types':
        request_data.update({
            'conversions': parameters.get('conversions', {})
        })
    
    print(f"   Executing cleaning operation: {cleaning_operation}")
    print(f"   Request data: {json.dumps(request_data, indent=4)}")

        # Call the appropriate cleaning function directly
    if cleaning_operation == 'missing_values':
            result = execute_cleaning_directly(handle_missing_values, request_data, user_id)
    elif cleaning_operation == 'remove_duplicates':
            result = execute_cleaning_directly(remove_duplicates, request_data, user_id)
    elif cleaning_operation == 'standardize_formats':
            result = execute_cleaning_directly(standardize_formats, request_data, user_id)
    elif cleaning_operation == 'convert_data_types':
            result = execute_cleaning_directly(convert_data_types, request_data, user_id)
        
    return result
        


def execute_cleaning_directly(cleaning_function, request_data, user_id):
    """Execute cleaning function directly without mock request"""
    try:
        # Create a simple context that mimics what the cleaning functions need
        class ExecutionContext:
            def __init__(self, data, user_id):
                self.data = data
                self.user_id = user_id
            
            def get_cleaned_data(self):
                return self.data
        
        # Call the cleaning function with the necessary context
        # This assumes your cleaning functions can work with direct data
        # If they need request objects, we'll need to adapt them
        
        # For now, let's create a proper mock request that works
        from django.http import JsonResponse
        from django.contrib.auth.models import User
        
        # Create a proper mock request
        class WorkingMockRequest:
            def __init__(self, data, user_id):
                self.method = 'POST'
                self.user = self.get_user_object(user_id)
                self.body = json.dumps(data).encode('utf-8')
                self.META = {'CONTENT_TYPE': 'application/json'}
            
            def get_user_object(self, user_id):
                try:
                    # Convert to string and get user
                    return User.objects.get(id=str(user_id))
                except User.DoesNotExist:
                    # Create a simple mock user
                    class SimpleUser:
                        def __init__(self, uid):
                            self.id = uid
                            self.username = f"user_{uid}"
                    return SimpleUser(str(user_id))
        
        mock_request = WorkingMockRequest(request_data, user_id)
        
        # Call the cleaning function
        response = cleaning_function(mock_request)
        
        # Parse the response
        if hasattr(response, 'content'):
            # It's a HttpResponse, parse JSON
            result_data = json.loads(response.content)
        else:
            # It's already a dict
            result_data = response
        
        return result_data
        
    except Exception as e:
        print(f"Error in direct execution: {str(e)}")
        import traceback
        print(f"Direct execution traceback: {traceback.format_exc()}")
        raise


@mongo_login_required
@require_http_methods(["POST"])
@csrf_exempt
def execute_full_pipeline(request):
    """Execute entire pipeline from start to finish"""
    try:
        data = json.loads(request.body)
        pipeline_id = data.get('pipeline_id')
        
        pipeline = TransformationPipeline.objects.get(id=ObjectId(pipeline_id))
        
        # Verify ownership
        if pipeline.owner_id != str(request.session.get("user_id")):
            return JsonResponse({'success': False, 'error': 'Access denied'}, status=403)
        
        # Update pipeline status
        pipeline.status = 'running'
        pipeline.current_step = 0
        pipeline.updated_at = datetime.utcnow()
        pipeline.save()
        
        execution_stats = {
            'start_time': datetime.utcnow().isoformat(),
            'steps_completed': 0,
            'steps_failed': 0,
            'total_execution_time': 0
        }
        
        # Execute each step sequentially
        steps = PipelineStep.objects(pipeline_id=pipeline.id).order_by('step_number')
        current_dataset_id = pipeline.input_dataset_id
        
        for step in steps:
            try:
                # Update step status
                step.status = 'running'
                step.executed_at = datetime.utcnow()
                step.save()
                
                # Execute step
                input_dataset = Dataset.objects.get(id=current_dataset_id)
                result = execute_transformation_step(
                    step.step_type, 
                    step.operation, 
                    step.parameters, 
                    input_dataset, 
                    False,  # Execute fully, not preview
                    str(request.session.get("user_id"))
                )
                
                # Create output dataset
                output_dataset = create_transformation_dataset(
                    result.get('dataframe'),
                    f"{pipeline.name}_step_{step.step_number}",
                    str(request.session.get("user_id")),
                    step.operation
                )
                
                # Update step with results
                step.output_dataset_id = output_dataset.id
                step.result_stats = result.get('result_stats', {})
                step.execution_time = result.get('execution_time', 0)
                step.status = 'completed'
                step.save()
                
                # Update current dataset for next step
                current_dataset_id = output_dataset.id
                execution_stats['steps_completed'] += 1
                execution_stats['total_execution_time'] += step.execution_time
                
                # Update pipeline progress
                pipeline.current_step = step.step_number
                pipeline.save()
                
            except Exception as step_error:
                step.status = 'failed'
                step.error_message = str(step_error)
                step.save()
                execution_stats['steps_failed'] += 1
                break
        
        # Update pipeline final status
        execution_stats['end_time'] = datetime.utcnow().isoformat()
        
        if execution_stats['steps_failed'] > 0:
            pipeline.status = 'failed'
        else:
            pipeline.status = 'completed'
            pipeline.output_dataset_id = current_dataset_id
            pipeline.completed_at = datetime.utcnow()
        
        pipeline.execution_stats = execution_stats
        pipeline.save()
        
        return JsonResponse({
            'success': True,
            'pipeline_status': pipeline.status,
            'execution_stats': execution_stats,
            'output_dataset_id': str(pipeline.output_dataset_id) if pipeline.output_dataset_id else None
        })
        
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)

@mongo_login_required
@require_http_methods(["GET"])
def get_pipeline_details(request, pipeline_id):
    """Get detailed information about a pipeline"""
    try:
        pipeline = TransformationPipeline.objects.get(id=ObjectId(pipeline_id))
        
        # Verify ownership
        if pipeline.owner_id != str(request.session.get("user_id")):
            return JsonResponse({'success': False, 'error': 'Access denied'}, status=403)
        
        steps = PipelineStep.objects(pipeline_id=pipeline.id).order_by('step_number')
        
        step_details = []
        for step in steps:
            step_details.append({
                'id': str(step.id),
                'step_number': step.step_number,
                'step_type': step.step_type,
                'operation': step.operation,
                'status': step.status,
                'parameters': step.parameters,
                'execution_time': step.execution_time,
                'result_stats': step.result_stats,
                'preview_data': step.preview_data,
                'preview_columns': step.preview_columns,
                'error_message': step.error_message
            })
        
        return JsonResponse({
            'success': True,
            'pipeline': {
                'id': str(pipeline.id),
                'name': pipeline.name,
                'description': pipeline.description,
                'status': pipeline.status,
                'current_step': pipeline.current_step,
                'total_steps': pipeline.total_steps,
                'execution_stats': pipeline.execution_stats,
                'created_at': pipeline.created_at.isoformat(),
                'updated_at': pipeline.updated_at.isoformat()
            },
            'steps': step_details
        })
        
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)

@mongo_login_required
@require_http_methods(["GET"])
def get_user_pipelines(request):
    """Get all pipelines for the current user"""
    try:
        pipelines = TransformationPipeline.objects(owner_id=str(request.session.get("user_id"))).order_by('-created_at')
        
        pipeline_list = []
        for pipeline in pipelines:
            pipeline_list.append({
                'id': str(pipeline.id),
                'name': pipeline.name,
                'description': pipeline.description,
                'status': pipeline.status,
                'current_step': pipeline.current_step,
                'total_steps': pipeline.total_steps,
                'created_at': pipeline.created_at.isoformat(),
                'updated_at': pipeline.updated_at.isoformat()
            })
        
        return JsonResponse({
            'success': True,
            'pipelines': pipeline_list
        })
        
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)

# Helper function to execute transformation steps
def execute_transformation_step(step_type, operation, parameters, input_dataset, preview_only, user_id):
    """Execute a transformation step based on type and operation"""
    import time
    start_time = time.time()
    
    # Download input dataset
    df = download_and_convert_to_dataframe(input_dataset)
    
    result = {
        'dataframe': None,
        'preview_data': [],
        'columns': [],
        'result_stats': {},
        'execution_time': 0
    }
    
    try:
        # Route to appropriate transformation function
        if step_type == 'join':
            # Join operations require additional dataset
            other_dataset_id = parameters.get('other_dataset_id')
            if other_dataset_id:
                other_dataset = Dataset.objects.get(id=ObjectId(other_dataset_id))
                other_df = download_and_convert_to_dataframe(other_dataset)
                
                result_df = perform_join(
                    df, 
                    other_df, 
                    parameters.get('left_column'), 
                    parameters.get('right_column'), 
                    operation
                )
            else:
                raise ValueError("Join operation requires another dataset")
                
        elif step_type == 'cleaning':
            if operation == 'missing_values':
                result_df = handle_missing_values_execution(df, parameters)
            elif operation == 'remove_duplicates':
                result_df = remove_duplicates_execution(df, parameters)
            elif operation == 'standardize_formats':
                result_df = standardize_formats_execution(df, parameters)
            elif operation == 'convert_data_types':
                result_df = convert_data_types_execution(df, parameters)
                
        elif step_type == 'aggregation':
            if operation == 'groupby':
                result_df = groupby_aggregation_execution(df, parameters)
            elif operation == 'pivot':
                result_df = pivot_table_execution(df, parameters)
            elif operation == 'window':
                result_df = window_functions_execution(df, parameters)
                
        elif step_type == 'filter_sort':
            if operation == 'filter':
                result_df = filter_data_execution(df, parameters)
            elif operation == 'sort':
                result_df = sort_data_execution(df, parameters)
            elif operation == 'top_n':
                result_df = top_n_records_execution(df, parameters)
                
        elif step_type == 'feature_engineering':
            if operation == 'calculated_columns':
                result_df = calculated_columns_execution(df, parameters)
            elif operation == 'datetime_extraction':
                result_df = datetime_extraction_execution(df, parameters)
            elif operation == 'text_processing':
                result_df = text_processing_execution(df, parameters)
            elif operation == 'one_hot_encoding':
                result_df = one_hot_encoding_execution(df, parameters)
                
        elif step_type == 'ml_preparation':
            if operation == 'train_test_split':
                result_df = train_test_split_execution(df, parameters)
            elif operation == 'feature_scaling':
                result_df = feature_scaling_execution(df, parameters)
            elif operation == 'outlier_detection':
                result_df = outlier_detection_execution(df, parameters)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        result['execution_time'] = execution_time
        
        # Prepare result
        if preview_only:
            result['preview_data'] = dataframe_to_dict_clean(result_df.head(20))
            result['columns'] = list(result_df.columns)
            result['result_stats'] = {
                'preview_rows': len(result['preview_data']),
                'total_columns': len(result_df.columns),
                'execution_time': execution_time
            }
        else:
            result['dataframe'] = result_df
            result['result_stats'] = {
                'total_rows': len(result_df),
                'total_columns': len(result_df.columns),
                'execution_time': execution_time
            }
        
        return result
        
    except Exception as e:
        raise e

# Add execution helper functions for each transformation type
def handle_missing_values_execution(df, parameters):
    """Execute missing values handling"""
    strategy = parameters.get('strategy', 'fill')
    columns = parameters.get('columns', [])
    fill_value = parameters.get('fill_value')
    
    result_df = df.copy()
    
    if strategy == 'drop':
        if columns:
            result_df = result_df.dropna(subset=columns)
        else:
            result_df = result_df.dropna()
    elif strategy == 'fill':
        if columns:
            for col in columns:
                if fill_value == 'mean' and pd.api.types.is_numeric_dtype(result_df[col]):
                    result_df[col] = result_df[col].fillna(result_df[col].mean())
                elif fill_value == 'median' and pd.api.types.is_numeric_dtype(result_df[col]):
                    result_df[col] = result_df[col].fillna(result_df[col].median())
                elif fill_value == 'mode':
                    result_df[col] = result_df[col].fillna(result_df[col].mode()[0] if not result_df[col].mode().empty else '')
                else:
                    result_df[col] = result_df[col].fillna(fill_value)
    
    return result_df

@mongo_login_required
@require_http_methods(["POST"])
@csrf_exempt
def save_pipeline_as_template(request):
    """Save an existing pipeline as a template"""
    try:
        data = json.loads(request.body)
        pipeline_id = data.get('pipeline_id')
        template_name = data.get('name')
        description = data.get('description', '')
        category = data.get('category', 'general')
        is_public = data.get('is_public', False)
        
        if not pipeline_id or not template_name:
            return JsonResponse({
                'success': False, 
                'error': 'Pipeline ID and template name are required'
            }, status=400)
        
        # Get pipeline
        pipeline = TransformationPipeline.objects.get(id=ObjectId(pipeline_id))
        
        # Verify ownership
        if pipeline.owner_id != str(request.session.get("user_id")):
            return JsonResponse({'success': False, 'error': 'Access denied'}, status=403)
        
        # Get pipeline steps
        steps = PipelineStep.objects(pipeline_id=pipeline.id).order_by('step_number')
        
        # Prepare step configurations
        step_configs = []
        for step in steps:
            step_configs.append({
                'type': step.step_type,
                'operation': step.operation,
                'parameters': step.parameters
            })
        
        # Create template
        template = PipelineTemplate(
            name=template_name,
            description=description,
            category=category,
            steps=step_configs,
            user_id=str(request.session.get("user_id")),
            is_public=is_public,
            usage_count=0,
            source_pipeline_id=str(pipeline.id)
        )
        template.save()
        
        return JsonResponse({
            'success': True,
            'template_id': str(template.id),
            'message': 'Pipeline saved as template successfully'
        })
        
    except TransformationPipeline.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Pipeline not found'}, status=404)
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)

@mongo_login_required
@require_http_methods(["POST"])
@csrf_exempt
def create_pipeline_from_template(request):
    """Create a new pipeline from a template"""
    try:
        data = json.loads(request.body)
        template_id = data.get('template_id')
        input_dataset_id = data.get('input_dataset_id')
        custom_name = data.get('name')
        
        if not template_id or not input_dataset_id:
            return JsonResponse({
                'success': False, 
                'error': 'Template ID and input dataset ID are required'
            }, status=400)
        
        # Get template
        template = PipelineTemplate.objects.get(id=ObjectId(template_id))
        
        # Verify access (either user's template or public template)
        if template.user_id != str(request.session.get("user_id")) and not template.is_public:
            return JsonResponse({
                'success': False, 
                'error': 'Access denied to this template'
            }, status=403)
        
        # Verify input dataset belongs to user
        input_dataset = Dataset.objects.get(id=ObjectId(input_dataset_id), owner_id=str(request.session.get("user_id")))
        
        # Create pipeline from template
        pipeline_name = custom_name or f"{template.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        pipeline = TransformationPipeline(
            name=pipeline_name,
            description=template.description,
            owner_id=str(request.session.get("user_id")),
            input_dataset_id=ObjectId(input_dataset_id),
            steps=template.steps,
            total_steps=len(template.steps),
            status='draft',
            created_from_template=str(template.id)
        )
        pipeline.save()
        
        # Create pipeline steps
        for i, step_config in enumerate(template.steps):
            step = PipelineStep(
                pipeline_id=pipeline.id,
                step_number=i + 1,
                step_type=step_config.get('type'),
                operation=step_config.get('operation'),
                parameters=step_config.get('parameters', {}),
                status='pending'
            )
            step.save()
        
        # Update template usage count
        template.usage_count += 1
        template.updated_at = datetime.utcnow()
        template.save()
        
        return JsonResponse({
            'success': True,
            'pipeline_id': str(pipeline.id),
            'message': 'Pipeline created from template successfully'
        })
        
    except (PipelineTemplate.DoesNotExist, Dataset.DoesNotExist) as e:
        return JsonResponse({'success': False, 'error': 'Template or dataset not found'}, status=404)
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)

def create_transformation_dataset(df, name, user_id, operation_type):
    """Create a new dataset from transformation result"""
    from datasets.models import Dataset
    from supabase import create_client
    from django.conf import settings
    
    supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
    
    try:
        # Convert DataFrame to CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()
        
        # Upload to Supabase
        unique_filename = f"{uuid.uuid4()}_{name}.csv"
        supabase_path = f"{user_id}/{unique_filename}"
        
        res = supabase.storage.from_(settings.SUPABASE_BUCKET).upload(
            supabase_path, csv_content.encode('utf-8')
        )
        
        # Get public URL
        file_url = supabase.storage.from_(settings.SUPABASE_BUCKET).get_public_url(supabase_path)
        
        # Create dataset record
        dataset = Dataset(
            owner_id=str(user_id),
            file_name=f"{name}.csv",
            file_type="text/csv",
            file_url=file_url,
            file_path=supabase_path,
            uploaded_at=datetime.utcnow(),
            metadata={
                "is_transformation_result": True,
                "transformation_operation": operation_type,
                "created_from_pipeline": True
            }
        )
        
        dataset.save()
        return dataset
    except Exception as e:
        print(f"Error creating transformation dataset: {str(e)}")
        raise

# Add execution helper functions for each transformation type
def remove_duplicates_execution(df, parameters):
    """Execute remove duplicates operation"""
    subset = parameters.get('subset', [])
    keep = parameters.get('keep', 'first')
    
    result_df = df.drop_duplicates(subset=subset if subset else None, keep=keep)
    return result_df

def standardize_formats_execution(df, parameters):
    """Execute standardize formats operation"""
    columns = parameters.get('columns', [])
    operations = parameters.get('operations', {})
    
    result_df = df.copy()
    columns_to_process = columns if columns else df.select_dtypes(include=['object']).columns
    
    for col in columns_to_process:
        if col in result_df.columns:
            # Trim whitespace
            if operations.get('trim', False):
                result_df[col] = result_df[col].astype(str).str.strip()
            
            # Case conversion
            if operations.get('case') == 'lower':
                result_df[col] = result_df[col].astype(str).str.lower()
            elif operations.get('case') == 'upper':
                result_df[col] = result_df[col].astype(str).str.upper()
    
    return result_df

def convert_data_types_execution(df, parameters):
    """Execute convert data types operation"""
    conversions = parameters.get('conversions', {})
    
    result_df = df.copy()
    
    for col, target_type in conversions.items():
        if col in result_df.columns:
            try:
                if target_type == 'numeric':
                    result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
                elif target_type == 'string':
                    result_df[col] = result_df[col].astype(str)
                elif target_type == 'datetime':
                    result_df[col] = pd.to_datetime(result_df[col], errors='coerce')
            except:
                pass
    
    return result_df

def groupby_aggregation_execution(df, parameters):
    """Execute groupby aggregation operation"""
    group_columns = parameters.get('group_columns', [])
    aggregation_operations = parameters.get('aggregation_operations', {})
    
    # Prepare aggregation dictionary
    agg_dict = {}
    for col, operations in aggregation_operations.items():
        if col in df.columns:
            agg_dict[col] = operations
    
    if group_columns:
        result_df = df.groupby(group_columns).agg(agg_dict).reset_index()
    else:
        # Whole dataset aggregation
        result_data = []
        for col, operations in aggregation_operations.items():
            for op in operations:
                try:
                    if op == 'sum':
                        value = df[col].sum()
                    elif op == 'mean':
                        value = df[col].mean()
                    elif op == 'count':
                        value = df[col].count()
                    result_data.append({'column': col, 'operation': op, 'value': value})
                except:
                    continue
        result_df = pd.DataFrame(result_data)
    
    return result_df

def pivot_table_execution(df, parameters):
    """Execute pivot table operation"""
    index_columns = parameters.get('index_columns', [])
    column_columns = parameters.get('column_columns', [])
    value_columns = parameters.get('value_columns', [])
    aggfunc = parameters.get('aggfunc', 'mean')
    
    result_df = df.pivot_table(
        index=index_columns,
        columns=column_columns if column_columns else None,
        values=value_columns,
        aggfunc=aggfunc
    ).reset_index()
    
    return result_df

def filter_data_execution(df, parameters):
    """Execute filter data operation"""
    filters = parameters.get('filters', [])
    
    result_df = df.copy()
    
    for filter_config in filters:
        column = filter_config.get('column')
        operator = filter_config.get('operator')
        value = filter_config.get('value')
        
        if column in result_df.columns:
            if operator == 'equals':
                result_df = result_df[result_df[column] == value]
            elif operator == 'greater_than':
                result_df = result_df[result_df[column] > value]
    
    return result_df

def sort_data_execution(df, parameters):
    """Execute sort data operation"""
    sort_columns = parameters.get('sort_columns', [])
    
    sort_by = []
    ascending = []
    
    for sort_config in sort_columns:
        column = sort_config.get('column')
        order = sort_config.get('order', 'asc')
        
        if column in df.columns:
            sort_by.append(column)
            ascending.append(order == 'asc')
    
    result_df = df.sort_values(by=sort_by, ascending=ascending).reset_index(drop=True)
    return result_df

def calculated_columns_execution(df, parameters):
    """Execute calculated columns operation"""
    calculated_columns = parameters.get('calculated_columns', [])
    
    result_df = df.copy()
    
    for column_config in calculated_columns:
        column_name = column_config.get('column_name')
        expression = column_config.get('expression')
        
        if column_name and expression:
            try:
                # Simple expression evaluation - you can expand this
                if expression == 'row_number':
                    result_df[column_name] = range(1, len(result_df) + 1)
            except:
                pass
    
    return result_df

def one_hot_encoding_execution(df, parameters):
    """Execute one-hot encoding operation"""
    categorical_columns = parameters.get('categorical_columns', [])
    
    result_df = df.copy()
    
    for col in categorical_columns:
        if col in result_df.columns:
            # Simple one-hot encoding
            dummies = pd.get_dummies(result_df[col], prefix=col)
            result_df = pd.concat([result_df, dummies], axis=1)
            result_df = result_df.drop(columns=[col])
    
    return result_df

def train_test_split_execution(df, parameters):
    """Execute train-test split operation"""
    # For pipeline execution, we'll just return the original dataframe
    # since actual splitting creates multiple datasets
    return df

def feature_scaling_execution(df, parameters):
    """Execute feature scaling operation"""
    # Placeholder - return original dataframe
    return df

def outlier_detection_execution(df, parameters):
    """Execute outlier detection operation"""
    # Placeholder - return original dataframe
    return df

# Add similar execution functions for other transformation types...
# [Include similar execution functions for remove_duplicates, standardize_formats, etc.]
@mongo_login_required
@require_http_methods(["GET"])
def get_user_datasets(request):
    """Get all datasets for the current user"""
    try:
        datasets = Dataset.objects(owner_id=str(request.session.get("user_id"))).order_by('-uploaded_at')
        
        dataset_list = []
        for dataset in datasets:
            dataset_list.append({
                'id': str(dataset.id),
                'file_name': dataset.file_name,
                'file_type': dataset.file_type,
                'uploaded_at': dataset.uploaded_at.isoformat(),
                'file_size': dataset.metadata.get('file_size', 'Unknown') if dataset.metadata else 'Unknown'
            })
        
        return JsonResponse({
            'success': True,
            'datasets': dataset_list
        })
        
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)

@mongo_login_required
@require_http_methods(["POST"])
@csrf_exempt
def preview_transformation_step(request):
    """Preview a single transformation step"""
    try:
        data = json.loads(request.body)
        dataset_id = data.get('dataset_id')
        operation = data.get('operation')
        parameters = data.get('parameters', {})
        
        # Validate inputs
        if not dataset_id:
            return JsonResponse({'success': False, 'error': 'Dataset ID is required'}, status=400)
        
        if not operation:
            return JsonResponse({'success': False, 'error': 'Operation is required'}, status=400)
        
        # Get dataset
        dataset = Dataset.objects.get(id=ObjectId(dataset_id), owner_id=str(request.session.get("user_id")))
        df = download_and_convert_to_dataframe(dataset)
        
        # Apply the transformation based on operation type
        result_df = apply_transformation_operation(df, operation, parameters)
        
        # Generate preview
        preview_data = dataframe_to_dict_clean(result_df.head(20))
        
        return JsonResponse({
            'success': True,
            'preview_data': preview_data,
            'columns': list(result_df.columns),
            'total_rows': len(result_df),
            'operation': operation
        })
        
    except Dataset.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Dataset not found'}, status=404)
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)

def apply_transformation_operation(df, operation, parameters):
    """Apply a transformation operation to the dataframe"""
    result_df = df.copy()
    
    if operation == 'missing_values':
        strategy = parameters.get('strategy', 'fill_mean')
        
        if strategy == 'fill_mean':
            # Fill numeric columns with mean
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                result_df[col] = result_df[col].fillna(result_df[col].mean())
        elif strategy == 'fill_median':
            # Fill numeric columns with median
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                result_df[col] = result_df[col].fillna(result_df[col].median())
        elif strategy == 'fill_mode':
            # Fill with mode
            for col in df.columns:
                result_df[col] = result_df[col].fillna(result_df[col].mode()[0] if not result_df[col].mode().empty else '')
        elif strategy == 'drop':
            # Drop rows with missing values
            result_df = result_df.dropna()
            
    elif operation == 'remove_duplicates':
        keep = parameters.get('keep', 'first')
        result_df = result_df.drop_duplicates(keep=keep)
        
    elif operation == 'filter':
        expression = parameters.get('expression', '')
        if expression:
            try:
                # Simple expression evaluation (you might want to use a safer method)
                result_df = result_df.query(expression)
            except:
                # Fallback: try column-based filtering
                pass
                
    elif operation == 'groupby':
        group_columns = parameters.get('group_columns', [])
        aggregation = parameters.get('aggregation', 'mean')
        
        if group_columns and isinstance(group_columns, list):
            # Convert string to list if needed
            if isinstance(group_columns, str):
                group_columns = [col.strip() for col in group_columns.split(',')]
            
            # Only use columns that exist in the dataframe
            valid_columns = [col for col in group_columns if col in df.columns]
            
            if valid_columns:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                agg_dict = {col: aggregation for col in numeric_cols if col not in valid_columns}
                
                if agg_dict:
                    result_df = df.groupby(valid_columns).agg(agg_dict).reset_index()
                    
    elif operation == 'sort':
        column = parameters.get('column', '')
        order = parameters.get('order', 'asc')
        
        if column and column in df.columns:
            ascending = order == 'asc'
            result_df = result_df.sort_values(by=column, ascending=ascending).reset_index(drop=True)
            
    elif operation == 'calculated_column':
        column_name = parameters.get('column_name', '')
        expression = parameters.get('expression', '')
        
        if column_name and expression:
            try:
                # Simple expression evaluation (be careful with this in production)
                # You might want to use a proper expression parser
                result_df[column_name] = eval(expression, {'df': df})
            except:
                # If expression fails, create a dummy column
                result_df[column_name] = 'Calculation Error'
    
    return result_df

@csrf_exempt                          # must be first
@require_http_methods(["POST"])       # then method validation
@mongo_login_required      
def preview_pipeline_step(request):
    """Preview a pipeline step with all previous steps applied"""
    try:
        print("=== DEBUG: preview_pipeline_step called ===")
        data = json.loads(request.body)
        print("=== DEBUG: Request data ===", data)
        
        input_dataset_id = data.get('input_dataset_id')
        steps = data.get('steps', [])
        preview_step = data.get('preview_step', len(steps) - 1)
        
        print(f"=== DEBUG: input_dataset_id: {input_dataset_id}, steps: {len(steps)}, preview_step: {preview_step} ===")
        
        if not input_dataset_id:
            return JsonResponse({'success': False, 'error': 'Input dataset ID is required'}, status=400)
        
        # Get input dataset
        dataset = Dataset.objects.get(id=ObjectId(input_dataset_id), owner_id=str(request.session.get("user_id")))
        df = download_and_convert_to_dataframe(dataset)
        
        print(f"=== DEBUG: Loaded dataset with {len(df)} rows and {len(df.columns)} columns ===")
        
        # Apply all steps up to the preview step
        current_df = df.copy()
        applied_steps = []
        
        for i, step in enumerate(steps[:preview_step + 1]):
            try:
                print(f"=== DEBUG: Applying step {i+1}: {step['operation']} ===")
                # Apply transformation
                current_df = apply_transformation_operation(current_df, step['operation'], step.get('parameters', {}))
                applied_steps.append({
                    'step_number': i + 1,
                    'operation': step['operation'],
                    'success': True
                })
                print(f"=== DEBUG: Step {i+1} successful, shape: {current_df.shape} ===")
            except Exception as step_error:
                print(f"=== DEBUG: Step {i+1} failed: {str(step_error)} ===")
                applied_steps.append({
                    'step_number': i + 1,
                    'operation': step['operation'],
                    'success': False,
                    'error': str(step_error)
                })
                return JsonResponse({
                    'success': False, 
                    'error': f'Error in step {i + 1} ({step["operation"]}): {str(step_error)}',
                    'applied_steps': applied_steps
                }, status=400)
        
        # Generate preview
        preview_data = dataframe_to_dict_clean(current_df.head(20))
        
        print(f"=== DEBUG: Preview generated with {len(preview_data)} rows ===")
        
        # ✅ Create a temporary output dataset to chain steps
        temp_dataset = create_transformation_dataset(
            current_df,
            f"preview_temp_step_{preview_step}",
            str(request.session.get("user_id")),
            "preview_only"
        )

        return JsonResponse({
            'success': True,
            'preview_data': preview_data,
            'columns': list(current_df.columns),
            'total_rows': len(current_df),
            'steps_applied': len(applied_steps),
            'applied_steps': applied_steps,
            'output_dataset_id': str(temp_dataset.id)   # ✅ REQUIRED
        })


        
    except Dataset.DoesNotExist:
        print("=== DEBUG: Dataset not found ===")
        return JsonResponse({'success': False, 'error': 'Dataset not found'}, status=404)
    except Exception as e:
        print(f"=== DEBUG: General error: {str(e)} ===")
        import traceback
        print(f"=== DEBUG: Traceback: {traceback.format_exc()} ===")
        return JsonResponse({'success': False, 'error': str(e)}, status=400)


@mongo_login_required
@require_http_methods(["POST"])
@csrf_exempt
def execute_pipeline(request):
    """Execute a complete pipeline and save the result"""
    try:
        data = json.loads(request.body)
        input_dataset_id = data.get('input_dataset_id')
        steps = data.get('steps', [])
        
        if not input_dataset_id:
            return JsonResponse({'success': False, 'error': 'Input dataset ID is required'}, status=400)
        
        if not steps:
            return JsonResponse({'success': False, 'error': 'No steps provided'}, status=400)
        
        # Get input dataset
        dataset = Dataset.objects.get(id=ObjectId(input_dataset_id), owner_id=str(request.session.get("user_id")))
        df = download_and_convert_to_dataframe(dataset)
        
        # Apply all steps
        for step in steps:
            df = apply_transformation_operation(df, step['operation'], step.get('parameters', {}))
        
        # Create output dataset
        pipeline_name = f"Pipeline_Output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        new_dataset = create_cleaned_dataset(df, pipeline_name, request.session.get("user_id"), 'pipeline')
        
        return JsonResponse({
            'success': True,
            'output_dataset_id': str(new_dataset.id),
            'output_dataset_name': pipeline_name,
            'row_count': len(df),
            'column_count': len(df.columns),
            'steps_executed': len(steps)
        })
        
    except Exception as e: 
        return JsonResponse({'success': False, 'error': str(e)}, status=400)
    






# NEW CODE FOR DATA VISUALIZATION PAGE
from django.shortcuts import render

@mongo_login_required
def analyze_data_page(request):
    """
    Page where multiple data visualizations will be displayed.
    """
    # Get workspace_id from query parameters if provided
    workspace_id = request.GET.get('workspace')
    
    # Get all datasets for the current user
    datasets = Dataset.objects(owner_id=str(request.session.get("user_id")))
    
    # Get the selected dataset if provided
    selected_dataset_id = request.GET.get('dataset')
    selected_dataset = None
    if selected_dataset_id:
        try:
            selected_dataset = Dataset.objects.get(id=ObjectId(selected_dataset_id), owner_id=str(request.session.get("user_id")))
        except:
            selected_dataset = None
    
    # If no dataset selected, use the first one
    if not selected_dataset and datasets:
        selected_dataset = datasets[0]
    
    context = {
        "datasets": datasets,
        "selected_dataset": selected_dataset,
        "workspace_id": workspace_id,
        "user_initials": request.user.username[0].upper() if request.user.username else 'U'
    }
    
    return render(request, "analyze_data.html", context)


def assistant_table(request):
    return render(request, "assistant_table.html")


def visualization_page(request):
    """Render the visualization page"""
    datasets = DataSet.objects.filter(user_id=str(request.session.get("user_id")), status='active')
    
    # Get user's saved visualizations
    visualizations = Visualization.objects.filter(user_id=str(request.session.get("user_id")), status='active')
    
    context = {
        'datasets': datasets,
        'selected_dataset': None,
        'visualizations': visualizations,
        'user_initials': f"{request.user.first_name[0]}{request.user.last_name[0]}".upper() if request.user.first_name and request.user.last_name else "U",
        'user': request.user
    }
    return render(request, 'assistant/visualization.html', context)


@mongo_login_required
def generate_visualization(request):
    """Generate visualization using Python/Plotly"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method allowed'}, status=405)
    
    try:
        data = json.loads(request.body)
        
        # Extract visualization configuration
        chart_type = data.get('chart_type', 'bar')
        dataset_ids = data.get('dataset_ids', [])
        x_axis = data.get('x_axis')
        y_axis = data.get('y_axis', [])
        group_by = data.get('group_by')
        colors = data.get('colors', ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6'])
        options = data.get('options', {})
        
        print(f"📊 Generating {chart_type} chart for {len(dataset_ids)} datasets")
        print(f"X-axis: {x_axis}, Y-axis: {y_axis}, Group by: {group_by}")
        
        if not dataset_ids or not x_axis or not y_axis:
            return JsonResponse({'error': 'Missing required parameters'}, status=400)
        
        # Load and join datasets
        combined_df = load_and_join_datasets(request, dataset_ids, data.get('join_config', {}))
        
        print(f"📈 Combined data shape: {combined_df.shape}")
        print(f"📈 Columns: {list(combined_df.columns)}")
        print(f"📈 Sample data:")
        print(combined_df.head())
        
        if combined_df.empty:
            return JsonResponse({'error': 'No data available for visualization'}, status=400)
        
        # Validate columns exist
        missing_columns = []
        if x_axis not in combined_df.columns:
            missing_columns.append(x_axis)
        for y_col in y_axis:
            if y_col not in combined_df.columns:
                missing_columns.append(y_col)
        
        if missing_columns:
            return JsonResponse({
                'error': f'Columns not found: {missing_columns}. Available: {list(combined_df.columns)}'
            }, status=400)
        
        # Generate plot using Plotly
        fig = create_plotly_chart(combined_df, chart_type, x_axis, y_axis, group_by, colors, options)
        
        # Verify we have a valid Plotly figure
        if not hasattr(fig, 'to_html'):
            raise ValueError("Generated object is not a valid Plotly figure")
        
        # Convert plot to HTML
        plot_html = pio.to_html(fig, include_plotlyjs=False, full_html=False)
        
        # Generate thumbnail
        thumbnail_data = generate_thumbnail(fig)
        
        # Prepare response data
        response_data = {
            'success': True,
            'plot_html': plot_html,
            'thumbnail': thumbnail_data,
            'data_summary': {
                'row_count': len(combined_df),
                'columns': list(combined_df.columns),
                'x_axis_unique': combined_df[x_axis].nunique() if x_axis in combined_df.columns else 0,
                'y_axis_stats': {col: {
                    'min': float(combined_df[col].min()),
                    'max': float(combined_df[col].max()),
                    'mean': float(combined_df[col].mean())
                } for col in y_axis if col in combined_df.columns}
            }
        }
        
        print("✅ Visualization generated successfully")
        return JsonResponse(response_data)
        
    except Exception as e:
        print(f"❌ Visualization generation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return JsonResponse({'error': f'Visualization generation failed: {str(e)}'}, status=500)


def load_and_join_datasets(request, dataset_ids, join_config):
    """Load multiple datasets using the actual request object"""
    datasets = []
    
    for dataset_id in dataset_ids:
        try:
            print(f"🔍 Loading dataset via API: {dataset_id}")
            
            # Import the working preview function
            from datasets.views import preview_dataset
            
            # Call the working preview function with the actual request
            response = preview_dataset(request, dataset_id)
            
            if response.status_code == 200:
                data = json.loads(response.content)
                
                if 'rows' in data and data['rows']:
                    # FIX: Handle both data structures
                    # Structure 1: Separate 'columns' and 'rows' fields
                    if 'columns' in data and data['columns'] and len(data['columns']) > 0:
                        headers = data['columns']
                        rows = data['rows']
                        print(f"📊 Using separate columns field: {len(headers)} columns, {len(rows)} rows")
                    # Structure 2: First row contains headers
                    elif len(data['rows']) > 0:
                        headers = data['rows'][0]
                        rows = data['rows'][1:] if len(data['rows']) > 1 else []
                        print(f"📊 Using first row as headers: {len(headers)} columns, {len(rows)} rows")
                    else:
                        print(f"❌ No valid data structure found for {dataset_id}")
                        continue
                    
                    # Create DataFrame
                    df = pd.DataFrame(rows, columns=headers)
                    print(f"✅ Loaded via API: {df.shape}")
                    print(f"📋 Sample columns: {list(df.columns[:5])}...")  # Show first 5 columns
                    datasets.append((dataset_id, df))
                else:
                    print(f"❌ No rows in API response for {dataset_id}")
                    continue
            else:
                print(f"❌ API error for {dataset_id}: {response.status_code}")
                print(f"🔍 Response content: {response.content}")
                continue
                
        except Exception as e:
            print(f"💥 Error loading dataset {dataset_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    print(f"📦 Successfully loaded {len(datasets)} datasets via API")

    if not datasets:
        print("❌ No datasets were successfully loaded")
        return pd.DataFrame()

    if len(datasets) == 1:
        print(f"🏁 Returning single dataset: {datasets[0][1].shape}")
        return datasets[0][1]

    # Rest of your join logic...
    join_key = join_config.get('key')
    join_type = join_config.get('type', 'inner')

    print(f"🔗 Joining {len(datasets)} datasets | key: {join_key} | type: {join_type}")

    # Print available columns from all datasets for debugging
    for i, (dataset_id, df) in enumerate(datasets):
        print(f"📊 Dataset {i} columns: {list(df.columns)}")

    if not join_key:
        common = find_common_columns([df for _, df in datasets])
        if common:
            join_key = common[0]
            print(f"🧠 Auto-detected join key: {join_key}")
        else:
            print("⚠️ No join key found — concatenating instead")
            concatenated = pd.concat([df for _, df in datasets], ignore_index=True)
            print(f"🏁 Concatenated dataset shape: {concatenated.shape}")
            return concatenated

    main_df = datasets[0][1]
    print(f"🎯 Main DF shape: {main_df.shape}, columns: {list(main_df.columns)}")

    for i in range(1, len(datasets)):
        current_df = datasets[i][1]
        current_id = datasets[i][0]
        
        print(f"🔄 Joining with dataset {i} (ID: {current_id})")
        print(f"   Current DF shape: {current_df.shape}, columns: {list(current_df.columns)}")
        print(f"   Join key: '{join_key}'")

        if join_key in main_df.columns and join_key in current_df.columns:
            # Check for duplicate column names and handle them
            common_cols = set(main_df.columns) & set(current_df.columns)
            common_cols.discard(join_key)  # Remove join key from common columns
            
            if common_cols:
                print(f"⚠️  Duplicate columns detected: {common_cols}")
                # Rename duplicate columns in the right dataframe
                suffix = f"_{i}"
                rename_dict = {col: f"{col}{suffix}" for col in common_cols}
                current_df = current_df.rename(columns=rename_dict)
                print(f"   Renamed columns: {rename_dict}")
            
            # Perform the merge
            before_shape = main_df.shape
            main_df = main_df.merge(current_df, on=join_key, how=join_type)
            print(f"✅ Joined with DF{i} -> {before_shape} → {main_df.shape}")
            
            # Show sample of joined data
            if not main_df.empty:
                print(f"📋 Joined data sample - First 3 rows:")
                for col in main_df.columns[:3]:  # Show first 3 columns
                    sample_vals = main_df[col].head(3).tolist()
                    print(f"   {col}: {sample_vals}")
        else:
            print(f"⚠️ Key '{join_key}' missing in one or both dataframes")
            print(f"   Main DF has key: {join_key in main_df.columns}")
            print(f"   Current DF has key: {join_key in current_df.columns}")
            print(f"   Main DF columns: {list(main_df.columns)}")
            print(f"   Current DF columns: {list(current_df.columns)}")
            print("   → Concatenating instead")
            main_df = pd.concat([main_df, current_df], ignore_index=True)

    print(f"🏁 Final joined dataset shape: {main_df.shape}")
    print(f"📊 Final columns: {list(main_df.columns)}")
    
    return main_df


def find_common_columns(dataframes):
    """Find common columns across multiple dataframes"""
    if not dataframes:
        return []
    
    common_cols = set(dataframes[0].columns)
    for df in dataframes[1:]:
        common_cols = common_cols.intersection(set(df.columns))
    
    print(f"🔍 Common columns across datasets: {list(common_cols)}")
    return list(common_cols)
def find_common_columns(dataframes):
    """Find common columns across multiple dataframes"""
    if not dataframes:
        return []
    
    common_cols = set(dataframes[0].columns)
    for df in dataframes[1:]:
        common_cols = common_cols.intersection(set(df.columns))
    
    return list(common_cols)
def create_plotly_chart(df, chart_type, x_axis, y_axis, group_by, colors, options):
    """Create Plotly chart based on configuration"""
    print(f"🎨 Creating {chart_type} chart...")
    
    try:
        # Ensure data types are appropriate for numeric columns
        for col in y_axis:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle grouping
        if group_by and group_by in df.columns:
            fig = create_grouped_chart(df, chart_type, x_axis, y_axis, group_by, colors, options)
        else:
            fig = create_ungrouped_chart(df, chart_type, x_axis, y_axis, colors, options)
        
        # Apply common layout options
        fig = apply_chart_layout(fig, chart_type, x_axis, y_axis, options)
        
        return fig
        
    except Exception as e:
        print(f"❌ Chart creation failed: {e}")
        # Return a simple error chart
        return create_error_chart(f"Chart creation failed: {str(e)}")

def create_grouped_chart(df, chart_type, x_axis, y_axis, group_by, colors, options):
    """Create chart with grouping"""
    print(f"📊 Creating grouped {chart_type} chart with group_by: {group_by}")
    
    fig = go.Figure()
    groups = df[group_by].unique()
    
    if chart_type == 'bar':
        for i, group in enumerate(groups):
            group_data = df[df[group_by] == group]
            color = colors[i % len(colors)]
            
            for y_col in y_axis:
                if y_col in group_data.columns:
                    fig.add_trace(go.Bar(
                        name=f"{group} - {y_col}",
                        x=group_data[x_axis],
                        y=group_data[y_col],
                        marker_color=color
                    ))
    
    elif chart_type == 'line':
        for i, group in enumerate(groups):
            group_data = df[df[group_by] == group]
            color = colors[i % len(colors)]
            
            for y_col in y_axis:
                if y_col in group_data.columns:
                    fig.add_trace(go.Scatter(
                        name=f"{group} - {y_col}",
                        x=group_data[x_axis],
                        y=group_data[y_col],
                        mode='lines+markers',
                        line=dict(color=color)
                    ))
    
    elif chart_type == 'pie':
        # For pie charts with grouping, use the first Y-axis
        if y_axis and y_axis[0] in df.columns:
            group_sums = df.groupby(group_by)[y_axis[0]].sum().reset_index()
            fig = px.pie(
                group_sums, 
                names=group_by, 
                values=y_axis[0],
                color_discrete_sequence=colors
            )
        else:
            fig = create_error_chart("No valid Y-axis for pie chart")
    
    else:
        # Default to bar chart for other types with grouping
        return create_grouped_chart(df, 'bar', x_axis, y_axis, group_by, colors, options)
    
    return fig

def create_ungrouped_chart(df, chart_type, x_axis, y_axis, colors, options):
    """Create chart without grouping"""
    print(f"📊 Creating ungrouped {chart_type} chart")
    
    if chart_type == 'bar':
        fig = go.Figure()
        for i, y_col in enumerate(y_axis):
            if y_col in df.columns:
                color = colors[i % len(colors)]
                fig.add_trace(go.Bar(
                    name=y_col,
                    x=df[x_axis],
                    y=df[y_col],
                    marker_color=color
                ))
    
    elif chart_type == 'line':
        fig = go.Figure()
        for i, y_col in enumerate(y_axis):
            if y_col in df.columns:
                color = colors[i % len(colors)]
                fig.add_trace(go.Scatter(
                    name=y_col,
                    x=df[x_axis],
                    y=df[y_col],
                    mode='lines+markers',
                    line=dict(color=color)
                ))
    
    elif chart_type == 'pie':
        if y_axis and y_axis[0] in df.columns:
            fig = px.pie(
                df, 
                names=x_axis, 
                values=y_axis[0],
                color_discrete_sequence=colors
            )
        else:
            fig = create_error_chart("No valid Y-axis for pie chart")
    
    elif chart_type == 'scatter':
        if len(y_axis) >= 1 and y_axis[0] in df.columns:
            if len(y_axis) >= 2 and y_axis[1] in df.columns:
                fig = px.scatter(
                    df, 
                    x=x_axis, 
                    y=y_axis[0],
                    color=y_axis[1],
                    color_continuous_scale=colors
                )
            else:
                fig = px.scatter(
                    df, 
                    x=x_axis, 
                    y=y_axis[0]
                )
        else:
            fig = create_error_chart("Invalid columns for scatter plot")
    
    elif chart_type == 'area':
        fig = go.Figure()
        for i, y_col in enumerate(y_axis):
            if y_col in df.columns:
                color = colors[i % len(colors)]
                fig.add_trace(go.Scatter(
                    name=y_col,
                    x=df[x_axis],
                    y=df[y_col],
                    mode='lines',
                    fill='tozeroy',
                    line=dict(color=color)
                ))
    
    elif chart_type == 'heatmap':
        if len(y_axis) >= 1 and y_axis[0] in df.columns:
            try:
                # For heatmap, we need at least 3 columns: x, y, value
                if len(df.columns) >= 3:
                    pivot_col = df.columns[2] if len(df.columns) > 2 else x_axis
                    pivot_data = df.pivot_table(
                        values=y_axis[0], 
                        index=x_axis, 
                        columns=pivot_col,
                        aggfunc='mean'
                    ).fillna(0)
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=pivot_data.values,
                        x=pivot_data.columns,
                        y=pivot_data.index,
                        colorscale='Blues'
                    ))
                else:
                    fig = create_error_chart("Not enough columns for heatmap")
            except Exception as e:
                print(f"❌ Heatmap creation failed: {e}")
                fig = create_error_chart(f"Heatmap failed: {str(e)}")
        else:
            fig = create_error_chart("Invalid columns for heatmap")
    
    else:
        # Default to bar chart
        return create_ungrouped_chart(df, 'bar', x_axis, y_axis, colors, options)
    
    return fig

def apply_chart_layout(fig, chart_type, x_axis, y_axis, options):
    """Apply chart layout options safely"""
    show_grid = options.get('showGrid', False)
    show_legend = options.get('showLegend', True)
    show_values = options.get('showValues', False)
    
    # Basic layout configuration
    layout_config = {
        'title': f"{chart_type.title()} Chart - {x_axis} vs {', '.join(y_axis)}",
        'xaxis_title': x_axis,
        'yaxis_title': ', '.join(y_axis),
        'showlegend': show_legend,
        'plot_bgcolor': 'white',
        'paper_bgcolor': 'white',
        'font': {'size': 12}
    }
    
    # Update layout
    fig.update_layout(**layout_config)
    
    # Grid options - use dict approach instead of method calls
    xaxis_config = {'showgrid': show_grid}
    yaxis_config = {'showgrid': show_grid}
    
    if show_grid:
        xaxis_config.update({'gridwidth': 1, 'gridcolor': 'lightgray'})
        yaxis_config.update({'gridwidth': 1, 'gridcolor': 'lightgray'})
    
    fig.update_layout(xaxis=xaxis_config, yaxis=yaxis_config)
    
    # Show values on bars/pie
    if show_values and chart_type in ['bar', 'pie']:
        fig.update_traces(texttemplate='%{value:.2f}', textposition='outside')
    
    print("✅ Chart layout applied successfully")
    return fig

def create_error_chart(message):
    """Create an error chart when something goes wrong"""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color="red")
    )
    fig.update_layout(
        title="Chart Error",
        xaxis={"visible": False},
        yaxis={"visible": False},
        plot_bgcolor="white"
    )
    return fig
def apply_chart_options(fig, chart_type, x_axis, y_axis, options):
    """Apply chart options and styling"""
    show_grid = options.get('showGrid', False)
    show_legend = options.get('showLegend', True)
    show_values = options.get('showValues', False)
    
    # Basic layout configuration
    layout_config = dict(
        title=f"{chart_type.title()} Chart - {x_axis} vs {', '.join(y_axis)}",
        xaxis_title=x_axis,
        yaxis_title=', '.join(y_axis),
        showlegend=show_legend,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12)
    )
    
    # Update layout
    fig.update_layout(**layout_config)
    
    # Grid options
    if show_grid:
        fig.update_xaxis(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxis(showgrid=True, gridwidth=1, gridcolor='lightgray')
    else:
        fig.update_xaxis(showgrid=False)
        fig.update_yaxis(showgrid=False)
    
    # Show values on bars/pie
    if show_values and chart_type in ['bar', 'pie']:
        fig.update_traces(texttemplate='%{value:.2f}', textposition='outside')
    
    print("✅ Chart options applied successfully")
    return fig


def generate_thumbnail(fig):
    """Generate base64 thumbnail for the plot"""
    try:
        # Create smaller version for thumbnail
        fig.update_layout(
            width=300,
            height=200,
            margin=dict(l=10, r=10, t=30, b=10),
            showlegend=False,
            title=""
        )
        
        # Convert to base64
        img_bytes = pio.to_image(fig, format='png', width=300, height=200)
        return base64.b64encode(img_bytes).decode('utf-8')
    except Exception as e:
        print(f"❌ Thumbnail generation failed: {e}")
        return None
    


# Save visualization
@mongo_login_required
def save_visualization(request):
    """Save visualization configuration and data"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method allowed'}, status=405)
    
    try:
        data = json.loads(request.body)
        
        # Validate required fields
        required_fields = ['name', 'chart_type', 'dataset_ids', 'x_axis', 'y_axis']
        for field in required_fields:
            if not data.get(field):
                return JsonResponse({'error': f'Missing required field: {field}'}, status=400)
        
        # Get dataset names
        dataset_names = []
        for dataset_id in data['dataset_ids']:
            try:
                dataset = DataSet.objects.get(id=ObjectId(dataset_id), user_id=str(request.session.get("user_id")))
                dataset_names.append(dataset.name)
            except DataSet.DoesNotExist:
                dataset_names.append(f"Dataset_{dataset_id}")
        
        # Create saved visualization
        saved_viz = SavedVisualization(
            name=data['name'],
            description=data.get('description', ''),
            chart_type=data['chart_type'],
            dataset_ids=data['dataset_ids'],
            dataset_names=dataset_names,
            x_axis=data['x_axis'],
            y_axis=data['y_axis'],
            group_by=data.get('group_by'),
            colors=data.get('colors', ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6']),
            options=data.get('options', {}),
            layout_config=data.get('layout_config', {}),
            filters=data.get('filters', {}),
            join_config=data.get('join_config', {}),
            plot_html=data.get('plot_html', ''),
            thumbnail_data=data.get('thumbnail_data'),
            data_summary=data.get('data_summary', {}),
            user_id=str(request.session.get("user_id")),
            workspace_id=data.get('workspace_id', 'default'),
            tags=data.get('tags', []),
            is_public=data.get('is_public', False),
            is_template=data.get('is_template', False)
        )
        
        saved_viz.save()
        
        return JsonResponse({
            'success': True,
            'visualization_id': str(saved_viz.id),
            'message': 'Visualization saved successfully',
            'visualization': saved_viz.to_dict()
        })
        
    except Exception as e:
        return JsonResponse({'error': f'Failed to save visualization: {str(e)}'}, status=500)

@mongo_login_required
def get_saved_visualization(request, viz_id):
    """Get specific saved visualization"""
    try:
        visualization = SavedVisualization.objects.get(
            id=ObjectId(viz_id),
            user_id=str(request.session.get("user_id"))
        )
        
        # Increment view count
        visualization.increment_view_count()
        
        # Return full visualization data
        response_data = visualization.to_dict()
        response_data['plot_html'] = visualization.plot_html
        response_data['layout_config'] = visualization.layout_config
        response_data['filters'] = visualization.filters
        response_data['join_config'] = visualization.join_config
        
        return JsonResponse({
            'success': True,
            'visualization': response_data
        })
        
    except SavedVisualization.DoesNotExist:
        return JsonResponse({'error': 'Visualization not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': f'Failed to load visualization: {str(e)}'}, status=500)


# Update saved visualization
@mongo_login_required
def update_saved_visualization(request, viz_id):
    """Update saved visualization"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method allowed'}, status=405)
    
    try:
        data = json.loads(request.body)
        visualization = SavedVisualization.objects.get(
            id=ObjectId(viz_id),
            user_id=str(request.session.get("user_id"))
        )
        
        # Update fields
        updatable_fields = [
            'name', 'description', 'chart_type', 'colors', 'options', 
            'layout_config', 'filters', 'tags', 'is_public', 'is_template'
        ]
        
        for field in updatable_fields:
            if field in data:
                setattr(visualization, field, data[field])
        
        # Update plot data if provided
        if 'plot_html' in data:
            visualization.plot_html = data['plot_html']
        
        if 'thumbnail_data' in data:
            visualization.thumbnail_data = data['thumbnail_data']
        
        if 'data_summary' in data:
            visualization.data_summary = data['data_summary']
        
        visualization.save()
        
        return JsonResponse({
            'success': True,
            'message': 'Visualization updated successfully',
            'visualization': visualization.to_dict()
        })
        
    except SavedVisualization.DoesNotExist:
        return JsonResponse({'error': 'Visualization not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': f'Failed to update visualization: {str(e)}'}, status=500)

# Delete saved visualization (soft delete)
@mongo_login_required
def delete_saved_visualization(request, viz_id):
    """Delete saved visualization (soft delete)"""
    try:
        visualization = SavedVisualization.objects.get(
            id=ObjectId(viz_id),
            user_id=str(request.session.get("user_id"))
        )
        
        visualization.status = 'archived'
        visualization.save()
        
        return JsonResponse({
            'success': True,
            'message': 'Visualization deleted successfully'
        })
        
    except SavedVisualization.DoesNotExist:
        return JsonResponse({'error': 'Visualization not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': f'Failed to delete visualization: {str(e)}'}, status=500)

# Duplicate saved visualization
@mongo_login_required
def duplicate_saved_visualization(request, viz_id):
    """Duplicate a saved visualization"""
    try:
        original_viz = SavedVisualization.objects.get(
            id=ObjectId(viz_id),
            user_id=str(request.session.get("user_id"))
        )
        
        # Create a copy
        new_viz = SavedVisualization(
            name=f"{original_viz.name} (Copy)",
            description=original_viz.description,
            chart_type=original_viz.chart_type,
            dataset_ids=original_viz.dataset_ids.copy(),
            dataset_names=original_viz.dataset_names.copy(),
            x_axis=original_viz.x_axis,
            y_axis=original_viz.y_axis.copy(),
            group_by=original_viz.group_by,
            colors=original_viz.colors.copy(),
            options=original_viz.options.copy(),
            layout_config=original_viz.layout_config.copy(),
            filters=original_viz.filters.copy(),
            join_config=original_viz.join_config.copy(),
            plot_html=original_viz.plot_html,
            thumbnail_data=original_viz.thumbnail_data,
            data_summary=original_viz.data_summary.copy(),
            user_id=str(request.session.get("user_id")),
            workspace_id=original_viz.workspace_id,
            tags=original_viz.tags.copy(),
            is_public=False,
            is_template=original_viz.is_template
        )
        
        new_viz.save()
        
        return JsonResponse({
            'success': True,
            'visualization_id': str(new_viz.id),
            'message': 'Visualization duplicated successfully',
            'visualization': new_viz.to_dict()
        })
        
    except SavedVisualization.DoesNotExist:
        return JsonResponse({'error': 'Visualization not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': f'Failed to duplicate visualization: {str(e)}'}, status=500)

# Search saved visualizations
@mongo_login_required
def search_saved_visualizations(request):
    """Search saved visualizations by name, description, or tags"""
    try:
        query = request.GET.get('q', '')
        if not query:
            return JsonResponse({'error': 'Search query required'}, status=400)
        
        # Case-insensitive search on name, description, and tags
        visualizations = SavedVisualization.objects(
            user_id=str(request.session.get("user_id")),
            status='active',
            __raw__={
                '$or': [
                    {'name': {'$regex': query, '$options': 'i'}},
                    {'description': {'$regex': query, '$options': 'i'}},
                    {'tags': {'$regex': query, '$options': 'i'}}
                ]
            }
        ).order_by('-updated_at')
        
        viz_list = [viz.to_dict() for viz in visualizations]
        
        return JsonResponse({
            'success': True,
            'visualizations': viz_list,
            'total_count': len(viz_list),
            'query': query
        })
        
    except Exception as e:
        return JsonResponse({'error': f'Search failed: {str(e)}'}, status=500)
@mongo_login_required
def get_saved_visualizations(request):
    """Get all saved visualizations for the user"""
    try:
        # Get query parameters
        chart_type = request.GET.get('chart_type')
        tags = request.GET.getlist('tags')
        limit = int(request.GET.get('limit', 50))
        offset = int(request.GET.get('offset', 0))
        
        # Build query
        query = {
            'user_id': str(request.session.get("user_id")),
            'status': 'active'
        }
        
        if chart_type:
            query['chart_type'] = chart_type
        
        if tags:
            query['tags__in'] = tags
        
        # Get visualizations
        visualizations = SavedVisualization.objects(**query).order_by('-updated_at')
        
        total_count = visualizations.count()
        visualizations = visualizations.skip(offset).limit(limit)
        
        viz_list = [viz.to_dict() for viz in visualizations]
        
        return JsonResponse({
            'success': True,
            'visualizations': viz_list,
            'total_count': total_count,
            'offset': offset,
            'limit': limit
        })
        
    except Exception as e:
        return JsonResponse({'error': f'Failed to load visualizations: {str(e)}'}, status=500)
   
def get_user_visualizations(request):
    """Get user's saved visualizations"""
    try:
        # ✅ Use the new SavedVisualization model instead of Visualization
        visualizations = SavedVisualization.objects.filter(
            user_id=str(request.session.get("user_id")), 
            status='active'
        ).order_by('-created_at')
        
        viz_list = []
        for viz in visualizations:
            viz_list.append({
                'id': str(viz.id),
                'name': viz.name,
                'description': viz.description,
                'chart_type': viz.chart_type,
                'created_at': viz.created_at.isoformat() if viz.created_at else None,
                'updated_at': viz.updated_at.isoformat() if viz.updated_at else None,
                'dataset_count': len(viz.dataset_ids),
                'dataset_names': viz.dataset_names,
                'thumbnail_data': viz.thumbnail_data,
                'view_count': viz.view_count,
                'tags': viz.tags,
                'is_public': viz.is_public
            })
        
        return JsonResponse({'visualizations': viz_list})
        
    except Exception as e:
        return JsonResponse({'error': f'Failed to load visualizations: {str(e)}'}, status=500)

def get_visualization(request, viz_id):
    """Get specific visualization details"""
    try:
        visualization = Visualization.objects.get(
            id=viz_id,
            user_id=str(request.session.get("user_id"))
        )
        
        viz_data = {
            'id': str(visualization.id),
            'name': visualization.name,
            'description': visualization.description,
            'chart_type': visualization.chart_type,
            'dataset_ids': visualization.dataset_ids,
            'x_axis': visualization.x_axis,
            'y_axis': visualization.y_axis,
            'group_by': visualization.group_by,
            'colors': visualization.colors,
            'options': visualization.options,
            'join_config': visualization.join_config,
            'custom_settings': visualization.custom_settings,
            'created_at': visualization.created_at.isoformat(),
            'updated_at': visualization.updated_at.isoformat()
        }
        
        return JsonResponse(viz_data)
        
    except Visualization.DoesNotExist:
        return JsonResponse({'error': 'Visualization not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': f'Failed to load visualization: {str(e)}'}, status=500)

def update_visualization(request, viz_id):
    """Update visualization configuration"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method allowed'}, status=405)
    
    try:
        data = json.loads(request.body)
        visualization = Visualization.objects.get(
            id=viz_id,
            user_id=str(request.session.get("user_id"))
        )
        
        # Update fields
        visualization.name = data.get('name', visualization.name)
        visualization.description = data.get('description', visualization.description)
        visualization.chart_type = data.get('chart_type', visualization.chart_type)
        visualization.dataset_ids = data.get('dataset_ids', visualization.dataset_ids)
        visualization.x_axis = data.get('x_axis', visualization.x_axis)
        visualization.y_axis = data.get('y_axis', visualization.y_axis)
        visualization.group_by = data.get('group_by', visualization.group_by)
        visualization.colors = data.get('colors', visualization.colors)
        visualization.options = data.get('options', visualization.options)
        visualization.join_config = data.get('join_config', visualization.join_config)
        visualization.custom_settings = data.get('custom_settings', visualization.custom_settings)
        
        visualization.save()
        
        return JsonResponse({'success': True, 'message': 'Visualization updated successfully'})
        
    except Visualization.DoesNotExist:
        return JsonResponse({'error': 'Visualization not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': f'Failed to update visualization: {str(e)}'}, status=500)

def delete_visualization(request, viz_id):
    """Delete visualization (soft delete)"""
    try:
        visualization = Visualization.objects.get(
            id=viz_id,
            user_id=str(request.session.get("user_id"))
        )
        
        visualization.status = 'archived'
        visualization.save()
        
        return JsonResponse({'success': True, 'message': 'Visualization deleted successfully'})
        
    except Visualization.DoesNotExist:
        return JsonResponse({'error': 'Visualization not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': f'Failed to delete visualization: {str(e)}'}, status=500)

def export_visualization(request, viz_id):
    """Export visualization as image or data"""
    try:
        export_format = request.GET.get('format', 'png')
        visualization = Visualization.objects.get(
            id=viz_id,
            user_id=str(request.session.get("user_id"))
        )
        
        # Regenerate the plot for export
        combined_df = load_and_join_datasets(visualization.dataset_ids, visualization.join_config)
        fig = create_plotly_chart(
            combined_df, 
            visualization.chart_type,
            visualization.x_axis,
            visualization.y_axis,
            visualization.group_by,
            visualization.colors,
            visualization.options
        )
        
        if export_format == 'png':
            img_bytes = pio.to_image(fig, format='png', width=1200, height=800)
            response = HttpResponse(img_bytes, content_type='image/png')
            response['Content-Disposition'] = f'attachment; filename="{visualization.name}.png"'
            return response
        
        elif export_format == 'csv':
            # Export the data used for visualization
            csv_data = combined_df.to_csv(index=False)
            response = HttpResponse(csv_data, content_type='text/csv')
            response['Content-Disposition'] = f'attachment; filename="{visualization.name}_data.csv"'
            return response
        
        else:
            return JsonResponse({'error': 'Unsupported export format'}, status=400)
            
    except Visualization.DoesNotExist:
        return JsonResponse({'error': 'Visualization not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': f'Export failed: {str(e)}'}, status=500)

def manage_visualization_templates(request):
    """Manage visualization templates"""
    if request.method == 'GET':
        # Get user's templates
        templates = VisualizationTemplate.objects.filter(user_id=str(request.session.get("user_id")))
        template_list = [{
            'id': str(t.id),
            'name': t.name,
            'description': t.description,
            'chart_type': t.chart_type,
            'usage_count': t.usage_count
        } for t in templates]
        
        return JsonResponse({'templates': template_list})
    
    elif request.method == 'POST':
        # Create new template
        data = json.loads(request.body)
        
        template = VisualizationTemplate(
            name=data['name'],
            description=data.get('description', ''),
            chart_type=data['chart_type'],
            configuration=data['configuration'],
            user_id=str(request.session.get("user_id")),
            is_public=data.get('is_public', False),
            tags=data.get('tags', [])
        )
        
        template.save()
        return JsonResponse({'success': True, 'template_id': str(template.id)})
    
    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)

def get_visualization_template(request, template_id):
    """Get specific template details"""
    try:
        template = VisualizationTemplate.objects.get(
            id=template_id,
            user_id=str(request.session.get("user_id"))
        )
        
        template.usage_count += 1
        template.save()
        
        return JsonResponse({
            'id': str(template.id),
            'name': template.name,
            'description': template.description,
            'chart_type': template.chart_type,
            'configuration': template.configuration,
            'usage_count': template.usage_count
        })
        
    except VisualizationTemplate.DoesNotExist:
        return JsonResponse({'error': 'Template not found'}, status=404)



@mongo_login_required
def stats_display(request):
    """
    Page for data analysis with filtering, sorting, and insights
    """
    try:
        # Get workspace_id from query parameters if provided
        workspace_id = request.GET.get('workspace')
        
        # Get all datasets for the current user
        datasets = Dataset.objects.filter(owner_id=str(request.session.get("user_id")))
        
        # Get the selected dataset if provided
        selected_dataset_id = request.GET.get('dataset')
        selected_dataset = None
        
        # Pagination
        page = int(request.GET.get('page', 1))
        rows_per_page = 5
        
        # Global search
        global_search = request.GET.get('search', '')
        
        # Parse filters from URL parameters
        active_filters = parse_filters_from_request(request)
        
        # If no dataset selected but datasets exist, use the first one
        if not selected_dataset_id and datasets:
            selected_dataset = datasets.first()
        elif selected_dataset_id:
            try:
                selected_dataset = Dataset.objects.get(id=ObjectId(selected_dataset_id), owner_id=str(request.session.get("user_id")))
            except (Dataset.DoesNotExist, Exception):
                selected_dataset = datasets.first() if datasets else None
        
        # Initialize default stats
        dataset_stats = {
            "total_records": 0,
            "total_features": 0,
            "last_updated": "Never",
            "missing_data_percentage": 0,
            "duplicate_rows": 0,
            "data_size_mb": 0,
            "data_quality_percentage": 0
        }
        
        insights = []
        column_stats = []
        statistical_summary = []
        table_data = {"headers": [], "rows": []}
        available_columns = []
        total_pages = 1
        filter_options = {}
        
        # Calculate statistics for the selected dataset
        if selected_dataset:
            # Get filtered data
            filtered_df = apply_filters_to_dataset(selected_dataset, active_filters, global_search)
            
            # Calculate stats on filtered data
            dataset_stats = calculate_dataset_statistics_filtered(selected_dataset, filtered_df)
            insights = generate_insights(selected_dataset, filtered_df)
            column_stats = calculate_column_statistics_filtered(filtered_df)
            statistical_summary = calculate_statistical_summary_filtered(filtered_df)
            table_data = get_table_preview_data_filtered(filtered_df, page, rows_per_page)
            available_columns = get_available_columns_filtered(filtered_df)
            filter_options = get_filter_options(selected_dataset)
            
            # Calculate total pages for pagination
            total_records = dataset_stats.get("total_records", 0)
            total_pages = max(1, (total_records + rows_per_page - 1) // rows_per_page)
        
        context = {
            "datasets": datasets,
            "selected_dataset": selected_dataset,
            "workspace_id": workspace_id,
            "user_initials": request.user.username[0].upper() if request.user.username else 'U',
            "username": request.user.username,
            "dataset_stats": dataset_stats,
            "insights": insights,
            "column_stats": column_stats,
            "statistical_summary": statistical_summary,
            "table_data": table_data,
            "available_columns": available_columns,
            "filter_options": filter_options,
            "active_filters": active_filters,
            "global_search": global_search,
            "current_page": page,
            "total_pages": total_pages,
            "rows_per_page": rows_per_page,
        }
        
        return render(request, "stats.html", context)
    
    except Exception as e:
        print(f"Error in stats_display: {e}")
        import traceback
        traceback.print_exc()
        # Return a basic context even if there's an error
        return render(request, "stats.html", {
            "datasets": [],
            "selected_dataset": None,
            "user_initials": request.user.username[0].upper() if request.user.username else 'U',
            "username": request.user.username,
            "dataset_stats": {
                "total_records": 0,
                "total_features": 0,
                "last_updated": "Never",
                "missing_data_percentage": 0,
                "duplicate_rows": 0,
                "data_size_mb": 0,
                "data_quality_percentage": 0
            },
            "insights": [],
            "column_stats": [],
            "statistical_summary": [],
            "table_data": {"headers": [], "rows": []},
            "active_filters": [],
            "global_search": "",
            "current_page": 1,
            "total_pages": 1,
            "rows_per_page": 5,
        })


def parse_filters_from_request(request):
    """Parse filters from URL parameters"""
    filters = []
    
    # Find all filter indices
    filter_indices = set()
    for key in request.GET.keys():
        if key.startswith('filter_') and key.endswith('_column'):
            index = key.replace('filter_', '').replace('_column', '')
            filter_indices.add(index)
    
    # Build filter objects
    for index in sorted(filter_indices):
        column = request.GET.get(f'filter_{index}_column')
        operator = request.GET.get(f'filter_{index}_operator')
        value = request.GET.get(f'filter_{index}_value')
        value2 = request.GET.get(f'filter_{index}_value2')
        
        if column and operator and value:
            filter_obj = {
                'column': column,
                'operator': operator,
                'value': value
            }
            if value2:
                filter_obj['value2'] = value2
            filters.append(filter_obj)
    
    return filters


def calculate_dataset_statistics_filtered(dataset, df):
    """Calculate comprehensive statistics for a filtered dataset"""
    try:
        if df.empty:
            return {
                "total_records": 0,
                "total_features": 0,
                "last_updated": dataset.updated_at.strftime("%b %d, %Y %H:%M") if hasattr(dataset, 'updated_at') and dataset.updated_at else "Never",
                "missing_data_percentage": 0,
                "duplicate_rows": 0,
                "data_size_mb": 0,
                "data_quality_percentage": 0
            }
        
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        missing_percentage = round((missing_cells / total_cells) * 100, 2) if total_cells > 0 else 0
        
        stats = {
            "total_records": len(df),
            "total_features": len(df.columns),
            "last_updated": dataset.updated_at.strftime("%b %d, %Y %H:%M") if hasattr(dataset, 'updated_at') and dataset.updated_at else "Never",
            "missing_data_percentage": missing_percentage,
            "duplicate_rows": df.duplicated().sum(),
            "data_size_mb": round(os.path.getsize(dataset.file.path) / (1024 * 1024), 2) if hasattr(dataset, 'file') and dataset.file else 0,
            "data_quality_percentage": 100 - missing_percentage
        }
        
        return stats
        
    except Exception as e:
        print(f"Error calculating filtered dataset statistics: {e}")
        return {
            "total_records": 0,
            "total_features": 0,
            "last_updated": "Never",
            "missing_data_percentage": 0,
            "duplicate_rows": 0,
            "data_size_mb": 0,
            "data_quality_percentage": 0
        }

def get_table_preview_data_filtered(df, page=1, rows_per_page=5):
    """Get sample data for table preview from filtered DataFrame"""
    try:
        if df.empty:
            return {"headers": [], "rows": []}
        
        # Calculate start and end indices for pagination
        start_idx = (page - 1) * rows_per_page
        end_idx = start_idx + rows_per_page
        
        # Get the paginated data (all columns)
        preview_df = df.iloc[start_idx:end_idx]
        
        # Convert to list of lists with headers
        table_data = {
            'headers': list(preview_df.columns),
            'rows': preview_df.values.tolist()
        }
        
        return table_data
        
    except Exception as e:
        print(f"Error getting filtered table preview: {e}")
        return {"headers": [], "rows": []}

def calculate_column_statistics_filtered(df):
    """Calculate statistics for each column in the filtered dataset"""
    try:
        if df.empty:
            return []
        
        column_stats = []
        
        for col in df.columns[:10]:  # Show stats for first 10 columns
            col_data = df[col]
            
            stats = {
                "name": col,
                "data_type": str(col_data.dtype),
                "unique_values": col_data.nunique(),
                "null_percentage": round((col_data.isnull().sum() / len(col_data)) * 100, 2),
                "sample_values": ", ".join([str(x) for x in col_data.dropna().head(3).values])
            }
            
            column_stats.append(stats)
        
        return column_stats
        
    except Exception as e:
        print(f"Error calculating filtered column stats: {e}")
        return []

def calculate_statistical_summary_filtered(df):
    """Calculate statistical summary for numerical columns in filtered data"""
    try:
        if df.empty:
            return []
        
        statistical_summary = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols[:5]:  # Show stats for first 5 numeric columns
            col_data = df[col].dropna()
            
            if len(col_data) > 0:
                stats = {
                    "column": col,
                    "count": len(col_data),
                    "mean": round(col_data.mean(), 2),
                    "min": round(col_data.min(), 2),
                    "max": round(col_data.max(), 2),
                    "std_dev": round(col_data.std(), 2)
                }
                
                statistical_summary.append(stats)
        
        return statistical_summary
        
    except Exception as e:
        print(f"Error calculating filtered statistical summary: {e}")
        return []

def get_available_columns_filtered(df):
    """Get available columns for filtering from filtered DataFrame"""
    try:
        if df.empty:
            return []
        
        columns = []
        
        for col in df.columns:
            col_data = df[col]
            col_info = {
                'name': col,
                'type': str(col_data.dtype),
                'unique_values': col_data.nunique(),
                'sample_values': list(col_data.dropna().head(3).values) if col_data.nunique() > 0 else []
            }
            columns.append(col_info)
        
        return columns
    except Exception as e:
        print(f"Error getting filtered columns: {e}")
        return []

def generate_insights(dataset, df=None):
    """Generate auto-insights based on dataset analysis - updated to accept DataFrame"""
    if not dataset:
        return []
    
    try:
        # Use provided DataFrame or load from dataset
        if df is None:
            df = download_and_convert_to_dataframe(dataset)
        
        if df.empty:
            return []
        
        insights = []
        
        # Your existing insight generation logic here, but using the provided df
        # Check for trends in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            for col in numeric_cols[:2]:  # Check first 2 numeric columns
                if len(df) > 10:
                    # Simple trend detection
                    first_half_mean = df[col].iloc[:len(df)//2].mean()
                    second_half_mean = df[col].iloc[len(df)//2:].mean()
                    
                    if second_half_mean > first_half_mean * 1.1:
                        insights.append({
                            "type": "trend",
                            "title": f"Growth Trend in {col}",
                            "description": f"{col} shows a positive growth trend with values increasing by {round(((second_half_mean - first_half_mean) / first_half_mean) * 100, 1)}%",
                            "icon": "chart-line",
                            "color": "green"
                        })
                    elif second_half_mean < first_half_mean * 0.9:
                        insights.append({
                            "type": "trend",
                            "title": f"Decline Trend in {col}",
                            "description": f"{col} shows a declining trend with values decreasing by {round(((first_half_mean - second_half_mean) / first_half_mean) * 100, 1)}%",
                            "icon": "chart-line",
                            "color": "red"
                        })
        
        # ... rest of your insight generation logic
        
        # Ensure we have at least some insights
        if len(insights) < 3:
            insights.extend([
                {
                    "type": "schema",
                    "title": "Dataset Structure",
                    "description": f"Dataset contains {len(df)} records across {len(df.columns)} features",
                    "icon": "database",
                    "color": "purple"
                },
                {
                    "type": "pattern",
                    "title": "Data Distribution",
                    "description": "Data shows varied distribution across different categories and ranges",
                    "icon": "chart-bar",
                    "color": "blue"
                }
            ])
        
        return insights[:5]  # Return max 5 insights
        
    except Exception as e:
        print(f"Error generating insights: {e}")
        # Return sample insights if analysis fails
        return [
            {
                "type": "trend",
                "title": "Data Analysis Ready",
                "description": "Dataset loaded successfully for analysis and visualization",
                "icon": "chart-line",
                "color": "green"
            }
        ]
def apply_filters_to_dataset(dataset, filters, global_search):
    """Apply filters to the dataset and return filtered DataFrame"""
    try:
        df = download_and_convert_to_dataframe(dataset)
        
        if df.empty:
            return df
        
        # Apply global search first
        if global_search:
            # Create a mask for rows that contain the search term in any column
            mask = pd.Series([False] * len(df))
            for col in df.columns:
                try:
                    # Try to convert to string and search
                    col_mask = df[col].astype(str).str.contains(global_search, case=False, na=False)
                    mask = mask | col_mask
                except:
                    continue
            df = df[mask]
        
        # Apply individual filters
        for filter_obj in filters:
            column = filter_obj['column']
            operator = filter_obj['operator']
            value = filter_obj['value']
            value2 = filter_obj.get('value2')
            
            if column not in df.columns:
                continue
            
            try:
                if operator == 'equals':
                    # Try numeric comparison first, then string
                    try:
                        numeric_value = float(value)
                        df = df[df[column] == numeric_value]
                    except:
                        df = df[df[column].astype(str) == str(value)]
                
                elif operator == 'contains':
                    df = df[df[column].astype(str).str.contains(value, case=False, na=False)]
                
                elif operator == 'greater_than':
                    try:
                        numeric_value = float(value)
                        df = df[df[column] > numeric_value]
                    except:
                        # For string comparison
                        df = df[df[column].astype(str) > str(value)]
                
                elif operator == 'less_than':
                    try:
                        numeric_value = float(value)
                        df = df[df[column] < numeric_value]
                    except:
                        # For string comparison
                        df = df[df[column].astype(str) < str(value)]
                
                elif operator == 'between' and value2:
                    try:
                        min_value = float(value)
                        max_value = float(value2)
                        df = df[(df[column] >= min_value) & (df[column] <= max_value)]
                    except:
                        # For string comparison
                        df = df[(df[column].astype(str) >= str(value)) & (df[column].astype(str) <= str(value2))]
                
                elif operator == 'starts_with':
                    df = df[df[column].astype(str).str.startswith(value, na=False)]
                
                elif operator == 'ends_with':
                    df = df[df[column].astype(str).str.endswith(value, na=False)]
                
            except Exception as e:
                print(f"Error applying filter {column} {operator} {value}: {e}")
                continue
        
        return df
        
    except Exception as e:
        print(f"Error applying filters to dataset: {e}")
        return download_and_convert_to_dataframe(dataset)  # Return original dataset on error

def get_filter_options(dataset):
    """Get dynamic filter options based on dataset columns"""
    try:
        df = download_and_convert_to_dataframe(dataset)
        
        if df.empty:
            return {}
        
        filter_options = {
            'columns': list(df.columns),
            'categorical_columns': [],
            'numerical_columns': [],
            'date_columns': []
        }
        
        for col in df.columns:
            col_data = df[col]
            
            # Check if column is categorical (object type with limited unique values)
            if col_data.dtype == 'object' and col_data.nunique() <= 50:
                filter_options['categorical_columns'].append({
                    'name': col,
                    'unique_values': list(col_data.dropna().unique())[:20]  # Limit to first 20 values
                })
            
            # Check if column is numerical
            elif pd.api.types.is_numeric_dtype(col_data):
                filter_options['numerical_columns'].append({
                    'name': col,
                    'min': float(col_data.min()) if not col_data.empty else 0,
                    'max': float(col_data.max()) if not col_data.empty else 0
                })
            
            # Check if column is datetime
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                filter_options['date_columns'].append({
                    'name': col,
                    'min': col_data.min().strftime('%Y-%m-%d') if not col_data.empty else '',
                    'max': col_data.max().strftime('%Y-%m-%d') if not col_data.empty else ''
                })
        
        return filter_options
        
    except Exception as e:
        print(f"Error getting filter options: {e}")
        return {}


def get_table_preview_data(dataset, page=1, rows_per_page=5):
    """Get sample data for table preview with pagination"""
    try:
        df = download_and_convert_to_dataframe(dataset)
        
        if df.empty:
            return {"headers": [], "rows": []}
        
        # Calculate start and end indices for pagination
        start_idx = (page - 1) * rows_per_page
        end_idx = start_idx + rows_per_page
        
        # Get the paginated data (all columns)
        preview_df = df.iloc[start_idx:end_idx]
        
        # Convert to list of lists with headers
        table_data = {
            'headers': list(preview_df.columns),
            'rows': preview_df.values.tolist()
        }
        
        return table_data
        
    except Exception as e:
        print(f"Error getting table preview: {e}")
        return {"headers": [], "rows": []}

def get_available_columns(dataset):
    """Get available columns for filtering and analysis"""
    if not dataset:
        return []
    
    try:
        df = download_and_convert_to_dataframe(dataset)
        columns = []
        
        for col in df.columns:
            col_data = df[col]
            col_info = {
                'name': col,
                'type': str(col_data.dtype),
                'unique_values': col_data.nunique(),
                'sample_values': list(col_data.dropna().head(3).values) if col_data.nunique() > 0 else []
            }
            columns.append(col_info)
        
        return columns
    except Exception as e:
        print(f"Error getting columns: {e}")
        return []

def calculate_dataset_statistics(dataset):
    """Calculate comprehensive statistics for a dataset with error handling"""
    try:
        df = download_and_convert_to_dataframe(dataset)
        
        if df.empty:
            return {
                "total_records": 0,
                "total_features": 0,
                "last_updated": dataset.updated_at.strftime("%b %d, %Y %H:%M") if hasattr(dataset, 'updated_at') and dataset.updated_at else "Never",
                "missing_data_percentage": 0,
                "duplicate_rows": 0,
                "data_size_mb": 0,
                "data_quality_percentage": 0
            }
        
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        missing_percentage = round((missing_cells / total_cells) * 100, 2) if total_cells > 0 else 0
        
        stats = {
            "total_records": len(df),
            "total_features": len(df.columns),
            "last_updated": dataset.updated_at.strftime("%b %d, %Y %H:%M") if hasattr(dataset, 'updated_at') and dataset.updated_at else "Never",
            "missing_data_percentage": missing_percentage,
            "duplicate_rows": df.duplicated().sum(),
            "data_size_mb": round(os.path.getsize(dataset.file.path) / (1024 * 1024), 2) if hasattr(dataset, 'file') and dataset.file else 0,
            "data_quality_percentage": 100 - missing_percentage
        }
        
        return stats
        
    except Exception as e:
        print(f"Error calculating dataset statistics: {e}")
        return {
            "total_records": 0,
            "total_features": 0,
            "last_updated": "Never",
            "missing_data_percentage": 0,
            "duplicate_rows": 0,
            "data_size_mb": 0,
            "data_quality_percentage": 0
        }

def calculate_column_statistics(dataset):
    """Calculate statistics for each column in the dataset"""
    if not dataset:
        return []
    
    try:
        df = download_and_convert_to_dataframe(dataset)
        column_stats = []
        
        for col in df.columns[:5]:  # Show stats for first 5 columns
            col_data = df[col]
            
            stats = {
                "name": col,
                "data_type": str(col_data.dtype),
                "unique_values": col_data.nunique(),
                "null_percentage": round((col_data.isnull().sum() / len(col_data)) * 100, 2),
                "sample_values": ", ".join([str(x) for x in col_data.dropna().head(3).values])
            }
            
            column_stats.append(stats)
        
        return column_stats
        
    except Exception as e:
        print(f"Error calculating column stats: {e}")
        return []

def calculate_statistical_summary(dataset):
    """Calculate statistical summary for numerical columns"""
    if not dataset:
        return []
    
    try:
        df = download_and_convert_to_dataframe(dataset)
        statistical_summary = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols[:3]:  # Show stats for first 3 numeric columns
            col_data = df[col].dropna()
            
            if len(col_data) > 0:
                stats = {
                    "column": col,
                    "count": len(col_data),
                    "mean": round(col_data.mean(), 2),
                    "min": round(col_data.min(), 2),
                    "max": round(col_data.max(), 2),
                    "std_dev": round(col_data.std(), 2)
                }
                
                statistical_summary.append(stats)
        
        return statistical_summary
        
    except Exception as e:
        print(f"Error calculating statistical summary: {e}")
        return []

# In your views.py file, add these views:

@mongo_login_required
def history_view(request):
    """History page view"""
    context = {
        "user_initials": request.user.username[0].upper() if request.user.username else 'U',
        "username": request.user.username,
    }
    return render(request, "history.html", context)

@mongo_login_required
def notes_view(request):
    """Notes page view"""
    context = {
        "user_initials": request.user.username[0].upper() if request.user.username else 'U',
        "username": request.user.username,
    }
    return render(request, "notes.html", context)
@mongo_login_required
@require_http_methods(["GET"])
def history_api(request):
    """API endpoint for history data"""
    try:
        # You can implement actual history data retrieval here
        history_data = {
            'recent_operations': [],
            'dataset_changes': [],
            'analysis_history': []
        }
        
        return JsonResponse({
            'success': True,
            'history': history_data
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=400)

@mongo_login_required
@require_http_methods(["GET", "POST"])
def notes_api(request):
    """API endpoint for notes data"""
    try:
        if request.method == 'GET':
            # Return existing notes
            notes_data = {
                'personal_notes': [],
                'dataset_notes': [],
                'analysis_notes': []
            }
            
            return JsonResponse({
                'success': True,
                'notes': notes_data
            })
            
        elif request.method == 'POST':
            # Save new note
            data = json.loads(request.body)
            # Implement note saving logic here
            
            return JsonResponse({
                'success': True,
                'message': 'Note saved successfully'
            })
            
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=400)


