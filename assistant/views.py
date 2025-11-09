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
import io
import uuid
from .models import Workspace, WorkspaceActivity,DataCleaningOperation, DataCleaningTemplate
from datasets.models import Dataset

@login_required
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
        dataset = Dataset.objects.get(id=ObjectId(dataset_id), owner_id=str(request.user.id))
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
        new_dataset = create_cleaned_dataset(df, operation_name, request.user.id, 'missing_values')
        
        # Save operation record
        operation = DataCleaningOperation(
            name=operation_name,
            operation_type='missing_values',
            input_dataset_id=dataset_id,
            output_dataset_id=str(new_dataset.id),
            user_id=str(request.user.id),
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

@login_required
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
        
        dataset = Dataset.objects.get(id=ObjectId(dataset_id), owner_id=str(request.user.id))
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
        new_dataset = create_cleaned_dataset(df_cleaned, operation_name, request.user.id, 'remove_duplicates')
        
        # Save operation record
        operation = DataCleaningOperation(
            name=operation_name,
            operation_type='remove_duplicates',
            input_dataset_id=dataset_id,
            output_dataset_id=str(new_dataset.id),
            user_id=str(request.user.id),
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

@login_required
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
        
        dataset = Dataset.objects.get(id=ObjectId(dataset_id), owner_id=str(request.user.id))
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
        
        if preview_only:
            return JsonResponse({
                'success': True,
                'preview_data': df.head(20).to_dict('records'),
                'original_sample': original_sample,
                'columns': list(df.columns)
            })
        
        # Create new dataset
        operation_name = f"Standardize_Formats_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        new_dataset = create_cleaned_dataset(df, operation_name, request.user.id, 'standardize_formats')
        
        # Save operation record
        operation = DataCleaningOperation(
            name=operation_name,
            operation_type='standardize_formats',
            input_dataset_id=dataset_id,
            output_dataset_id=str(new_dataset.id),
            user_id=str(request.user.id),
            parameters={
                'columns': columns,
                'operations': operations
            },
            result_stats={'total_rows': len(df)},
            status='completed',
            completed_at=datetime.utcnow()
        )
        operation.save()
        
        return JsonResponse({
            'success': True,
            'operation_id': str(operation.id),
            'new_dataset_id': str(new_dataset.id),
            'preview_data': df.head(10).to_dict('records')
        })
        
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)

@login_required
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
        
        dataset = Dataset.objects.get(id=ObjectId(dataset_id), owner_id=str(request.user.id))
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
        new_dataset = create_cleaned_dataset(df, operation_name, request.user.id, 'convert_data_types')
        
        # Save operation record
        operation = DataCleaningOperation(
            name=operation_name,
            operation_type='convert_data_types',
            input_dataset_id=dataset_id,
            output_dataset_id=str(new_dataset.id),
            user_id=str(request.user.id),
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

@login_required
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

@login_required
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

@login_required
@require_http_methods(["GET"])
def get_cleaning_history(request):
    """Get user's data cleaning history"""
    try:
        operations = DataCleaningOperation.objects(user_id=str(request.user.id)).order_by('-created_at')[:50]
        
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

@login_required
@require_http_methods(["GET", "POST", "DELETE"])
@csrf_exempt
def manage_cleaning_templates(request):
    """Manage reusable data cleaning templates"""
    try:
        if request.method == 'GET':
            # Get user's templates
            templates = DataCleaningTemplate.objects(user_id=str(request.user.id))
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
                user_id=str(request.user.id),
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
            template = DataCleaningTemplate.objects.get(id=ObjectId(template_id), user_id=str(request.user.id))
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
@login_required
def dashboard(request):
    datasets = Dataset.objects(owner_id=str(request.user.id))
    workspaces = Workspace.objects(owner_id=str(request.user.id))
    pinned_workspaces = Workspace.objects(owner_id=str(request.user.id), pinned=True)
    
    return render(request, "dashboard.html", {
        "datasets": datasets,
        "workspaces": workspaces,
        "pinned_workspaces": pinned_workspaces
    })

@login_required
def workspace(request):
    # Initialize sample data if no workspaces exist
    user_workspaces = Workspace.objects(owner_id=str(request.user.id))
    if not user_workspaces:
        initialize_sample_data(str(request.user.id))
        user_workspaces = Workspace.objects(owner_id=str(request.user.id))
    
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

@login_required
def table_view(request):
    # Get workspace_id from query parameters if provided
    workspace_id = request.GET.get('workspace')
    
    # Get all datasets for the current user
    datasets = Dataset.objects(owner_id=str(request.user.id))
    
    # Get the selected dataset if provided
    selected_dataset_id = request.GET.get('dataset')
    selected_dataset = None
    if selected_dataset_id:
        try:
            selected_dataset = Dataset.objects.get(id=ObjectId(selected_dataset_id), owner_id=str(request.user.id))
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

@login_required
def get_dataset_preview(request, dataset_id):
    """API endpoint to get dataset preview data"""
    try:
        dataset = Dataset.objects.get(id=ObjectId(dataset_id), owner_id=str(request.user.id))
        
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

@login_required
def transformation(request):
    datasets = Dataset.objects(owner_id=str(request.user.id))
    return render(request, "transformation.html", {
        "datasets": datasets,
        "user_initials": request.user.username[0].upper() if request.user.username else 'U'
    })

@login_required
def schema_page(request):
    datasets = Dataset.objects(owner_id=str(request.user.id))
    return render(request, "schema_page.html", {"datasets": datasets})

# Join Operations API Views
@login_required
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
        left_dataset = Dataset.objects.get(id=ObjectId(left_dataset_id), owner_id=str(request.user.id))
        right_dataset = Dataset.objects.get(id=ObjectId(right_dataset_id), owner_id=str(request.user.id))
        
        # Download and process datasets
        left_df = download_and_convert_to_dataframe(left_dataset)
        right_df = download_and_convert_to_dataframe(right_dataset)
        
        # Perform join
        result_df = perform_join(left_df, right_df, left_column, right_column, join_type)
        
        # Create new dataset record for the join result
        join_dataset = create_join_dataset(result_df, join_name, request.user.id)
        
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

@login_required
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
        left_dataset = Dataset.objects.get(id=ObjectId(left_dataset_id), owner_id=str(request.user.id))
        right_dataset = Dataset.objects.get(id=ObjectId(right_dataset_id), owner_id=str(request.user.id))
        
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

@login_required
@require_http_methods(["GET"])
def get_dataset_columns(request, dataset_id):
    """Get column names from a dataset"""
    try:
        print(f"DEBUG: Getting columns for dataset {dataset_id}")
        
        # Validate dataset_id
        if not dataset_id:
            return JsonResponse({'success': False, 'error': 'Dataset ID is required'}, status=400)
            
        dataset = Dataset.objects.get(id=ObjectId(dataset_id), owner_id=str(request.user.id))
        df = download_and_convert_to_dataframe(dataset)
        
        print(f"DEBUG: Found {len(df.columns)} columns: {list(df.columns)}")
        
        return JsonResponse({
            'success': True,
            'columns': list(df.columns),
            'sample_data': df.head(5).to_dict('records')
        })
        
    except Dataset.DoesNotExist:
        print(f"DEBUG: Dataset not found: {dataset_id}")
        return JsonResponse({'success': False, 'error': 'Dataset not found'}, status=404)
    except InvalidId:
        print(f"DEBUG: Invalid dataset ID: {dataset_id}")
        return JsonResponse({'success': False, 'error': 'Invalid dataset ID'}, status=400)
    except Exception as e:
        print(f"DEBUG: Error getting columns: {str(e)}")
        return JsonResponse({'success': False, 'error': str(e)}, status=400)

# Helper functions
def download_and_convert_to_dataframe(dataset):
    """Download dataset from Supabase and convert to pandas DataFrame"""
    from supabase import create_client
    from django.conf import settings
    
    supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
    
    try:
        # Download file
        file_bytes = supabase.storage.from_(settings.SUPABASE_BUCKET).download(dataset.file_path)
        
        # Convert to DataFrame based on file type
        if dataset.file_name.lower().endswith('.csv'):
            decoded = file_bytes.decode("utf-8")
            return pd.read_csv(io.StringIO(decoded))
        else:
            raise ValueError("Only CSV files are supported for join operations")
    except Exception as e:
        print(f"Error downloading dataset {dataset.id}: {str(e)}")
        raise ValueError(f"Failed to download dataset: {str(e)}")

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
@login_required
@require_http_methods(["POST"])
@csrf_exempt
def create_workspace(request):
    try:
        data = json.loads(request.body)
        workspace = Workspace(
            name=data.get('name', 'New Workspace'),
            description=data.get('description', ''),
            owner_id=str(request.user.id),
            members=data.get('members', []),
            color=data.get('color', 'yellow'),
            dataset_count=data.get('dataset_count', 0)
        )
        workspace.save()
        
        activity = WorkspaceActivity(
            workspace_id=str(workspace.id),
            user_id=str(request.user.id),
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

@login_required
@require_http_methods(["POST"])
@csrf_exempt
def toggle_pin_workspace(request, workspace_id):
    try:
        # Try to find workspace by string ID
        user_workspaces = Workspace.objects(owner_id=str(request.user.id))
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
            user_id=str(request.user.id),
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

@login_required
@require_http_methods(["POST"])
@csrf_exempt
def edit_workspace(request, workspace_id):
    try:
        # Try to find workspace by string ID
        user_workspaces = Workspace.objects(owner_id=str(request.user.id))
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
            user_id=str(request.user.id),
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

@login_required
@require_http_methods(["POST"])
@csrf_exempt
def delete_workspace(request, workspace_id):
    try:
        print(f"Deleting workspace: {workspace_id}")
        
        # Get all workspaces for the user
        user_workspaces = Workspace.objects(owner_id=str(request.user.id))
        
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
            user_id=str(request.user.id),
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

@login_required
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