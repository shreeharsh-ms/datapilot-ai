import uuid
import csv
import io
import json
from datetime import datetime
from functools import wraps
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.conf import settings
from supabase import create_client
from .forms import DatasetUploadForm
from .models import Dataset, DatasetIncomingData
from django.views.decorators.csrf import csrf_exempt

# -----------------------------
# Auth Wrapper
# -----------------------------
def mongo_login_required(view_func):
    @wraps(view_func)
    def _wrapped(request, *args, **kwargs):
        if not request.session.get("user_id"):
            if request.path.startswith("/api/"):
                return JsonResponse({"success": False, "error": "Authentication required"}, status=401)
            return redirect("/users/login/")
        return view_func(request, *args, **kwargs)
    return _wrapped

# Supabase client
supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)

# -----------------------------
# DASHBOARD
# -----------------------------
@mongo_login_required
def dashboard(request):
    user_id = str(request.session.get("user_id"))
    datasets = Dataset.objects(owner_id=user_id).order_by('-uploaded_at')

    for dataset in datasets:
        try:
            # Generate signed URLs for file-based datasets
            if dataset.is_file_based and dataset.file_path_display:
                signed = supabase.storage.from_(settings.SUPABASE_BUCKET).create_signed_url(
                    dataset.file_path_display, expires_in=3600
                )
                # Update both new and old URL fields for compatibility
                dataset.file_info["signed_url"] = signed["signedURL"]
                dataset.file_url = signed["signedURL"]
            elif dataset.is_api_based:
                # Generate the endpoint URL for API datasets
                dataset.file_info["signed_url"] = dataset.get_api_endpoint(request)
                dataset.file_url = dataset.get_api_endpoint(request)
            else:
                dataset.file_info["signed_url"] = "#"
                dataset.file_url = "#"
        except Exception as e:
            print(f"❌ Error generating signed URL for {dataset.id}: {e}")
            dataset.file_info["signed_url"] = "#"
            dataset.file_url = "#"

    pending_count = Dataset.objects(owner_id=user_id, metadata__analysis_status="pending").count()

    return render(request, "datasets/dashboard.html", {
        "datasets": datasets,
        "pending_count": pending_count,
    })

# -----------------------------
# UPLOAD DATASET
# -----------------------------
@mongo_login_required
def upload_dataset(request):
    if request.method == "POST":
        print("📌 Upload endpoint hit. Method = POST")
        print("🧾 Incoming POST data:", request.POST)
        print("🗂 Incoming FILES data:", request.FILES)

        user_id = str(request.session.get("user_id"))
        dataset_name = request.POST.get("name", "").strip()
        description = request.POST.get("description", "").strip()
        data_source = request.POST.get("data_source", "file")

        print("🔍 Data Source:", data_source)
        print("👤 User ID:", user_id)
        print("📁 Dataset Name:", dataset_name)

        if not dataset_name:
            return render(request, "datasets/upload.html", {
                "form": DatasetUploadForm(request.POST),
                "error": "Dataset name is required"
            })

        # ---------------- FILE UPLOAD ----------------
        if data_source == "file":
            file = request.FILES.get("file")
            if not file:
                return render(request, "datasets/upload.html", {
                    "form": DatasetUploadForm(request.POST),
                    "error": "Please upload a file"
                })

            unique_filename = f"{uuid.uuid4()}_{file.name}"
            storage_path = f"{user_id}/{unique_filename}"

            try:
                # Read file content
                file_content = file.read()
                
                # Upload to Supabase
                upload_response = supabase.storage.from_(settings.SUPABASE_BUCKET).upload(
                    storage_path, file_content
                )
                
                # Get public URL
                file_url = supabase.storage.from_(settings.SUPABASE_BUCKET).get_public_url(storage_path)

                print(f"✅ File uploaded to Supabase: {storage_path}")

                # Create dataset record with new structure
                dataset = Dataset(
                    owner_id=user_id,
                    name=dataset_name,
                    description=description,
                    source_type="file",
                    file_info={
                        "url": file_url,
                        "path": storage_path,
                        "original_name": file.name,
                        "size": file.size,
                        "content_type": file.content_type
                    },
                    connection_info={},
                    metadata={
                        "analysis_status": "pending",
                        "original_filename": file.name
                    },
                    # Backward compatibility fields
                    file_name=file.name,
                    file_type=file.content_type,
                    file_url=file_url,
                    file_path=storage_path
                )
                dataset.save()
                print(f"✅ File dataset saved with ID: {dataset.id}")
                return redirect("dashboard")

            except Exception as e:
                print(f"❌ Error uploading file: {e}")
                return render(request, "datasets/upload.html", {
                    "form": DatasetUploadForm(request.POST),
                    "error": f"File upload failed: {str(e)}"
                })

        # ---------------- MONGODB CONNECTION ----------------
        elif data_source == "mongodb":
            dataset = Dataset(
                owner_id=user_id,
                name=dataset_name,
                description=description,
                source_type="mongodb",
                file_info={},
                connection_info={
                    "uri": request.POST.get("mongo_uri", ""),
                    "database": request.POST.get("mongo_db", ""),
                    "collection": request.POST.get("mongo_collection", ""),
                    "ssl": request.POST.get("mongo_ssl", ""),
                    "read_pref": request.POST.get("mongo_read_pref", "")
                },
                metadata={"analysis_status": "pending"}
            )
            dataset.save()
            print(f"✅ MongoDB dataset saved with ID: {dataset.id}")
            return redirect("dashboard")

        # ---------------- POSTGRESQL CONNECTION ----------------
        elif data_source == "postgresql":
            dataset = Dataset(
                owner_id=user_id,
                name=dataset_name,
                description=description,
                source_type="postgresql",
                file_info={},
                connection_info={
                    "host": request.POST.get("pg_host", ""),
                    "port": request.POST.get("pg_port", ""),
                    "database": request.POST.get("pg_db", ""),
                    "schema": request.POST.get("pg_schema", "public"),
                    "user": request.POST.get("pg_user", ""),
                    "password": request.POST.get("pg_password", ""),
                    "table": request.POST.get("pg_table", ""),
                    "ssl": request.POST.get("pg_ssl", "")
                },
                metadata={"analysis_status": "pending"}
            )
            dataset.save()
            print(f"✅ PostgreSQL dataset saved with ID: {dataset.id}")
            return redirect("dashboard")

        # ---------------- MYSQL CONNECTION ----------------
        elif data_source == "mysql":
            dataset = Dataset(
                owner_id=user_id,
                name=dataset_name,
                description=description,
                source_type="mysql",
                file_info={},
                connection_info={
                    "host": request.POST.get("my_host", ""),
                    "port": request.POST.get("my_port", ""),
                    "database": request.POST.get("my_db", ""),
                    "user": request.POST.get("my_user", ""),
                    "password": request.POST.get("my_password", ""),
                    "table": request.POST.get("my_table", ""),
                    "charset": request.POST.get("my_charset", "utf8mb4"),
                    "ssl": request.POST.get("my_ssl", "")
                },
                metadata={"analysis_status": "pending"}
            )
            dataset.save()
            print(f"✅ MySQL dataset saved with ID: {dataset.id}")
            return redirect("dashboard")

        # ---------------- API CONNECTION (WEBHOOK PUSH) ----------------
        # ---------------- API CONNECTION (WEBHOOK PUSH) ----------------
        elif data_source == "api":
            unique_key = uuid.uuid4().hex[:12]  # endpoint id
            
            website_url = request.POST.get("website_url", "").strip()
            if not website_url:
                return render(request, "datasets/upload.html", {
                    "form": DatasetUploadForm(request.POST),
                    "error": "Website URL is required for API datasets"
                })
            
            dataset = Dataset(
                owner_id=user_id,
                name=dataset_name,
                description=description,
                source_type="api_callback",
                file_info={},
                connection_info={
                    "endpoint_key": unique_key,
                    "website_url": website_url,  # Store the allowed website
                    "auth_token": request.POST.get("api_auth", "").strip(),
                    "json_schema": request.POST.get("api_schema", "").strip(),
                    "strict_validation": bool(request.POST.get("api_strict", False))
                },
                metadata={
                    "analysis_status": "active",
                    "total_received": 0,
                    "last_received": None
                }
            )
            dataset.save()
            
            # Generate the full endpoint URL
            endpoint_url = request.build_absolute_uri(f"/datasets/api/incoming/{unique_key}/")
            
            print(f"✅ API Callback dataset saved with ID: {dataset.id}")
            print(f"🔗 Endpoint URL: {endpoint_url}")
            print(f"🌐 Allowed Website: {website_url}")
            
            # Render the same upload page but with success data
            return render(request, "datasets/upload.html", {
                "form": DatasetUploadForm(),
                "success": True,
                "endpoint_url": endpoint_url,
                "dataset_name": dataset_name,
                "website_url": website_url,  # Pass to template
                "show_api_success": True
            })
    else:
        print("📌 Upload endpoint hit. Method = GET")
        form = DatasetUploadForm()

    return render(request, "datasets/upload.html", {"form": form})

# -----------------------------
# INCOMING API DATA ENDPOINT
# -----------------------------
@csrf_exempt
def api_ingest_data(request, endpoint_key):
    """
    Public API endpoint for receiving data from external websites
    POST /datasets/api/incoming/<endpoint_key>/
    """
    print(f"🔍 API Ingest: {request.method} {request.path}")
    print(f"🔍 Origin: {request.META.get('HTTP_ORIGIN', 'None')}")
    print(f"🔍 Content-Type: {request.content_type}")
    
    # Handle OPTIONS preflight requests - THIS MUST BE FIRST
    if request.method == "OPTIONS":
        print(f"🔍 Handling OPTIONS preflight for {endpoint_key}")
        response = JsonResponse({"status": "ok", "message": "Preflight OK"})
        
        # Get the origin from the request
        origin = request.META.get('HTTP_ORIGIN', '')
        
        # For OPTIONS, we need to validate against the dataset
        try:
            dataset = Dataset.objects(connection_info__endpoint_key=endpoint_key).first()
            if dataset:
                allowed_website = dataset.connection_info.get('website_url', '')
                
                if allowed_website and origin:
                    # Validate the origin
                    from urllib.parse import urlparse
                    try:
                        origin_parsed = urlparse(origin)
                        allowed_parsed = urlparse(allowed_website)
                        
                        if (origin_parsed.scheme == allowed_parsed.scheme and 
                            (origin_parsed.netloc == allowed_parsed.netloc or 
                             origin_parsed.netloc.endswith('.' + allowed_parsed.netloc))):
                            response["Access-Control-Allow-Origin"] = origin
                            print(f"✅ Preflight allowed: {origin}")
                        else:
                            response["Access-Control-Allow-Origin"] = origin  # Still allow for preflight
                            print(f"⚠️  Preflight origin mismatch but allowing: {origin}")
                    except Exception as e:
                        response["Access-Control-Allow-Origin"] = origin or "*"
                        print(f"❌ Preflight URL parse error: {e}")
                else:
                    response["Access-Control-Allow-Origin"] = origin or "*"
            else:
                response["Access-Control-Allow-Origin"] = origin or "*"
        except Exception as e:
            response["Access-Control-Allow-Origin"] = origin or "*"
            print(f"❌ Preflight dataset error: {e}")
        
        # Set required CORS headers
        response["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response["Access-Control-Allow-Headers"] = "Content-Type, X-API-Token, Authorization, X-CSRFToken"
        response["Access-Control-Allow-Credentials"] = "true"
        response["Access-Control-Max-Age"] = "86400"
        
        print(f"🔍 Preflight response headers set")
        return response
    
    # Handle POST requests
    if request.method != "POST":
        print(f"❌ Method not allowed: {request.method}")
        response = JsonResponse({"error": "Only POST method allowed"}, status=405)
        response["Access-Control-Allow-Origin"] = "*"
        return response
    
    # Find the dataset by endpoint key
    print(f"🔍 Looking for dataset with endpoint_key: {endpoint_key}")
    dataset = Dataset.objects(connection_info__endpoint_key=endpoint_key).first()
    
    if not dataset:
        print(f"❌ Dataset not found for endpoint_key: {endpoint_key}")
        response = JsonResponse({"error": "Invalid endpoint"}, status=404)
        response["Access-Control-Allow-Origin"] = "*"
        return response
    
    print(f"✅ Dataset found: {dataset.name} (ID: {dataset.id})")
    
    # Get origin and validate
    origin = request.META.get('HTTP_ORIGIN', '')
    allowed_website = dataset.connection_info.get('website_url', '')
    
    print(f"🔍 Origin: {origin}, Allowed: {allowed_website}")
    
    if origin and allowed_website:
        # Validate origin against allowed website
        from urllib.parse import urlparse
        try:
            origin_parsed = urlparse(origin)
            allowed_parsed = urlparse(allowed_website)
            
            print(f"🔍 Origin parsed: {origin_parsed.netloc}, Allowed parsed: {allowed_parsed.netloc}")
            
            if (origin_parsed.scheme != allowed_parsed.scheme or 
                not (origin_parsed.netloc == allowed_parsed.netloc or 
                     origin_parsed.netloc.endswith('.' + allowed_parsed.netloc))):
                print(f"🚫 Blocked request: Origin {origin} not allowed for {allowed_website}")
                response = JsonResponse({"error": "Origin not allowed"}, status=403)
                response["Access-Control-Allow-Origin"] = "*"
                return response            
            else:
                print(f"✅ Origin validation passed: {origin}")
        except Exception as e:
            print(f"❌ Error parsing URLs: {e}")
            # Continue processing despite parsing error
    
    # Check authentication token if configured
    auth_token = dataset.connection_info.get("auth_token")
    if auth_token:
        provided_token = request.headers.get("X-API-Token") or request.POST.get("token")
        if provided_token != auth_token:
            print(f"❌ Authentication failed for endpoint: {endpoint_key}")
            response = JsonResponse({"error": "Invalid authentication token"}, status=401)
            response["Access-Control-Allow-Origin"] = "*"
            return response
    else:
        print("🔍 No authentication token configured")
    
    try:
        # Parse incoming data
        print(f"🔍 Parsing incoming data, Content-Type: {request.content_type}")
        
        if request.content_type == "application/json":
            try:
                incoming_data = json.loads(request.body)
                print(f"✅ JSON parsed successfully: {type(incoming_data)}")
            except json.JSONDecodeError as e:
                print(f"❌ JSON decode error: {e}")
                response = JsonResponse({"error": "Invalid JSON data"}, status=400)
                response["Access-Control-Allow-Origin"] = "*"
                return response
        else:
            print(f"🔍 Non-JSON content type, using form data")
            incoming_data = dict(request.POST)
            if not incoming_data and request.body:
                try:
                    # Try to parse as JSON anyway
                    incoming_data = json.loads(request.body)
                    print(f"✅ JSON parsed from body anyway")
                except:
                    incoming_data = {"raw_data": request.body.decode('utf-8')}
                    print(f"🔍 Storing as raw data")
        
        print(f"🔍 Incoming data type: {type(incoming_data)}, data: {incoming_data}")
        
        # Add origin information to the stored data
        metadata = {
            "received_at": datetime.utcnow().isoformat(),
            "ip_address": request.META.get('REMOTE_ADDR', ''),
            "user_agent": request.META.get('HTTP_USER_AGENT', '')[:200]  # Limit length
        }
        
        if origin:
            metadata["origin"] = origin
        
        # Ensure incoming_data is a dict for metadata addition
        if not isinstance(incoming_data, dict):
            print(f"🔍 Converting non-dict data to dict")
            incoming_data = {"data": incoming_data}
        
        # Add metadata
        incoming_data["_ingestion_metadata"] = metadata
        
        # Validate JSON schema if configured
        json_schema = dataset.connection_info.get("json_schema")
        if json_schema:
            try:
                schema = json.loads(json_schema)
                if isinstance(schema, dict) and isinstance(incoming_data, dict):
                    print(f"🔍 Validating against JSON schema")
                    for key, expected_type in schema.items():
                        if key in incoming_data:
                            actual_type = type(incoming_data[key]).__name__
                            if expected_type != actual_type:
                                if not dataset.connection_info.get("strict_validation"):
                                    try:
                                        if expected_type == "number":
                                            incoming_data[key] = float(incoming_data[key])
                                        elif expected_type == "string":
                                            incoming_data[key] = str(incoming_data[key])
                                        elif expected_type == "boolean":
                                            incoming_data[key] = bool(incoming_data[key])
                                        print(f"🔍 Auto-converted {key} to {expected_type}")
                                    except Exception as conv_error:
                                        print(f"⚠️  Could not convert {key}: {conv_error}")
            except json.JSONDecodeError:
                print("⚠️  Invalid JSON schema, skipping validation")
        
        # Store the incoming data
        print(f"🔍 Saving incoming data to database")
        incoming_record = DatasetIncomingData(
            dataset_id=str(dataset.id),
            user_id=dataset.owner_id,
            data=incoming_data,
            received_at=datetime.utcnow()
        )
        incoming_record.save()
        
        # Update dataset metadata
        dataset.metadata["last_received"] = datetime.utcnow().isoformat()
        dataset.metadata["total_received"] = dataset.metadata.get("total_received", 0) + 1
        dataset.save()
        
        print(f"✅ Incoming data saved for dataset: {dataset.id}")
        print(f"📊 Total received: {dataset.metadata['total_received']}")
        
        # Prepare success response
        response_data = {
            "success": True,
            "message": "Data received successfully",
            "record_id": str(incoming_record.id),
            "dataset_id": str(dataset.id),
            "endpoint_key": endpoint_key,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        response = JsonResponse(response_data)
        
        # Set CORS headers for POST response
        if origin and allowed_website:
            # Use the validated origin
            response["Access-Control-Allow-Origin"] = origin
        else:
            response["Access-Control-Allow-Origin"] = "*"
            
        response["Access-Control-Allow-Credentials"] = "true"
        
        print(f"✅ Request completed successfully")
        return response
        
    except Exception as e:
        print(f"❌ Error processing incoming data: {e}")
        import traceback
        traceback.print_exc()
        
        error_response = JsonResponse({
            "success": False,
            "error": f"Failed to process data: {str(e)}",
            "endpoint_key": endpoint_key
        }, status=500)
        
        error_response["Access-Control-Allow-Origin"] = "*"
        return error_response
# -----------------------------
# DELETE DATASET
# -----------------------------
@mongo_login_required
def delete_dataset(request, dataset_id):
    dataset = Dataset.objects(id=dataset_id, owner_id=str(request.session.get("user_id"))).first()
    if not dataset:
        return redirect("dashboard")

    # Delete file from Supabase if it's a file dataset
    if dataset.is_file_based and dataset.file_path_display:
        try:
            supabase.storage.from_(settings.SUPABASE_BUCKET).remove([dataset.file_path_display])
            print(f"✅ Deleted file from Supabase: {dataset.file_path_display}")
        except Exception as e:
            print(f"❌ Error deleting file from Supabase: {e}")

    # Also delete any incoming data for API datasets
    if dataset.is_api_based:
        try:
            DatasetIncomingData.objects(dataset_id=str(dataset.id)).delete()
            print(f"✅ Deleted incoming data for API dataset: {dataset.id}")
        except Exception as e:
            print(f"❌ Error deleting incoming data: {e}")

    dataset.delete()
    print(f"✅ Deleted dataset: {dataset_id}")
    return redirect("dashboard")

# -----------------------------
# SIGNED URL
# -----------------------------
@mongo_login_required
def get_signed_url(request, dataset_id):
    dataset = Dataset.objects(id=dataset_id, owner_id=str(request.session.get("user_id"))).first()
    if not dataset:
        return JsonResponse({"error": "Dataset not found"}, status=404)

    if not dataset.is_file_based or not dataset.file_path_display:
        return JsonResponse({"error": "No file available for this dataset"}, status=400)

    try:
        signed = supabase.storage.from_(settings.SUPABASE_BUCKET).create_signed_url(
            dataset.file_path_display, expires_in=3600
        )
        return JsonResponse({"signed_url": signed["signedURL"]})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

# -----------------------------
# PREVIEW DATASET
# -----------------------------
@mongo_login_required
def preview_dataset(request, dataset_id):
    print(f"🔍 Preview requested for dataset: {dataset_id}")
    
    # Fetch dataset for current user
    dataset = Dataset.objects(id=dataset_id, owner_id=str(request.session.get("user_id"))).first()
    if not dataset:
        print("❌ Dataset not found")
        return JsonResponse({"error": "Dataset not found"}, status=404)

    source_type = dataset.source_type
    max_rows = int(request.GET.get('limit', 50))

    print(f"🔍 Previewing {source_type} dataset: {dataset.display_name}")

    try:
        # ---------------- FILE PREVIEW ----------------
        if dataset.is_file_based:
            file_path = dataset.file_path_display
            if not file_path:
                return JsonResponse({"error": "No file available for preview"}, status=400)

            file_bytes = supabase.storage.from_(settings.SUPABASE_BUCKET).download(file_path)
            
            # Get original filename for type detection
            original_name = dataset.file_info.get("original_name", dataset.file_name or "").lower()
            
            if original_name.endswith(".csv"):
                # Use csv module for simple CSV reading
                decoded = file_bytes.decode("utf-8")
                reader = csv.reader(io.StringIO(decoded))
                rows = []
                for i, row in enumerate(reader):
                    rows.append(row)
                    if i >= max_rows:
                        break
                
                if rows:
                    columns = rows[0]
                    data_rows = rows[1:max_rows+1] if len(rows) > 1 else []
                else:
                    columns = []
                    data_rows = []
                    
                return JsonResponse({
                    "columns": columns,
                    "rows": data_rows,
                    "total_rows": len(data_rows),
                    "dataset_name": dataset.display_name,
                    "source_type": dataset.get_connection_type_display()
                })
                
            else:
                return JsonResponse({
                    "error": f"Preview not supported for {original_name}. Only CSV preview is available."
                }, status=400)

        # ---------------- MONGODB PREVIEW ----------------
        elif dataset.source_type == "mongodb":
            try:
                from pymongo import MongoClient
                from bson import ObjectId

                conn_info = dataset.connection_info

                uri = conn_info.get("uri", "").strip()
                database_name = conn_info.get("database", "").strip()
                collection_name = conn_info.get("collection", "").strip()

                if not uri:
                    return JsonResponse({"error": "MongoDB URI required"}, status=400)
                if not database_name:
                    return JsonResponse({"error": "MongoDB database required"}, status=400)
                if not collection_name:
                    return JsonResponse({"error": "MongoDB collection required"}, status=400)

                print(f"🔧 MongoDB → DB: {database_name}, Collection: {collection_name}")

                # Add stable params to avoid DNS issues
                if "retryWrites" not in uri:
                    uri += "&retryWrites=true"
                if "tls" not in uri and "ssl" not in uri:
                    uri += "&tls=true"
                if "directConnection" not in uri:
                    uri += "&directConnection=false"

                # Faster connection (no DNS resolution choke)
                client = MongoClient(uri, serverSelectionTimeoutMS=5000, connectTimeoutMS=5000)

                db = client.get_database(database_name)
                collection = db[collection_name]

                # Fetch sample documents
                docs = list(collection.find().limit(max_rows))
                total_count = collection.estimated_document_count()

                if not docs:
                    client.close()
                    return JsonResponse({
                        "columns": ["Message"],
                        "rows": [["No documents found"]],
                        "total_rows": 0
                    })

                # Extract dynamic columns
                columns = sorted({key for doc in docs for key in doc.keys()})

                # Convert rows
                rows = []
                for doc in docs:
                    row = []
                    for col in columns:
                        val = doc.get(col, "")
                        if isinstance(val, ObjectId):
                            val = str(val)
                        elif isinstance(val, (dict, list)):
                            val = json.dumps(val, default=str)
                        row.append(val)
                    rows.append(row)

                client.close()
                print("🔌 MongoDB connection closed")

                return JsonResponse({
                    "columns": columns,
                    "rows": rows,
                    "total_rows": total_count,
                    "dataset_name": dataset.display_name,
                    "source_type": dataset.get_connection_type_display()
                })

            except Exception as e:
                print(f"❌ MongoDB Preview Error: {e}")
                return JsonResponse({
                    "error": f"MongoDB preview failed: {str(e)}"
                }, status=500)

        # ---------------- POSTGRESQL PREVIEW ----------------
        elif dataset.source_type == "postgresql":
            try:
                # Try to import psycopg2
                try:
                    import psycopg2
                    from psycopg2.extras import RealDictCursor
                except ImportError as e:
                    print(f"❌ psycopg2 not installed: {e}")
                    return JsonResponse({
                        "error": "PostgreSQL support not available. Please install psycopg2: pip install psycopg2-binary"
                    }, status=500)
                
                conn_info = dataset.connection_info
                host = conn_info.get("host", "localhost").strip()
                port = conn_info.get("port", "5432").strip()
                database = conn_info.get("database", "").strip()
                user = conn_info.get("user", "").strip()
                password = conn_info.get("password", "").strip()
                schema = conn_info.get("schema", "public").strip()
                table = conn_info.get("table", "").strip()
                
                print(f"🔧 PostgreSQL Connection Details:")
                print(f"   Host: {host}, Port: {port}")
                print(f"   Database: {database}, User: {user}")
                print(f"   Schema: {schema}, Table: {table}")
                
                if not all([host, database, user, table]):
                    return JsonResponse({
                        "error": "Missing PostgreSQL connection parameters (host, database, user, table)"
                    }, status=400)
                
                # Connect to PostgreSQL
                try:
                    conn = psycopg2.connect(
                        host=host,
                        port=port,
                        database=database,
                        user=user,
                        password=password,
                        connect_timeout=10
                    )
                except Exception as e:
                    print(f"❌ PostgreSQL connection failed: {e}")
                    return JsonResponse({
                        "error": f"PostgreSQL connection failed: {str(e)}"
                    }, status=500)
                
                try:
                    # Get table data
                    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                        # Check if table exists
                        cursor.execute("""
                            SELECT EXISTS (
                                SELECT FROM information_schema.tables 
                                WHERE table_schema = %s AND table_name = %s
                            )
                        """, (schema, table))
                        table_exists = cursor.fetchone()['exists']
                        
                        if not table_exists:
                            conn.close()
                            return JsonResponse({
                                "error": f"Table '{schema}.{table}' not found"
                            }, status=400)
                        
                        # Get column names
                        cursor.execute("""
                            SELECT column_name 
                            FROM information_schema.columns 
                            WHERE table_schema = %s AND table_name = %s 
                            ORDER BY ordinal_position
                        """, (schema, table))
                        columns = [row['column_name'] for row in cursor.fetchall()]
                        
                        # Get sample data
                        cursor.execute(f'SELECT * FROM "{schema}"."{table}" LIMIT %s', (max_rows,))
                        rows_data = cursor.fetchall()
                        
                        # Convert to list of lists
                        rows = []
                        for row in rows_data:
                            row_list = []
                            for col in columns:
                                value = row.get(col, "")
                                if value is None:
                                    value = ""
                                row_list.append(str(value))
                            rows.append(row_list)
                    
                    conn.close()
                    
                    return JsonResponse({
                        "columns": columns,
                        "rows": rows,
                        "total_rows": len(rows),
                        "dataset_name": dataset.display_name,
                        "source_type": dataset.get_connection_type_display()
                    })
                    
                except Exception as e:
                    conn.close()
                    print(f"❌ PostgreSQL query error: {e}")
                    return JsonResponse({
                        "error": f"PostgreSQL query error: {str(e)}"
                    }, status=500)
                
            except Exception as e:
                print(f"❌ PostgreSQL general error: {e}")
                return JsonResponse({
                    "error": f"PostgreSQL error: {str(e)}"
                }, status=500)

        # ---------------- MYSQL PREVIEW ----------------
        elif dataset.source_type == "mysql":
            try:
                # Try to import mysql connector
                try:
                    import mysql.connector
                    from mysql.connector import Error
                except ImportError as e:
                    print(f"❌ mysql-connector not installed: {e}")
                    return JsonResponse({
                        "error": "MySQL support not available. Please install mysql-connector-python: pip install mysql-connector-python"
                    }, status=500)
                
                conn_info = dataset.connection_info
                host = conn_info.get("host", "localhost").strip()
                port = conn_info.get("port", "3306").strip()
                database = conn_info.get("database", "").strip()
                user = conn_info.get("user", "").strip()
                password = conn_info.get("password", "").strip()
                table = conn_info.get("table", "").strip()
                
                print(f"🔧 MySQL Connection Details:")
                print(f"   Host: {host}, Port: {port}")
                print(f"   Database: {database}, User: {user}")
                print(f"   Table: {table}")
                
                if not all([host, database, user, table]):
                    return JsonResponse({
                        "error": "Missing MySQL connection parameters (host, database, user, table)"
                    }, status=400)
                
                # Connect to MySQL
                try:
                    conn = mysql.connector.connect(
                        host=host,
                        port=port,
                        database=database,
                        user=user,
                        password=password,
                        connection_timeout=10
                    )
                except Exception as e:
                    print(f"❌ MySQL connection failed: {e}")
                    return JsonResponse({
                        "error": f"MySQL connection failed: {str(e)}"
                    }, status=500)
                
                try:
                    cursor = conn.cursor(dictionary=True)
                    
                    # Check if table exists
                    cursor.execute("""
                        SELECT COUNT(*) as count 
                        FROM information_schema.tables 
                        WHERE table_schema = %s AND table_name = %s
                    """, (database, table))
                    table_exists = cursor.fetchone()['count'] > 0
                    
                    if not table_exists:
                        cursor.close()
                        conn.close()
                        return JsonResponse({
                            "error": f"Table '{table}' not found in database '{database}'"
                        }, status=400)
                    
                    # Get table data
                    cursor.execute(f"SELECT * FROM `{table}` LIMIT %s", (max_rows,))
                    rows_data = cursor.fetchall()
                    
                    if not rows_data:
                        # If no data, get column structure
                        cursor.execute(f"DESCRIBE `{table}`")
                        columns = [row['Field'] for row in cursor.fetchall()]
                        rows = []
                    else:
                        # Extract columns from first row
                        columns = list(rows_data[0].keys())
                        
                        # Convert to list of lists
                        rows = []
                        for row in rows_data:
                            row_list = []
                            for col in columns:
                                value = row.get(col, "")
                                if value is None:
                                    value = ""
                                row_list.append(str(value))
                            rows.append(row_list)
                    
                    cursor.close()
                    conn.close()
                    
                    return JsonResponse({
                        "columns": columns,
                        "rows": rows,
                        "total_rows": len(rows),
                        "dataset_name": dataset.display_name,
                        "source_type": dataset.get_connection_type_display()
                    })
                    
                except Exception as e:
                    cursor.close()
                    conn.close()
                    print(f"❌ MySQL query error: {e}")
                    return JsonResponse({
                        "error": f"MySQL query error: {str(e)}"
                    }, status=500)
                
            except Exception as e:
                print(f"❌ MySQL general error: {e}")
                return JsonResponse({
                    "error": f"MySQL error: {str(e)}"
                }, status=500)

        # ---------------- API CALLBACK PREVIEW (Webhook Data) ----------------
        elif dataset.source_type == "api_callback":
            try:
                # Get latest incoming data for this dataset
                records = DatasetIncomingData.objects(
                    dataset_id=str(dataset.id)
                ).order_by('-received_at').limit(max_rows)
                
                if not records:
                    return JsonResponse({
                        "columns": ["Message"],
                        "rows": [["No data received yet via API"]],
                        "total_rows": 0,
                        "dataset_name": dataset.display_name,
                        "source_type": "API (Webhook)",
                        "last_received": dataset.metadata.get("last_received", "Never"),
                        "total_received": dataset.metadata.get("total_received", 0)
                    })
                
                # Extract columns from the first record
                sample_data = records[0].data
                if isinstance(sample_data, dict):
                    columns = list(sample_data.keys())
                elif isinstance(sample_data, list) and sample_data:
                    # Handle array data
                    if isinstance(sample_data[0], dict):
                        columns = list(sample_data[0].keys())
                    else:
                        columns = ["value"]
                else:
                    columns = ["data"]
                
                # Convert records to rows
                rows = []
                for record in records:
                    data = record.data
                    if isinstance(data, dict):
                        row = []
                        for col in columns:
                            value = data.get(col, "")
                            if value is None:
                                value = ""
                            elif not isinstance(value, (str, int, float, bool)):
                                value = str(value)
                            row.append(value)
                        rows.append(row)
                    elif isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                row = []
                                for col in columns:
                                    value = item.get(col, "")
                                    if value is None:
                                        value = ""
                                    elif not isinstance(value, (str, int, float, bool)):
                                        value = str(value)
                                    row.append(value)
                                rows.append(row)
                            else:
                                rows.append([str(item)])
                    else:
                        rows.append([str(data)])
                
                total_count = DatasetIncomingData.objects(dataset_id=str(dataset.id)).count()
                
                return JsonResponse({
                    "columns": columns,
                    "rows": rows[:max_rows],  # Ensure we don't exceed limit
                    "total_rows": total_count,
                    "dataset_name": dataset.display_name,
                    "source_type": "API (Webhook)",
                    "last_received": dataset.metadata.get("last_received", "Never"),
                    "total_received": dataset.metadata.get("total_received", 0)
                })
                
            except Exception as e:
                print(f"❌ API dataset preview error: {e}")
                return JsonResponse({
                    "error": f"Failed to preview API data: {str(e)}"
                }, status=500)

        # ---------------- API PREVIEW (Legacy - Fetch from external API) ----------------
        elif dataset.source_type == "api":
            try:
                import requests
                import json
                
                conn_info = dataset.connection_info
                url = conn_info.get("url", "").strip()
                method = conn_info.get("method", "GET").upper()
                headers_str = conn_info.get("headers", "{}").strip()
                params_str = conn_info.get("params", "{}").strip()
                
                print(f"🔧 API Connection Details:")
                print(f"   URL: {url}")
                print(f"   Method: {method}")
                print(f"   Headers: {headers_str}")
                print(f"   Params: {params_str}")
                
                if not url:
                    return JsonResponse({
                        "error": "Missing API URL"
                    }, status=400)
                
                # Parse headers and parameters
                try:
                    headers = json.loads(headers_str) if headers_str else {}
                    params = json.loads(params_str) if params_str else {}
                except json.JSONDecodeError as e:
                    return JsonResponse({
                        "error": f"Invalid JSON in headers or parameters: {str(e)}"
                    }, status=400)
                
                # Make API request
                try:
                    if method == "GET":
                        response = requests.get(url, headers=headers, params=params, timeout=10)
                    elif method == "POST":
                        response = requests.post(url, headers=headers, json=params, timeout=10)
                    else:
                        return JsonResponse({
                            "error": f"Unsupported HTTP method: {method}"
                        }, status=400)
                    
                    response.raise_for_status()
                    
                except requests.exceptions.RequestException as e:
                    print(f"❌ API request failed: {e}")
                    return JsonResponse({
                        "error": f"API request failed: {str(e)}"
                    }, status=500)
                
                # Parse response
                content_type = response.headers.get('content-type', '')
                
                if 'application/json' in content_type:
                    try:
                        data = response.json()
                    except json.JSONDecodeError as e:
                        return JsonResponse({
                            "error": f"Invalid JSON response: {str(e)}"
                        }, status=500)
                else:
                    # Try to parse as JSON anyway, fallback to text
                    try:
                        data = response.json()
                    except:
                        data = {"raw_content": response.text}
                
                # Normalize data structure
                if isinstance(data, list):
                    items = data[:max_rows]
                elif isinstance(data, dict):
                    possible_list_keys = ['data', 'results', 'items', 'records']
                    items = None
                    
                    for key in possible_list_keys:
                        if key in data and isinstance(data[key], list):
                            items = data[key][:max_rows]
                            break
                    
                    if items is None:
                        items = [data]
                else:
                    items = [{"value": str(data)}]
                
                # Extract columns and rows
                if items and len(items) > 0:
                    if isinstance(items[0], dict):
                        columns = list(items[0].keys())
                        rows = []
                        for item in items:
                            row = []
                            for col in columns:
                                value = item.get(col, "")
                                if value is None:
                                    value = ""
                                elif not isinstance(value, (str, int, float, bool)):
                                    value = str(value)
                                row.append(value)
                            rows.append(row)
                    else:
                        columns = ["value"]
                        rows = [[str(item)] for item in items]
                else:
                    columns = ["message"]
                    rows = [["No data returned from API"]]
                
                return JsonResponse({
                    "columns": columns,
                    "rows": rows,
                    "total_rows": len(rows),
                    "dataset_name": dataset.display_name,
                    "source_type": dataset.get_connection_type_display(),
                    "api_status": response.status_code
                })
                
            except Exception as e:
                print(f"❌ API general error: {e}")
                return JsonResponse({
                    "error": f"API error: {str(e)}"
                }, status=500)

        else:
            return JsonResponse({"error": "Invalid data source type"}, status=400)

    except Exception as e:
        print(f"❌ Preview general error: {str(e)}")
        return JsonResponse({"error": str(e)}, status=500)