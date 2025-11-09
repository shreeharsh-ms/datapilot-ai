import uuid
from datetime import datetime
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.conf import settings
from django.http import JsonResponse 
import csv
import io

from .forms import DatasetUploadForm
from .models import Dataset
from supabase import create_client

# Initialize Supabase client
supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)

@login_required
def dashboard(request):
    # Fetch datasets for this user
    user_datasets = Dataset.objects(owner_id=str(request.user.id)).order_by('-uploaded_at')

    # Generate signed URL for each dataset
    for dataset in user_datasets:
        try:
            # Signed URL valid for 1 hour
            signed_url = supabase.storage.from_(settings.SUPABASE_BUCKET).create_signed_url(
                dataset.file_path,
                expires_in=60 * 60  # 1 hour
            )
            dataset.file_url = signed_url["signedURL"]
        except Exception as e:
            print("❌ Error generating signed URL:", e)
            dataset.file_url = "#"  # fallback if signing fails

    # Count pending analysis
    pending_count = Dataset.objects(
        owner_id=str(request.user.id),
        metadata__analysis_status="pending"
    ).count()

    return render(request, "datasets/dashboard.html", {
        "datasets": user_datasets,
        "pending_count": pending_count,
    })

@login_required
def upload_dataset(request):
    if request.method == "POST":
        print("🚀 Upload Dataset View Accessed by User:", request.user.id)

        form = DatasetUploadForm(request.POST, request.FILES)

        if form.is_valid():
            file = request.FILES["file"]
            unique_filename = f"{uuid.uuid4()}_{file.name}"
            supabase_path = f"{request.user.id}/{unique_filename}"
            try:
                file_content = file.read()

                # ✅ Upload (your client returns UploadResponse on success)
                res = supabase.storage.from_(settings.SUPABASE_BUCKET).upload(
                    supabase_path, file_content
                )

                print("✅ Supabase upload successful:", res.path)

                # ✅ Public URL extract
                file_url = supabase.storage.from_(settings.SUPABASE_BUCKET).get_public_url(supabase_path)


                # ✅ Save to MongoDB
                dataset = Dataset(
                    owner_id=str(request.user.id),
                    file_name=file.name,
                    file_type=file.content_type,
                    file_url=file_url,
                    file_path=supabase_path,
                    uploaded_at=datetime.utcnow(),
                    metadata={}
                )

                dataset.save()
                print("✅ MongoDB save successful:", dataset.id)

                return redirect("dashboard")

            except Exception as e:
                print("❌ Exception occurred:", str(e))
                return render(request, "datasets/upload.html", {"form": form, "error": str(e)})
        else:
            print("❌ Form validation errors:", form.errors)

    else:
        form = DatasetUploadForm()

    return render(request, "datasets/upload.html", {"form": form})

@login_required
def delete_dataset(request, dataset_id):
    dataset = Dataset.objects(id=dataset_id, owner_id=str(request.user.id)).first()

    if dataset:
        try:
            delete_path = dataset.file_path.strip("/")
            print("🔧 Deleting supabase file:", delete_path)

            delete_res = supabase.storage.from_(settings.SUPABASE_BUCKET).remove([
                delete_path
            ])

            print("✅ Supabase delete response:", delete_res)

        except Exception as e:
            print("❌ Supabase deletion error:", e)

        dataset.delete()

    return redirect("dashboard")

@login_required
def get_signed_url(request, dataset_id):
    """
    Returns a temporary signed URL for a dataset.
    """
    dataset = Dataset.objects(id=dataset_id, owner_id=str(request.user.id)).first()

    if not dataset:
        return JsonResponse({"error": "Dataset not found"}, status=404)

    try:
        signed = supabase.storage.from_(settings.SUPABASE_BUCKET).create_signed_url(
            dataset.file_path,
            expires_in=60 * 60  # 1 hour
        )
        return JsonResponse({"signed_url": signed["signedURL"]})

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

import uuid
from datetime import datetime
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.conf import settings
from django.http import JsonResponse 

from .forms import DatasetUploadForm
from .models import Dataset
from supabase import create_client

# Initialize Supabase client
supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)

@login_required
def dashboard(request):
    # Fetch datasets for this user
    user_datasets = Dataset.objects(owner_id=str(request.user.id)).order_by('-uploaded_at')

    # Generate signed URL for each dataset
    for dataset in user_datasets:
        try:
            # Signed URL valid for 1 hour
            signed_url = supabase.storage.from_(settings.SUPABASE_BUCKET).create_signed_url(
                dataset.file_path,
                expires_in=60 * 60  # 1 hour
            )
            dataset.file_url = signed_url["signedURL"]
        except Exception as e:
            print("❌ Error generating signed URL:", e)
            dataset.file_url = "#"  # fallback if signing fails

    # Count pending analysis
    pending_count = Dataset.objects(
        owner_id=str(request.user.id),
        metadata__analysis_status="pending"
    ).count()

    return render(request, "datasets/dashboard.html", {
        "datasets": user_datasets,
        "pending_count": pending_count,
    })

@login_required
def upload_dataset(request):
    if request.method == "POST":
        print("🚀 Upload Dataset View Accessed by User:", request.user.id)

        form = DatasetUploadForm(request.POST, request.FILES)

        if form.is_valid():
            file = request.FILES["file"]
            unique_filename = f"{uuid.uuid4()}_{file.name}"
            supabase_path = f"{request.user.id}/{unique_filename}"
            try:
                file_content = file.read()

                # ✅ Upload (your client returns UploadResponse on success)
                res = supabase.storage.from_(settings.SUPABASE_BUCKET).upload(
                    supabase_path, file_content
                )

                print("✅ Supabase upload successful:", res.path)

                # ✅ Public URL extract
                file_url = supabase.storage.from_(settings.SUPABASE_BUCKET).get_public_url(supabase_path)


                # ✅ Save to MongoDB
                dataset = Dataset(
                    owner_id=str(request.user.id),
                    file_name=file.name,
                    file_type=file.content_type,
                    file_url=file_url,
                    file_path=supabase_path,
                    uploaded_at=datetime.utcnow(),
                    metadata={}
                )

                dataset.save()
                print("✅ MongoDB save successful:", dataset.id)

                return redirect("dashboard")

            except Exception as e:
                print("❌ Exception occurred:", str(e))
                return render(request, "datasets/upload.html", {"form": form, "error": str(e)})
        else:
            print("❌ Form validation errors:", form.errors)

    else:
        form = DatasetUploadForm()

    return render(request, "datasets/upload.html", {"form": form})

@login_required
def delete_dataset(request, dataset_id):
    dataset = Dataset.objects(id=dataset_id, owner_id=str(request.user.id)).first()

    if dataset:
        try:
            delete_path = dataset.file_path.strip("/")
            print("🔧 Deleting supabase file:", delete_path)

            delete_res = supabase.storage.from_(settings.SUPABASE_BUCKET).remove([
                delete_path
            ])

            print("✅ Supabase delete response:", delete_res)

        except Exception as e:
            print("❌ Supabase deletion error:", e)

        dataset.delete()

    return redirect("dashboard")

@login_required
def get_signed_url(request, dataset_id):
    """
    Returns a temporary signed URL for a dataset.
    """
    dataset = Dataset.objects(id=dataset_id, owner_id=str(request.user.id)).first()

    if not dataset:
        return JsonResponse({"error": "Dataset not found"}, status=404)

    try:
        signed = supabase.storage.from_(settings.SUPABASE_BUCKET).create_signed_url(
            dataset.file_path,
            expires_in=60 * 60  # 1 hour
        )
        return JsonResponse({"signed_url": signed["signedURL"]})

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@login_required
def preview_dataset(request, dataset_id):
    dataset = Dataset.objects(id=dataset_id, owner_id=str(request.user.id)).first()

    if not dataset:
        return JsonResponse({"error": "Dataset not found"}, status=404)

    try:
        # Get limit from query parameters, default to 50
        limit = int(request.GET.get('limit', 50))
        
        file_bytes = supabase.storage.from_(settings.SUPABASE_BUCKET).download(
            dataset.file_path
        )

        if not dataset.file_name.lower().endswith(".csv"):
            return JsonResponse({"error": "Preview only available for CSV files"}, status=400)

        decoded = file_bytes.decode("utf-8")
        reader = csv.reader(io.StringIO(decoded))

        rows = []
        for i, row in enumerate(reader):
            rows.append(row)
            if i >= limit:  # Use the limit parameter
                break

        return JsonResponse({"rows": rows})

    except Exception as e:
        print("❌ Preview error:", str(e))
        return JsonResponse({"error": str(e)}, status=500)
    # ✅ Fetch from MongoDB (MongoEngine)
    dataset = Dataset.objects(id=dataset_id, owner_id=str(request.user.id)).first()

    if not dataset:
        return JsonResponse({"error": "Dataset not found"}, status=404)

    try:
        # ✅ Download raw file bytes from Supabase storage
        file_bytes = supabase.storage.from_(settings.SUPABASE_BUCKET).download(
            dataset.file_path
        )

        # ✅ Only allow CSV preview
        if not dataset.file_name.lower().endswith(".csv"):
            return JsonResponse({"error": "Preview only available for CSV files"}, status=400)

        # ✅ Decode and parse CSV
        decoded = file_bytes.decode("utf-8")
        reader = csv.reader(io.StringIO(decoded))

        rows = []
        for i, row in enumerate(reader):
            rows.append(row)
            if i >= 4:   # first 5 lines
                break

        return JsonResponse({"rows": rows})

    except Exception as e:
        print("❌ Preview error:", str(e))
        return JsonResponse({"error": str(e)}, status=500)