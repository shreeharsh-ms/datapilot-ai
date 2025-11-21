from django.shortcuts import render, redirect
from django.contrib import messages
from datasets.models import Dataset
from .models import CustomUser
from functools import wraps
from django.http import JsonResponse

# ----------------------------
# CUSTOM LOGIN REQUIRED DECORATOR
# ----------------------------
def mongo_login_required(view_func):
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        user_id = request.session.get('user_id')
        if not user_id:
            if request.path.startswith("/api/"):
                return JsonResponse({'success': False, 'error': 'Authentication required'}, status=401)
            else:
                return redirect('login')
        return view_func(request, *args, **kwargs)
    return _wrapped_view

# ----------------------------
# LOGIN VIEW
# ----------------------------
def login_view(request):
    if request.method == "POST":
        username = request.POST.get("username", "").strip()
        password = request.POST.get("password", "").strip()

        if not username or not password:
            messages.error(request, "Please fill all fields")
            return render(request, "users/login.html")

        user_doc = CustomUser.objects(username=username).first()
        if user_doc and user_doc.check_password(password):
            # Set session manually
            request.session['user_id'] = str(user_doc.id)
            request.session['username'] = user_doc.username
            return redirect("/api/assistant/workspace/")
        else:
            messages.error(request, "Invalid username or password")

    return render(request, "users/login.html")

# ----------------------------
# REGISTER VIEW
# ----------------------------
def register(request):
    if request.method == "POST":
        username = request.POST.get("username")
        email = request.POST.get("email")
        password1 = request.POST.get("password1")
        password2 = request.POST.get("password2")

        if not username or not email or not password1 or not password2:
            messages.error(request, "All fields are required")
            return render(request, "users/register.html")

        if password1 != password2:
            messages.error(request, "Passwords do not match")
            return render(request, "users/register.html")

        if CustomUser.objects(username=username).first():
            messages.error(request, "Username already exists")
            return render(request, "users/register.html")

        if CustomUser.objects(email=email).first():
            messages.error(request, "Email already exists")
            return render(request, "users/register.html")

        user_doc = CustomUser(username=username, email=email)
        user_doc.set_password(password1)
        user_doc.save()

        # Set session manually
        request.session['user_id'] = str(user_doc.id)
        request.session['username'] = user_doc.username
        return redirect("/api/assistant/workspace/")

    return render(request, "users/register.html")

# ----------------------------
# LOGOUT
# ----------------------------
def logout_view(request):
    request.session.flush()
    messages.success(request, "You have been logged out successfully.")
    return redirect("login")

# ----------------------------
# HOME
# ----------------------------
def home(request):
    return render(request, "index.html")

# ----------------------------
# DASHBOARD
# ----------------------------
@mongo_login_required
def dashboard(request):
    user_id = request.session.get('user_id')
    user_datasets = Dataset.objects(owner_id=user_id).order_by('-uploaded_at')

    pending_count = Dataset.objects(
        owner_id=user_id,
        metadata__analysis_status="pending"
    ).count()

    return render(request, "dashboard.html", {
        "datasets": user_datasets,
        "pending_count": pending_count,
    })
