from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from datasets.models import Dataset  # important import

@login_required
def dashboard(request):
    user_datasets = Dataset.objects(owner_id=str(request.user.id)).order_by('-uploaded_at')
    
    # Optional: pending analysis count
    pending_count = Dataset.objects( 
        owner_id=str(request.user.id),
        metadata__analysis_status="pending"
    ).count()

    return render(request, "dashboard.html", {
        "datasets": user_datasets,
        "pending_count": pending_count,
    })

def register(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect("dashboard")
    else:
        form = UserCreationForm()

    return render(request, "users/register.html", {"form": form})

def home(request):
    return render(request, "index.html")
