from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from .forms import CustomUserCreationForm
from .models import User, UserProfile
from .mongodb import mongo_manager
import json

def register(request):
    if request.method == "POST":
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            try:
                user = form.save()
                
                # Create user profile
                UserProfile.objects.create(user=user)
                
                # Store user in MongoDB
                mongo_user_data = {
                    "username": user.username,
                    "email": user.email,
                    "first_name": user.first_name,
                    "last_name": user.last_name,
                    "company": form.cleaned_data.get('company', ''),
                    "created_at": user.created_at.isoformat(),
                    "postgres_user_id": str(user.id)
                }
                mongo_id = mongo_manager.create_user(mongo_user_data)
                
                if mongo_id:
                    user.mongo_id = mongo_id
                    user.save()
                
                # Log the user in
                login(request, user)
                messages.success(request, f"Welcome to DataPilot AI, {user.username}!")
                return redirect("dashboard")
                
            except Exception as e:
                messages.error(request, f"An error occurred during registration: {str(e)}")
        else:
            messages.error(request, "Please correct the errors below.")
    else:
        form = CustomUserCreationForm()

    return render(request, "users/register.html", {"form": form})

def custom_login(request):
    if request.method == "POST":
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            
            if user is not None:
                login(request, user)
                messages.success(request, f"Welcome back, {username}!")
                
                # Redirect to next parameter or dashboard
                next_url = request.GET.get('next', 'dashboard')
                return redirect(next_url)
            else:
                messages.error(request, "Invalid username or password.")
        else:
            messages.error(request, "Invalid username or password.")
    else:
        form = AuthenticationForm()
    
    return render(request, "users/login.html", {"form": form})


@login_required
def profile(request):
    user_profile, created = UserProfile.objects.get_or_create(user=request.user)
    
    if request.method == "POST":
        # Handle profile updates
        user.first_name = request.POST.get('first_name', '')
        user.last_name = request.POST.get('last_name', '')
        user.email = request.POST.get('email', '')
        user.save()
        
        user_profile.bio = request.POST.get('bio', '')
        user_profile.location = request.POST.get('location', '')
        user_profile.website = request.POST.get('website', '')
        user_profile.theme_preference = request.POST.get('theme', 'light')
        user_profile.save()
        
        messages.success(request, "Profile updated successfully!")
        return redirect('profile')
    
    return render(request, "users/profile.html", {"profile": user_profile})
def home(request):
    return render(request, "index.html")
