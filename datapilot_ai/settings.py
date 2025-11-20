"""
Django settings for datapilot_ai project.
Production-ready for Vercel + MongoDB + Supabase.
"""

import os
from pathlib import Path
from mongoengine import connect

# ---------------------------------------------------------
# BASE DIRECTORY
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------
# MONGOENGINE CONNECTION (Document DB)
# ---------------------------------------------------------
try:
    connect(
        db="datapilot",
        host=os.environ.get("MONGO_URL", "mongodb://localhost:27017/datapilot")
    )
except Exception as e:
    print("❌ MongoDB connection failed:", e)

# ---------------------------------------------------------
# SUPABASE STORAGE CONFIG
# ---------------------------------------------------------
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://vioeqcdpamatksaalpde.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
SUPABASE_BUCKET = "datasets"

# ---------------------------------------------------------
# SECURITY
# ---------------------------------------------------------
SECRET_KEY = os.environ.get(
    "SECRET_KEY",
    "django-insecure-rilt(np4nj%5nnab^&)u_qx^flm#7^lim=1lbw+xpg*!lc+b_c"
)

DEBUG = os.environ.get("DEBUG", "False") == "True"

ALLOWED_HOSTS = [
    ".vercel.app",
    "localhost",
    "127.0.0.1",
]

# ---------------------------------------------------------
# APPLICATIONS
# ---------------------------------------------------------
INSTALLED_APPS = [
    "whitenoise.runserver_nostatic",  # For static files on Vercel
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",

    # Third-party
    "rest_framework",

    # Project apps
    "users",
    "datasets",
    "assistant",
]

# ---------------------------------------------------------
# MIDDLEWARE
# ---------------------------------------------------------
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "datapilot_ai.urls"

# ---------------------------------------------------------
# TEMPLATES
# ---------------------------------------------------------
TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "datapilot_ai.wsgi.application"

# ---------------------------------------------------------
# SQLITE REMOVED (NOT SUPPORTED ON VERCEL)
# ---------------------------------------------------------
# We don't need Django ORM if using MongoDB only.
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.dummy"
    }
}

# ---------------------------------------------------------
# PASSWORD VALIDATION
# ---------------------------------------------------------
AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

# ---------------------------------------------------------
# INTERNATIONALIZATION
# ---------------------------------------------------------
LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

# ---------------------------------------------------------
# STATIC FILES (Required for Vercel)
# ---------------------------------------------------------
STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"

# ---------------------------------------------------------
# DEFAULT PK
# ---------------------------------------------------------
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# ---------------------------------------------------------
# AUTH REDIRECTS
# ---------------------------------------------------------
LOGIN_REDIRECT_URL = "/api/assistant/workspace/"
LOGOUT_REDIRECT_URL = "/login/"
LOGIN_URL = "/login/"
