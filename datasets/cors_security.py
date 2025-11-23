import re
from urllib.parse import urlparse
from django.http import JsonResponse
from django.conf import settings
from .models import Dataset

class SecureCorsMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        
        # Only apply to API ingestion endpoints
        if re.match(r'^/datasets/api/incoming/[^/]+/$', request.path):
            endpoint_key = request.path.split('/')[-2]  # Extract endpoint key
            
            try:
                # Find the dataset for this endpoint
                dataset = Dataset.objects(connection_info__endpoint_key=endpoint_key).first()
                
                if dataset:
                    allowed_website = dataset.connection_info.get('website_url', '')
                    origin = request.META.get('HTTP_ORIGIN', '')
                    
                    # Set CORS headers for preflight and actual requests
                    if request.method == 'OPTIONS' or request.method == 'POST':
                        if allowed_website and origin:
                            # Validate origin against allowed website
                            if self.is_origin_allowed(origin, allowed_website):
                                response["Access-Control-Allow-Origin"] = origin
                                print(f"✅ Allowed CORS request from {origin} to {endpoint_key}")
                            else:
                                # Still allow but log the mismatch
                                response["Access-Control-Allow-Origin"] = origin
                                print(f"⚠️  CORS origin mismatch: {origin} not in {allowed_website}")
                        else:
                            # No website configured or no origin header, allow the request
                            response["Access-Control-Allow-Origin"] = origin or "*"
                        
                        # Always set these headers
                        response["Access-Control-Allow-Methods"] = "POST, OPTIONS, GET"
                        response["Access-Control-Allow-Headers"] = "Content-Type, X-API-Token, Authorization, X-CSRFToken"
                        response["Access-Control-Allow-Credentials"] = "true"
                        response["Access-Control-Max-Age"] = "86400"  # 24 hours
                
            except Exception as e:
                print(f"❌ CORS middleware error: {e}")
                # Fallback: allow the request
                response["Access-Control-Allow-Origin"] = "*"
                response["Access-Control-Allow-Methods"] = "POST, OPTIONS"
                response["Access-Control-Allow-Headers"] = "Content-Type, X-API-Token"
        
        return response

    def is_origin_allowed(self, origin, allowed_website):
        """
        Check if the request origin matches the allowed website
        Supports exact matches and subdomains
        """
        try:
            origin_parsed = urlparse(origin)
            allowed_parsed = urlparse(allowed_website)
            
            # Compare schemes and netloc (domain + port)
            if origin_parsed.scheme != allowed_parsed.scheme:
                return False
            
            # Allow subdomains of the allowed website
            origin_domain = origin_parsed.netloc
            allowed_domain = allowed_parsed.netloc
            
            # Exact match or subdomain match
            return (origin_domain == allowed_domain or 
                   origin_domain.endswith('.' + allowed_domain))
            
        except Exception as e:
            print(f"❌ Error parsing URLs in CORS check: {e}")
            return False