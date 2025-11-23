import re
from urllib.parse import urlparse
from django.http import JsonResponse
from .models import Dataset

class SecureCorsMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        
        # Only apply CORS to API ingestion endpoints
        if request.path.startswith('/datasets/api/incoming/'):
            endpoint_key = request.path.split('/')[-2]  # Extract endpoint key from URL
            
            try:
                # Find the dataset for this endpoint
                dataset = Dataset.objects(connection_info__endpoint_key=endpoint_key).first()
                
                if dataset:
                    allowed_website = dataset.connection_info.get('website_url', '')
                    
                    if allowed_website:
                        # Extract origin from request
                        origin = request.META.get('HTTP_ORIGIN', '')
                        
                        if origin and self.is_origin_allowed(origin, allowed_website):
                            response["Access-Control-Allow-Origin"] = origin
                        else:
                            # For requests without Origin header (like curl), allow but log
                            if not origin:
                                print(f"⚠️  Request without Origin header to {endpoint_key}")
                            else:
                                print(f"🚫 Blocked CORS request from {origin} to {endpoint_key}")
                            
                            # Still allow the request but don't set CORS headers
                            # This allows server-to-server calls while blocking browser CORS
                    else:
                        # No website configured, allow all (backward compatibility)
                        response["Access-Control-Allow-Origin"] = "*"
                
                # Always set these headers for preflight
                response["Access-Control-Allow-Methods"] = "POST, OPTIONS"
                response["Access-Control-Allow-Headers"] = "Content-Type, X-API-Token, Authorization"
                response["Access-Control-Allow-Credentials"] = "true"
                
            except Exception as e:
                print(f"❌ CORS middleware error: {e}")
                # Fallback: allow all for safety
                response["Access-Control-Allow-Origin"] = "*"
        
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