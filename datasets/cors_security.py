# datasets/cors_security.py
class SecureCorsMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        return response  # Just pass through, let the view handle CORS