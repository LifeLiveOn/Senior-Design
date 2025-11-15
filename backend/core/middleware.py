from django.contrib.auth.models import User
from django.utils.deprecation import MiddlewareMixin
from django.conf import settings
from django.contrib.auth.models import User


class DebugSessionUserMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):

        # Only override user IF DEBUG=true
        if settings.DEBUG:
            debug = request.session.get("debug_user")
            if debug:
                try:
                    request.user = User.objects.get(email=debug["email"])
                except User.DoesNotExist:
                    pass

        return self.get_response(request)
