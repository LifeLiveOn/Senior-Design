# core/authentication.py

from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework.exceptions import AuthenticationFailed


class CookieJWTAuthentication(JWTAuthentication):
    def authenticate(self, request):
        access_token = request.COOKIES.get("access")
        if not access_token:
            return None

        try:
            validated = self.get_validated_token(access_token)
            user = self.get_user(validated)
            return (user, validated)
        except Exception:
            raise AuthenticationFailed("Invalid or expired JWT")
