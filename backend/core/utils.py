from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from django.conf import settings


def verify_google_token(token: str) -> dict | None:
    """
    Verifies the Google OAuth2 ID token.

    Args:
        token (str): The ID token from the Google Sign-In button.

    Returns:
        dict: The decoded token payload (user info) if valid, otherwise None.
    """
    try:
        # Verify the token and audience
        idinfo = id_token.verify_oauth2_token(
            token,
            google_requests.Request(),
            settings.GOOGLE_CLIENT_ID
        )

        # Validate issuer
        issuer = idinfo.get('iss')
        if issuer not in ('accounts.google.com', 'https://accounts.google.com'):
            raise ValueError('Wrong issuer.')

        # Optionally: check that the token is not expired (verify_oauth2_token does this)
        # Optionally: check hosted_domain (hd) if restricting to G Suite domain.

        return idinfo

    except ValueError as e:
        # Log the error if needed
        return None
