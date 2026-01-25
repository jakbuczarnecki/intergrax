# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.


from __future__ import annotations


class NotAuthenticatedError(Exception):
    """
    Raised when an endpoint requires authentication
    but the request is not authenticated.

    This is a domain-level error.
    HTTP mapping is handled by the global error handler.
    """

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or "Authentication is required.")
