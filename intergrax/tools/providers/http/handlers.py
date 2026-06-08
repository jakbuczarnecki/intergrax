# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.http.contracts import HttpRequestInput, HttpRequestOutput
from intergrax.tools.providers.http.service import http_request


class HttpRequestHandler(ServiceToolHandler[HttpRequestInput, HttpRequestOutput]):
    _service = http_request
