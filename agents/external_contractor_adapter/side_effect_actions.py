# © Artur Czarnecki. All rights reserved.

"""External Work policy-relevant action ids (GEC-5 consumer vocabulary).

These are domain action labels supplied to the platform
``MeaningfulSideEffectRequest.action`` field. They are not platform enums and
must not be treated as policy rules.
"""

from __future__ import annotations

from typing import Final

ACTION_CREATE_EXTERNAL_WORK: Final = "CREATE_EXTERNAL_WORK"
ACTION_ACCEPT_QUOTE: Final = "ACCEPT_QUOTE"
ACTION_CANCEL_EXTERNAL_WORK: Final = "CANCEL_EXTERNAL_WORK"
