"""Verification layer errors — distinct from catalog infrastructure failures."""


class ProductIdentificationVerificationError(ValueError):
    """Raised when verification inputs or decision contracts are invalid."""
