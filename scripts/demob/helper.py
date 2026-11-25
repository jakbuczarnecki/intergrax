# © Artur Czarnecki. All rights reserved.

"""Text normalization for the factory E2E demo."""


def normalize(text: str) -> str:
    """Strip, collapse inner whitespace to one space, and lowercase."""
    return " ".join(text.strip().split()).lower()
