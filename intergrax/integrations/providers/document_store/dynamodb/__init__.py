# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_dynamodb_document_store", "register_dynamodb_integration"]

def __getattr__(name: str):
    if name == "register_dynamodb_integration":
        from intergrax.integrations.providers.document_store.dynamodb.register import register_dynamodb_integration
        return register_dynamodb_integration
    if name == "create_dynamodb_document_store":
        from intergrax.integrations.providers.document_store.dynamodb.bundle import create_dynamodb_document_store
        return create_dynamodb_document_store
    raise AttributeError(name)
