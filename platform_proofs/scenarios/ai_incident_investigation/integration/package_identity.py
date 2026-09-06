# © Artur Czarnecki. All rights reserved.

"""Deterministic private package identity for incident investigator lifecycle proofs."""

from __future__ import annotations

import hashlib
import json

from intergrax.agent_distribution.binding import AgentBindingFactoryReference
from intergrax.agent_distribution.identity import AgentPackageIdentity

INCIDENT_INVESTIGATOR_DISTRIBUTION_PACKAGE_ID = "intergrax-ai-incident-investigator"
INCIDENT_INVESTIGATOR_PUBLISHER_ID = "publisher:ai-incident-investigator"
INCIDENT_INVESTIGATOR_PACKAGE_VERSION = "1.0.0"
INCIDENT_INVESTIGATOR_FACTORY_PATH = "incident_investigator_agent.factory.build_agent"
INCIDENT_INVESTIGATOR_FACTORY_REFERENCE = AgentBindingFactoryReference(
    factory_path=INCIDENT_INVESTIGATOR_FACTORY_PATH,
)
INCIDENT_INVESTIGATOR_METADATA_REF = "meta://ai-incident-investigator"
INCIDENT_INVESTIGATOR_CATALOG_ENTRY_ID = "cat-ai-incident-investigator"
INCIDENT_INVESTIGATOR_CATALOG_SOURCE_ID = "builtin-ai-incident-investigator"
INCIDENT_INVESTIGATOR_INSTALLATION_ID = "inst-ai-incident-investigator"
INCIDENT_INVESTIGATOR_INSTALLATION_SLOT_ID = "slot-ai-incident-investigator"
INCIDENT_INVESTIGATOR_APPLICATION_BINDING_ID = "bind-ai-incident-investigator"
INCIDENT_INVESTIGATOR_RUNTIME_REVISION_ID = "rev-ai-incident-investigator-v1"
INCIDENT_INVESTIGATOR_APPLICATION_ID = "scenario_ai_incident_investigation"
INCIDENT_INVESTIGATOR_ENVIRONMENT_ID = "env_ai_incident_prod_validation"

_PACKAGE_IDENTITY_CANON = {
    "distribution_package_id": INCIDENT_INVESTIGATOR_DISTRIBUTION_PACKAGE_ID,
    "package_version": INCIDENT_INVESTIGATOR_PACKAGE_VERSION,
    "factory_path": INCIDENT_INVESTIGATOR_FACTORY_PATH,
    "logical_agent_id": "incident_investigator",
    "proof": "AIPV-1",
}


def incident_investigator_package_digest() -> str:
    payload = json.dumps(_PACKAGE_IDENTITY_CANON, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


INCIDENT_INVESTIGATOR_PACKAGE_DIGEST = incident_investigator_package_digest()


def incident_investigator_package_identity() -> AgentPackageIdentity:
    return AgentPackageIdentity(
        distribution_package_id=INCIDENT_INVESTIGATOR_DISTRIBUTION_PACKAGE_ID,
        package_version=INCIDENT_INVESTIGATOR_PACKAGE_VERSION,
        package_digest=INCIDENT_INVESTIGATOR_PACKAGE_DIGEST,
    )
