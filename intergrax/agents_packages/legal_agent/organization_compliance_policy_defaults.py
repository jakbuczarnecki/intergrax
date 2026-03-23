# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Default plain-text policy for legal compliance checks.

Kept separate from :mod:`legal_agent_config` so this module stays free of
config / agent imports (avoids import cycles).
"""

DEFAULT_ORGANIZATION_COMPLIANCE_POLICY = """Organization policy:
- Intellectual property must be transferred progressively per milestone.
- Unlimited liability is not allowed.
- Automatic renewal clauses require explicit approval.
- Payment terms should not exceed 30 days unless approved.
"""
