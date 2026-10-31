# EBH-7 — Enterprise architecture family coverage (remediation)

| Family | Owner gate | Remediation task |
| --- | --- | --- |
| Tier boundaries | `check_agents_no_tier3_imports` | EBH7-W4-FAM-TIER |
| Vendor / RPL | EBH-5 + normative appendix | EBH7-W2-RPL-* |
| Tenant isolation | TEN/XDM appendix | EBH7-W3-* |
| Governance / Execution | W4 dup/gov gates | EBH7-W4-FAM-GOV |
| PROHIBITED hardcoding | EBH-6 register | EBH7-W4-C3 |
| Trace / replay | XDM composed-proxy rows | EBH7-W3-XDM-* |

Full family catalog cross-reference: EBH-6 `_family_catalog.py` @ replay baseline.
