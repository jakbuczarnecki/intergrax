# IIAC verifier corpus (T-IGX-044-E, node PT-E04)

Test data for `scripts/governance/verify_iiac_manifest_corpus.py`, `verify_iiac_manifest_artifacts.py` and
`verify_iiac_delegation_refs.py`. Read by `tests/unit/governance/test_iiac_new_verifiers.py`, which builds a
throw-away git repository, substitutes the `@@TOKEN@@` placeholders with real commit SHAs and digests, and runs
every case in `cases.json`.

* `base_manifest.json.tmpl` is the one well-formed manifest; every manifest case is the base plus `set` / `delete`.
* `evidence/` holds the four artifacts the base manifest binds by digest.
* `delegations/` holds the delegation records tickets can point at.
* `cases.json` lists each case with its polarity, expected exit code and expected finding codes. Every negative
  case names the weakness it targets in `weakness` (the test fails if it is empty).

The files are fixtures: nothing here is an IIAC manifest of the repository, a certification or an approval, and the
file names deliberately avoid the `iiac_audit_session_manifest*.json` pattern that the repository guards glob.
