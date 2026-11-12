# SFX canonicalization, digest and determinism scope (design, PROPOSED)

> **Status:** PROPOSED design draft for T-IGX-037-C (wave W1). Not code. Normative once ADR-SFX-002 is Accepted (OD-SFX-06). Implementation: SFX-P5+.
> **Reference SHA:** `eadb1c2116fe`. Every `path:line` citation is at this SHA.
> **Related:** [ADR-SFX-002](../../../../technical/adr/entries/2026-11-12/ADR-SFX-002.md) §D3 · master plan R1 §7.7 (PC-5) · [`schemas/`](schemas/) · [`SFX_METRICS.md`](SFX_METRICS.md)

## 1. Purpose

Every SFX door (Scenario Studio, CLI, MCP) must give the same SFX record the same bytes and the same digest. That single byte form is shared by:

- the Collaborative Work content digest;
- the approval binding;
- the engineering handoff (`blueprint_digest`, `D_bp`);
- the parity tests.

This document fixes the canonical form, the digest, the scope in which SFX claims determinism, and the tests that prove it.

## 2. Canonical form (rules CF-1..CF-11)

| Rule | Statement |
|------|-----------|
| **CF-1 Input** | The input is the pydantic `model_dump(mode="json")` of a validated SFX contract model. Every declared field is present, an absent optional value is `null`, and defaults are included. `exclude_none`, `exclude_unset` and `exclude_defaults` are never used. |
| **CF-2 Exclusion** | The top-level `canonical_digest` member is removed before canonicalization. Nothing else is removed. Nested references keep the `canonical_digest` of the record they reference. |
| **CF-3 JCS** | The bytes are the RFC 8785 JSON Canonicalization Scheme serialization of the CF-2 value: members sorted by the UTF-16 code units of their names, no insignificant whitespace, RFC 8785 string escaping (only `"`, `\` and control characters below U+0020 are escaped; everything else is emitted as UTF-8), UTF-8 output without a BOM. |
| **CF-4 Numbers** | SFX contracts contain only integers in the I-JSON safe range (absolute value at most 2^53 − 1). Floats are forbidden. Ratios are basis points (0..10000), money is integer minor units plus an ISO 4217 code, and decimal quantities that need a fraction are decimal strings with a pattern. This avoids the differences between ECMAScript and Python float serialization. |
| **CF-5 Timestamps** | RFC 3339, UTC, exactly six fractional digits, `Z` suffix (pattern in every schema `$defs/timestamp`). The contract boundary normalizes non-UTC input and rejects naive datetimes, as the CW contracts already do (`intergrax/contracts/collaborative_work.py:1238`, timezone-aware validators). |
| **CF-6 Strings** | Doors normalize every string to Unicode NFC at ingress, before validation. JCS itself does not normalize. Lone surrogates are rejected. Untrusted free text (`*_untrusted`) gets NFC only; it is never otherwise rewritten. |
| **CF-7 Arrays** | Order is significant. Each array field is declared either **ordered** (for example discovery `turns` by `turn_index`) or a **set**, which is sorted by its declared key before dumping and rejects duplicates. The schema `description` names the sort key. |
| **CF-8 Digest** | `canonical_digest = "sha256:" + lowercase_hex(SHA-256(CF-3 bytes))`, in the platform digest format (`intergrax/contracts/validation.py:13`, `:24`). |
| **CF-9 Storage** | The CW content body of an SFX artifact version is exactly the CF-3 bytes, with media type `application/vnd.intergrax.sfx.<name>.v1+json`. `ArtifactContentStore.put` hashes the body (`intergrax/collaborative_work/content_storage.py:185`), so `ArtifactContentRef.integrity_digest` (`intergrax/contracts/collaborative_work.py:1221`) equals `canonical_digest`. Readers recompute it, and a mismatch fails closed (`sfx.canonical.digest_mismatch`). |
| **CF-10 Presentation** | YAML, pretty JSON and Studio forms are presentations. They are parsed into the model and pass through CF-1..CF-8. Presentation bytes are never hashed. |
| **CF-11 References** | A cross-record reference (`artifact_version_ref`) carries the referenced `canonical_digest`. It resolves only when the stored version's recomputed digest matches. |

**Not interchangeable with platform fingerprints.** Existing platform fingerprints use `json.dumps(sort_keys=True, separators=(",", ":"))` (`intergrax/collaborative_work/repository.py:216`). That output differs from JCS in non-ASCII escaping (`ensure_ascii`), in number formatting and in key ordering for non-BMP names. SFX digests and platform fingerprints are separate families and are never compared.

**Generated package tree digest.** `generated_tree_digest` (SFX-CON-15) is the CF-8 digest of a JCS array of `[relative_posix_path, file_digest]` pairs:

- the array is sorted by path;
- `file_digest` is the `sha256:` digest of the file bytes;
- generated text files use LF line endings and contain no timestamps;
- paths are relative to the scenario workspace root.

## 3. Determinism scope

| Stage | Class | Determinism claim | Comparison method |
|-------|-------|-------------------|-------------------|
| Validate (schema + service rules) | Deterministic | Same input and code SHA give the same verdict and the same `sfx.error` code and pointer | Byte equality of the error payload |
| Resolve (closed-world capability resolution) | Deterministic | Same specification, inventory digest and SHA give the same `CapabilityResolutionReport` bytes | Canonical byte equality |
| Compose (blueprint, integration plan, test plan from resolved inputs) | Deterministic | Same inputs and SHA give the same bytes | Canonical byte equality |
| Canonicalize / digest | Deterministic | CF-1..CF-11 | Test vectors §5 |
| Scaffold (`DETERMINISTIC_SCAFFOLD`) | Deterministic | Same blueprint digest and SHA give the same `generated_tree_digest` | Tree digest equality |
| Discovery interview, specification drafting, blueprint suggestion (LLM) | Non-deterministic | **No determinism claim for live LLM output** | Replay: recorded turns (`turn_recording_ref`) are replayed through a replay provider. Equivalence means the same canonical output from the same recorded turns. |
| Agentic engineering (`AGENTIC_ENGINEERING`, SEF-X) | Non-deterministic | No regeneration claim | HO-1 lineage: `DeliveryManifest` artifact digests are reproducible from the delivered bytes |

**Controlled inputs** in deterministic stages: the clock, ID generation, random seeds, locale and time zone, directory-listing order (always sorted), line endings, and environment variables. Each is injected or fixed. A deterministic stage that reads any other ambient value fails its parity test.

## 4. Parity tests PC-1..PC-6

| ID | Rule | Test design | Pass condition |
|----|------|-------------|----------------|
| PC-1 Schemas | One schema set generated from the Tier-2 pydantic contracts. Studio client and MCP tool schemas are generated from the same source. | Regenerate the schemas and the client/tool schemas, then diff them against the committed files. | Empty diff; every `$id` matches `sfx.<name>.v1`. |
| PC-2 Validation | All doors call the same Tier-2 service functions; doors only parse transport. | Feed a corpus of invalid inputs (one per validation rule) to the Studio API, the CLI and MCP. | Identical `sfx.error` code and JSON pointer in all three doors. |
| PC-3 Permissions | One authorization path; the offline CLI is L0/L1 on local files only. | Matrix test: authority level × door × operation, including shared-store writes and L2 runs from the offline CLI. | Allowed cells succeed; every other cell is denied with the same code. |
| PC-4 Semantics | Same `artifact_id`, `version`, `canonical_digest`, transitions and side effects for the same operation. | Run each operation in the service catalog through all doors with an injected clock and IDs. | Identical lifecycle stream and identical stored bytes. |
| PC-5 Canonicalization | CF-1..CF-11. | Run test vectors §5 through each door and through an independent JCS implementation. | Byte identity of the canonical form and identical digests; replay equivalence for LLM stages. |
| PC-6 Versioning | API v1 = MCP tool set v1 = CLI v1; 1:1 operation map; one version constant. | Compare the operation maps and version constants. | Complete 1:1 map; a single constant. |

## 5. Test vectors (described in prose; expected digests are fixed at SFX-P5)

The expected bytes and digests are produced at SFX-P5. Two independent JCS implementations must agree on them before they are committed as fixtures. This document fixes only what each vector must prove.

- **TV-01 Key order.** A record whose top-level and nested members are given in reverse alphabetical order canonicalizes to members sorted by UTF-16 code units. Uppercase sorts before lowercase, and a key containing a non-BMP character sorts by its surrogate pair, not by its code point.
- **TV-02 Non-ASCII text.** A problem statement in Polish with diacritics and one emoji. The canonical bytes contain the characters as raw UTF-8, not `\u` escapes. The digest differs from the `json.dumps(sort_keys=True)` digest of the same value, which demonstrates that the two families are separate.
- **TV-03 NFC.** The same statement given once in NFC and once in NFD gives the same canonical bytes after ingress normalization.
- **TV-04 Control characters.** A string with a tab, a newline and U+001F is escaped as `\t`, `\n` and `\u001f` (lowercase hex). A string with `/` is not escaped.
- **TV-05 Float rejection.** A record with `confidence_bp: 87.5` is rejected at the contract boundary. The same value as `8750` is accepted.
- **TV-06 Integer bounds.** 2^53 − 1 is accepted and 2^53 is rejected.
- **TV-07 Timestamp normalization.** `2026-11-12T13:00:00+01:00` and `2026-11-12T12:00:00Z` both canonicalize to `2026-11-12T12:00:00.000000Z`. A naive timestamp is rejected.
- **TV-08 Digest exclusion.** The same record with any value in `canonical_digest`, or with the field missing, gives the same canonical bytes. The recomputed digest equals the CW `integrity_digest` of the stored body.
- **TV-09 Null versus absent.** An optional field omitted on input appears as `null` in the canonical bytes. Two doors, one sending `null` and one omitting the field, give identical bytes.
- **TV-10 Set arrays.** The `capability_bindings` set given in two different orders gives identical bytes. A duplicate `binding_id` is rejected.
- **TV-11 Presentation round trip.** A blueprint rendered as YAML and as pretty JSON, then parsed back, gives the original canonical bytes and digest.
- **TV-12 Secret references.** An integration setup plan with a settings key `api_key` is rejected by the schema. The same credential as a `secret_refs` entry is accepted, and the canonical bytes contain only the reference string.
- **TV-13 Tree digest.** The same generated workspace written with CRLF and with LF line endings gives a different digest. The scaffold must emit LF, and the CRLF variant fails the scaffold determinism test.
- **TV-14 Cross-door identity.** The same `BusinessProblemStatement` submitted through Studio, CLI and MCP, with an injected clock and IDs, gives byte-identical stored bodies.
