# specTableContent format audit (Implementation-2A)

Bounded audit over `dataset/processed/selected_offers_sample_5000.json` (5,000 records).

## Coverage

- Records inspected: **5,000**
- Non-empty `specTableContent`: **5,000** (100% of sample)

## Observed format families

| Family | Approx. share | Parsing decision |
|--------|---------------|------------------|
| Single free-form blob (newline-separated prose, space-separated labels) | ~71% | **No structured extraction** — remains lexical/semantic only |
| Single segment with embedded colons (marketing / boilerplate text) | ~29% | **No structured extraction** — colon count does not form stable pairs |
| HTML / markup fragments | <0.2% | **No structured extraction** |
| Vertical-tab `key:` / `value` alternation | 1 record in sample | **Deterministic parser** (`\x0b` colon alternation) |
| Strict newline `Key: Value` on every line (2+ lines) | 0 in sample | **Supported when all lines match** (conservative contract) |

## Policy

Structured attributes are emitted only for the two deterministic families above.
Ambiguous content is never split heuristically into fake key/value pairs.
