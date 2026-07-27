# Web Content Capture

**Status:** platform capability — `LKW-WORKSPACE-CONTENTS-1B-5-1` (`IMPLEMENTED / CORRECTION REQUIRED`; review gate: audit `LKW-WORKSPACE-CONTENTS-1B-5-1-C1`)

## Purpose

`WebContentCapture` is the provider-neutral platform capability for securely fetching and extracting **public HTTPS web pages**. It is the prerequisite for:

- LKW `WEB_URL` Knowledge Intake (`1B-5-2` … `1B-5-4`)
- `websearch.read_url` and `websearch.fetch_batch` cutover
- optional future browser automation backends (Playwright, Firecrawl, Browserbase, Apify)

This capability does **not** implement LKW endpoints, Slack intake, indexing, or queue/worker integration.

## Package

```text
intergrax/websearch/capture/
  contracts.py      — WebContentCapture port, request/result, errors
  url_policy.py     — WebUrlAccessPolicy (canonicalization, DNS, SSRF)
  http_transport.py — pinned HTTPS transport
  service.py        — SecureHttpWebContentCapture
```

Public import:

```python
from intergrax.websearch.capture import (
    CapturedWebContent,
    SecureHttpWebContentCapture,
    WebContentCapture,
    WebContentCaptureError,
    WebContentCaptureRequest,
    WebUrlAccessPolicy,
)
```

## Architecture boundary

```text
WebContentCaptureRequest
        │
        ▼
WebUrlAccessPolicy
        ├── syntax and canonicalization
        ├── scheme and port policy
        ├── host policy
        ├── DNS resolution
        └── public IP validation
        │
        ▼
approved hostname + approved IP addresses
        │
        ▼
Pinned HTTPS transport
        ├── connect to approved IP
        ├── TLS SNI = original hostname
        ├── certificate validation = original hostname
        ├── Host header = original hostname
        └── streamed bounded response
        │
        ▼
manual redirect handling (each target revalidated)
        │
        ▼
content type and size validation
        │
        ▼
existing HTML extraction (extract_basic / extract_advanced)
        │
        ▼
CapturedWebContent
```

Policy, DNS, transport, and extraction are separate layers. Transport never receives an unvalidated URL string alone.

## Default HTTPS-only policy (v1)

| Rule | Default |
|------|---------|
| Schemes | `https` only |
| Ports | `443` only |
| Userinfo | forbidden |
| IP literals in URL | forbidden |
| Fragments | removed before identity |
| Query | preserved in private fingerprint; never in safe display URL |
| Host allowlist | optional exact host matches; empty = any valid public hostname |
| Redirects | manual; default max 5; every target fully revalidated |
| HTTP downgrade | forbidden |
| Cookies / auth headers / proxy | not supported |
| Environment proxies | disabled |

## SSRF protection

For each hostname:

1. Resolve all IPv4 and IPv6 addresses (injectable resolver for tests).
2. Deduplicate and sort deterministically.
3. Reject empty results.
4. Reject if **any** address is not globally routable (`is_global` plus explicit private/loopback/link-local/multicast/reserved checks).
5. Reject mixed public + private result sets.

Blocked hostnames include `localhost`, `*.localhost`, `*.local`, `*.internal`.

If policy or DNS rejects a target, **transport call count must remain 0**.

## Pinned IP connection

Transport receives:

- approved connect IP
- original hostname
- request target (path + query)

The socket connects to the approved IP. For HTTPS:

- TLS SNI = original hostname
- certificate validation = original hostname
- `Host` header = original hostname (with non-default port when configured)
- `Accept-Encoding: identity`

### HTTP/1.1 response framing

Supported body framing modes:

- `Content-Length` — exact byte count; read stops immediately after declared length
- `Transfer-Encoding: chunked` — hex chunk sizes, optional chunk extensions, CRLF boundaries, zero chunk, bounded trailer block
- close-delimited — when neither `Content-Length` nor `Transfer-Encoding` is present, read until connection close under the global size limit

Rejected framing:

- conflicting duplicate `Content-Length` values
- `Transfer-Encoding` combined with `Content-Length`
- multiple `Location` headers
- malformed header lines / obs-fold
- unsupported `Transfer-Encoding` codings (for example `gzip`)
- premature EOF before complete `Content-Length` body

Body is read in bounded chunks under the global response size limit.

### Global deadline

One monotonic deadline covers the entire capture operation:

- URL canonicalization
- DNS resolution (`approve_target`)
- connect attempts across approved IPs
- TLS handshake
- header and body reads (including slowloris protection — partial reads do not reset the full timeout)
- redirects (same deadline, not a fresh timeout window)
- decode, extraction and normalization

DNS or extraction exceeding remaining time → `web_url_timeout` (`retryable=true`); transport call count remains `0` when DNS times out before any connect.

### Content-Encoding

Accepted final response encodings:

- absent `Content-Encoding`
- `Content-Encoding: identity`

Rejected (no decompression in v1):

- `gzip`, `br`, `deflate`, multiple encodings, or any other value → `web_url_content_encoding_unsupported`

Compressed bytes must not be indexed as text.

## Redirects

Automatic `follow_redirects` is **not** used. Supported statuses: `301`, `302`, `303`, `307`, `308`.

Each redirect:

1. reads a single `Location`
2. resolves relative URLs with `urljoin`
3. re-runs full URL policy and DNS approval
4. performs the next pinned request

Redirect to private, localhost, HTTP, or forbidden port → `web_url_redirect_target_blocked` without a transport call for the blocked target.

## Size limits (request defaults)

| Field | Default | Range |
|-------|---------|-------|
| `timeout_seconds` | 20 | 5–60 |
| `max_redirects` | 5 | 0–10 |
| `max_response_bytes` | 5 MiB | 1 KiB – 20 MiB |
| `max_extracted_chars` | 2 MiB | 1 KiB – 5 MiB |

## Content types (v1)

Accepted MIME types (parameters ignored):

- `text/html`
- `application/xhtml+xml`
- `text/plain`

Missing `Content-Type` → `web_url_content_type_unsupported` (no guessing from URL).

## Safe result contract

`CapturedWebContent` exposes:

- `safe_display_url` (scheme + host + optional shortened path; no query/fragment/userinfo)
- URL fingerprints (`sha256:` over canonical private URL)
- normalized `title`, `text`, `content_type`, `content_hash`
- operational metadata (`status_code`, `redirect_count`, `content_bytes`, `text_chars`, `capture_mode`, `extraction_method`, `fetched_at`)

It does **not** expose raw URL, query string, fragment, IP, response headers, cookies, redirect chain, raw HTML, or transport exception text.

Errors use stable `WebContentCaptureErrorCode` values; `str(error)` returns only the code.

## Extraction

HTML uses existing `intergrax.websearch.fetcher.extractor`:

- `extract_basic` — title and baseline text
- `extract_advanced` — trafilatura with BeautifulSoup fallback

Plain text bypasses the HTML parser.

## Legacy research fetch (do not use for durable ingestion)

`intergrax.websearch.fetcher.http_fetcher.fetch_page` remains for legacy research paths. It uses automatic redirects, no private-network blocking, no pinned IP transport, no response size cap, and returns headers in metadata. **Do not use it as the durable ingestion backend.**

## Future backends

Browser automation and vendor fetch providers (BrowserAutomation contract, Firecrawl, Browserbase, Apify) may implement `WebContentCapture` or sit behind it, but must:

- reuse `WebUrlAccessPolicy` (or equivalent policy port)
- return `CapturedWebContent` (or map to it)
- preserve the safe result and error contracts

## Related LKW tasks

| Task | Scope |
|------|-------|
| `1B-5-1` | Shared secure Web Content Capture contract and HTTPS backend (this document) |
| `1B-5-2` | LKW `WEB_URL` intake and private Source locator |
| `1B-5-3` | Web URL ingestion, indexing and Ask proof |
| `1B-5-4` | Slack explicit Web URL intake |
