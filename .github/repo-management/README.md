# GitHub repository management

Maintainer tooling for **repository-level GitHub settings** that should stay in sync with the codebase (description, homepage, topics).

**Source of truth:** [`repository-metadata.json`](repository-metadata.json)

---

## Layout

| File | Purpose |
|------|---------|
| `repository-metadata.json` | Public description, homepage URL, and topics (max 20) |
| `sync_github_repository_metadata.py` | Validate or apply the manifest via GitHub API / `gh` |
| `sync-github-metadata.bat` | Windows wrapper (default: apply) |
| `sync-github-metadata.sh` | Linux / macOS / Git Bash wrapper |
| [`../workflows/sync-repository-metadata.yml`](../workflows/sync-repository-metadata.yml) | CI auto-sync on push to `main` |

---

## One-time setup (local sync)

### 1. Create a fine-grained Personal Access Token

1. GitHub avatar → **Settings** → **Developer settings**
2. **Personal access tokens** → **Fine-grained tokens** → **Generate new token**
3. **Repository access:** Only select repositories → choose `intergrax`
4. **Repository permissions:**
   - **Administration:** Read and write
   - **Metadata:** Read (default)
5. Generate and copy the token (shown once)

Do **not** commit the token. Do **not** open a pull request containing it.

### 2. Store the token in `.env`

From the repository root:

```bash
cp .env.example .env   # if you do not have .env yet
```

Add to `.env` (already gitignored):

```env
GH_TOKEN=github_pat_xxxxxxxx
```

The sync script loads `.env` automatically via `python-dotenv`.

Alternative auth (no `.env`):

- `gh auth login` if GitHub CLI is installed, or
- `$env:GH_TOKEN = "..."` in PowerShell for the current session only

### 3. Verify and sync

From the repository root:

```bash
# Validate manifest only (no GitHub changes)
.github/repo-management/sync-github-metadata.bat check      # Windows
./.github/repo-management/sync-github-metadata.sh check    # Linux/macOS

# Apply to GitHub (default)
.github/repo-management/sync-github-metadata.bat           # Windows
./.github/repo-management/sync-github-metadata.sh          # Linux/macOS
```

Success output ends with:

```text
Synced metadata to jakbuczarnecki/intergrax
```

---

## CI auto-sync (no PAT required)

On push to `main` when `repository-metadata.json` or sync scripts change, workflow **Sync repository metadata** applies the manifest using `GITHUB_TOKEN`.

Ensure in the repository:

**Settings → Actions → General → Workflow permissions** → **Read and write permissions**

---

## Editing description or topics

1. Edit [`repository-metadata.json`](repository-metadata.json)
2. Run `sync-github-metadata.bat check` to validate locally
3. Run `sync-github-metadata.bat` (or merge to `main` and let CI apply)
4. Keep [`pyproject.toml`](../../pyproject.toml) `description` and `keywords` aligned for packaging consistency

GitHub limits:

- Description: 350 characters
- Topics: 20 unique tags

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `No GitHub credentials found` | Add `GH_TOKEN` to `.env` or run `gh auth login` |
| `GitHub API ... failed (403)` | Token missing **Administration: Read and write** on this repo |
| `Could not resolve target repository` | Run from a git clone with `origin` pointing at GitHub, or set `"repository": "owner/name"` in the manifest |
| CI workflow does not update settings | Enable **Read and write** workflow permissions for Actions |

---

## Security

- `.env` is gitignored — never commit tokens
- Prefer fine-grained tokens scoped to `intergrax` only
- Rotate tokens if exposed
