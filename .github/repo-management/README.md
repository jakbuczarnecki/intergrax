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

## CI auto-sync

On push to `main` when `repository-metadata.json` or sync scripts change, workflow **Sync repository metadata** applies the manifest using a repository secret.

### One-time CI setup

1. Create the same fine-grained PAT as in [One-time setup (local sync)](#1-create-a-fine-grained-personal-access-token) (**Administration: Read and write** on `intergrax`).
2. Repository **Settings → Secrets and variables → Actions → New repository secret**
3. Name: `REPO_METADATA_TOKEN`
4. Value: the PAT (do not reuse a committed or logged token)

`GITHUB_TOKEN` cannot update repository description or topics - GitHub Actions workflow permissions do not include an `administration` scope.

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

## Manual repository feature settings

The metadata sync workflow updates **only** description, homepage, and topics. Repository **feature toggles** and the Social Preview image are **manual** - set in GitHub **Settings → General** by a maintainer, not by CI.

| Category | Settings |
|----------|----------|
| **Auto-synced** | Description, homepage, topics |
| **Manual** | Social preview image ([below](#social-preview)), Wiki, Projects, Discussions, Pages |

### Approved public state

| Feature | Target | Reason |
|---------|--------|--------|
| Issues | ON | Canonical public feedback / bug / scenario / partner intake |
| Projects | OFF | No intentionally curated public project board yet |
| Wiki | OFF | Canonical documentation lives in version-controlled repo docs |
| Discussions | OFF | Enable only when real community discussion volume justifies it |
| Pages | OFF | No dedicated public landing/docs site yet |

---

## Release checklist (manual GitHub settings)

After the relevant release reaches `main`:

1. Repository **Settings → General**
2. Disable **Wiki**
3. Disable **Projects**
4. Confirm **Issues** remains enabled
5. Confirm **Discussions** remains disabled
6. Confirm **Pages** remains disabled
7. Upload canonical Social Preview if not already applied - see [Social preview](#social-preview)
8. Verify About description/homepage/topics after metadata CI completes

---

## Social preview

**Manual** repository setting - not applied by the metadata sync workflow. See [Manual repository feature settings](#manual-repository-feature-settings) for the auto-sync vs manual boundary.

**Canonical source asset (version-controlled):**

`docs/project/assets/public/github/intergrax-social-preview.png`

The PNG in the repository is the canonical source for the social preview graphic. The existing metadata synchronization workflow (`sync_github_repository_metadata.py`, CI on `main`) updates description, homepage, and topics only - it does **not** upload or change the Social Preview image.

**Manual upload (maintainer):**

1. Repository **Settings → General**
2. **Social preview → Edit**
3. Upload `docs/project/assets/public/github/intergrax-social-preview.png`

Apply this step after the asset is released on `main`; do not assume CI will set it.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `No GitHub credentials found` | Add `GH_TOKEN` to `.env` or run `gh auth login` |
| `GitHub API ... failed (403)` | Token missing **Administration: Read and write** on this repo |
| `Could not resolve target repository` | Run from a git clone with `origin` pointing at GitHub, or set `"repository": "owner/name"` in the manifest |
| CI workflow does not update settings | Add `REPO_METADATA_TOKEN` repository secret (fine-grained PAT with **Administration: Read and write**) |

---

## Security

- `.env` is gitignored - never commit tokens
- Prefer fine-grained tokens scoped to `intergrax` only
- Rotate tokens if exposed
