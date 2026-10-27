# © Artur Czarnecki. All rights reserved.
# Requires: gh auth login (admin on repo).

$ErrorActionPreference = "Stop"
Set-Location (git rev-parse --show-toplevel)

$remote = git remote get-url origin
if ($remote -match "github\.com[:/](?<owner>[^/]+)/(?<repo>[^/.]+)") {
    $owner = $Matches.owner
    $repo = $Matches.repo -replace '\.git$', ''
} else {
    Write-Error "Could not parse GitHub owner/repo from origin: $remote"
}

$branches = @("development", "main")
$context = "ci-smoke"

foreach ($branch in $branches) {
    Write-Host "Applying branch protection: $owner/$repo @ $branch (required: $context)"
    $payload = @{
        required_status_checks = @{
            strict   = $true
            contexts = @($context)
        }
        enforce_admins                    = $false
        required_pull_request_reviews     = $null
        restrictions                      = $null
        required_linear_history           = $false
        allow_force_pushes                = $false
        allow_deletions                   = $false
    } | ConvertTo-Json -Depth 6 -Compress

    $payload | gh api -X PUT "repos/$owner/$repo/branches/$branch/protection" --input -
}

Write-Host "Done. Verify: GitHub -> Settings -> Branches."
