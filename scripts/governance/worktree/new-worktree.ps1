# Create a git worktree for a delivery ticket (Windows-first)
param(
    [Parameter(Mandatory = $true)]
    [string] $TicketId,
    [string] $BaseBranch = "development"
)

$ErrorActionPreference = "Stop"
$Root = git rev-parse --show-toplevel
Set-Location $Root

$safe = $TicketId -replace '[^A-Za-z0-9\-]', '-'
$branch = "feat/$safe"
$parent = Split-Path $Root -Parent
$name = Split-Path $Root -Leaf
$wtPath = Join-Path $parent "${name}-wt-$safe"

if (Test-Path $wtPath) {
    Write-Error "Worktree path already exists: $wtPath"
}

git fetch origin $BaseBranch 2>$null
git worktree add -B $branch $wtPath "origin/$BaseBranch"
if ($LASTEXITCODE -ne 0) {
    git worktree add -B $branch $wtPath $BaseBranch
}

Write-Host "[INFO] Worktree: $wtPath branch $branch"
Write-Host "[INFO] Next: cd $wtPath; uv sync; uv run python -m scripts.governance.igx_gov bootstrap"
