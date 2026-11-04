# Optional wrapper for igx_gov (Delivery DevOS)
param(
    [Parameter(Position = 0)]
    [ValidateSet("bootstrap", "doctor", "ticket", "before-push", "lint", "inventory")]
    [string] $Command = "ticket"
)

$ErrorActionPreference = "Stop"
$Root = git rev-parse --show-toplevel
Set-Location $Root
& uv run python -m scripts.governance.igx_gov $Command @args
exit $LASTEXITCODE
