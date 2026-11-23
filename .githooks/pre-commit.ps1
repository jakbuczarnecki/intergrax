$ErrorActionPreference = "Stop"
$Root = git rev-parse --show-toplevel
Set-Location $Root
$uvLocal = Join-Path $env:USERPROFILE ".local\bin"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    $uvExe = Join-Path $uvLocal "uv.exe"
    if (Test-Path -LiteralPath $uvExe) {
        $env:Path = "$uvLocal;$env:Path"
    }
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "pre-commit: uv not found on PATH (install Astral uv or add %USERPROFILE%\.local\bin)"
}
& uv run --no-sync python -u scripts/ci/run_pre_commit_gate.py
exit $LASTEXITCODE
