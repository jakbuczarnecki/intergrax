# © Artur Czarnecki. All rights reserved.
# Reviewer-friendly hard reset for the LKW local platform proof.

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$appDir = Resolve-Path (Join-Path $scriptDir "..")
$repoRoot = Resolve-Path (Join-Path $appDir "..\..")
$dockerDir = Resolve-Path (Join-Path $appDir "docker")
$sentryProofDir = Join-Path $dockerDir "sentry-proof"
$runAll = Join-Path $scriptDir "run-local-docker-all.bat"
$startupLog = Join-Path $dockerDir "lkw-platform-proof-startup.log"

if (-not (Test-Path $runAll)) {
    Write-Host "Missing canonical docker runner: $runAll"
    exit 1
}

Set-Location $repoRoot

Write-Host "LKW local Docker hard reset"
Write-Host "Repository root: $repoRoot"
Write-Host ""
Write-Host "This will remove Docker containers, volumes, orphans, and local Sentry proof runtime state."
Write-Host "It will not remove source files, .env, committed relay credentials, or sample documents."
Write-Host ""

Write-Host "[1/3] Stopping and removing local Docker stack with volumes..."
& $runAll down -v --remove-orphans
if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker compose down failed with exit code $LASTEXITCODE."
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "[2/3] Removing local Sentry proof runtime state..."
$runtimeFiles = @(
    "generated.env",
    "generated.env.tmp",
    ".bootstrapped"
)

foreach ($fileName in $runtimeFiles) {
    $path = Join-Path $sentryProofDir $fileName
    if (Test-Path $path) {
        Remove-Item -LiteralPath $path -Force
        Write-Host "Removed $path"
    } else {
        Write-Host "No $fileName to remove."
    }
}

if (Test-Path $startupLog) {
    Remove-Item -LiteralPath $startupLog -Force
}

Write-Host ""
Write-Host "[3/3] Launching Docker Compose startup in the background..."
Write-Host "Startup log: $startupLog"

$command = "Set-Location '$repoRoot'; & '$runAll' up -d --build *> '$startupLog'"
Start-Process -FilePath "powershell" -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $command) -WindowStyle Minimized

Write-Host ""
Write-Host "LKW local Docker hard reset complete."
Write-Host "Stack startup is continuing in the background."
Write-Host ""
Write-Host "Next step:"
Write-Host "  applications\local_workspace_application\scripts\check-lkw-platform-proof-status.bat"
Write-Host ""
Write-Host "If the status checker prints proof_status=WAIT, wait 30-60 seconds and run it again."
Write-Host "If the status checker prints proof_status=FAIL, inspect:"
Write-Host "  $startupLog"

exit 0
