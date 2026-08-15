# © Artur Czarnecki. All rights reserved.
# Reviewer-friendly LKW platform proof readiness checker.

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..\..\..")
$dockerDir = Resolve-Path (Join-Path $scriptDir "..\docker")
$baseCompose = Join-Path $dockerDir "docker-compose.yml"
$composeProject = "lkw-core-platform-proof"

Set-Location $repoRoot

$composeArgs = @("-p", $composeProject, "-f", $baseCompose)
Get-ChildItem -Path $dockerDir -Filter "docker-compose.*.yml" | Sort-Object FullName | ForEach-Object {
    $composeArgs += @("-f", $_.FullName)
}

$requiredUp = @(
    "local_workspace",
    "elasticsearch",
    "kibana",
    "lkw-redis",
    "lkw-kafka",
    "lkw-kafka-ui",
    "lkw-background-worker",
    "sentry-web",
    "sentry-relay",
    "sentry-nginx",
    "sentry-events-consumer"
)

$requiredExitedZero = @(
    "sentry-bootstrap",
    "sentry-upgrade",
    "sentry-snuba-bootstrap",
    "sentry-kafka-topics",
    "lkw-kafka-topics"
)

$rows = & docker compose @composeArgs ps -a --format "{{.Service}}|{{.Status}}"
$statusByService = @{}
foreach ($row in $rows) {
    $parts = $row -split "\|", 2
    if ($parts.Count -eq 2) {
        $statusByService[$parts[0]] = $parts[1]
    }
}

$failures = 0
$waiting = 0

Write-Host "LKW platform proof status check"
Write-Host "Repository root: $repoRoot"
Write-Host ""

foreach ($service in $requiredUp) {
    $status = $statusByService[$service]
    if ([string]::IsNullOrWhiteSpace($status)) {
        Write-Host "[WAIT] $service missing"
        $waiting += 1
    } elseif ($status -match "^Up") {
        Write-Host "[ OK ] $service $status"
    } elseif ($status -match "Restarting|Created") {
        Write-Host "[WAIT] $service $status"
        $waiting += 1
    } else {
        Write-Host "[FAIL] $service $status"
        $failures += 1
    }
}

foreach ($service in $requiredExitedZero) {
    $status = $statusByService[$service]
    if ([string]::IsNullOrWhiteSpace($status)) {
        Write-Host "[WAIT] $service missing"
        $waiting += 1
    } elseif ($status -match "Exited \(0\)") {
        Write-Host "[ OK ] $service $status"
    } elseif ($status -match "^Up|Created|Restarting") {
        Write-Host "[WAIT] $service $status"
        $waiting += 1
    } else {
        Write-Host "[FAIL] $service $status"
        $failures += 1
    }
}

Write-Host ""
if ($failures -eq 0 -and $waiting -eq 0) {
    Write-Host "proof_status=PASS"
    Write-Host "next_step=run-sentry-observability-proof"
    exit 0
}

if ($failures -gt 0) {
    Write-Host "proof_status=FAIL"
    Write-Host "failed_checks=$failures"
    Write-Host "waiting_checks=$waiting"
    Write-Host ""
    Write-Host "Inspect details with:"
    Write-Host "  applications\local_workspace_application\scripts\run-local-docker-all.bat ps -a"
    exit 1
}

Write-Host "proof_status=WAIT"
Write-Host "waiting_checks=$waiting"
Write-Host ""
Write-Host "Wait 30-60 seconds and run this status checker again."
exit 2
