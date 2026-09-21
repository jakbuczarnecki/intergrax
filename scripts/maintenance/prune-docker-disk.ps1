# © Artur Czarnecki. All rights reserved.
# Reclaim disk space from Docker volumes, containers, and (optionally) images/build cache.

param(
    [ValidateSet("volumes", "all")]
    [string] $Level = "volumes",

    [switch] $Force,
    [switch] $WhatIf,
    [switch] $KeepRunning
)

$ErrorActionPreference = "Stop"

function Assert-Docker {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Write-Host "Docker CLI not found on PATH. Install Docker Desktop or add docker to PATH."
        exit 1
    }
    $null = docker version 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Docker daemon is not reachable. Start Docker Desktop and retry."
        exit 1
    }
}

function Show-DockerDiskUsage {
    Write-Host ""
    Write-Host "=== docker system df ==="
    docker system df
    Write-Host ""
}

function Invoke-DockerStep {
    param(
        [string] $Label,
        [string[]] $Arguments
    )
    $cmd = "docker $($Arguments -join ' ')"
    Write-Host $Label
    Write-Host "  $cmd"
    if ($WhatIf) {
        Write-Host "  (WhatIf - skipped)"
        return
    }
    & docker @Arguments
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Command failed with exit code $LASTEXITCODE."
        exit $LASTEXITCODE
    }
}

Assert-Docker

Write-Host "Intergrax - Docker disk cleanup"
Write-Host "Level: $Level"
if ($WhatIf) { Write-Host "Mode: WhatIf (no changes)" }
Write-Host ""

Show-DockerDiskUsage

$summary = switch ($Level) {
    "volumes" {
@"

Will stop all running containers (unless -KeepRunning), remove stopped containers,
then prune unused Docker volumes. Images and build cache are kept.

"@
    }
    "all" {
@"

Will stop all running containers (unless -KeepRunning), then run:
  docker system prune -a --volumes -f

This removes unused containers, networks, images, volumes, and build cache.
Images not referenced by any container will be deleted.

"@
    }
}

Write-Host $summary.Trim()

if (-not $Force -and -not $WhatIf) {
    $answer = Read-Host "Continue? [y/N]"
    if ($answer -notmatch '^[yY]') {
        Write-Host "Cancelled."
        exit 0
    }
}

if (-not $KeepRunning) {
    $running = @(docker ps -q 2>$null)
    if ($running.Count -gt 0) {
        Invoke-DockerStep "Stopping running containers..." (@("stop") + $running)
    } else {
        Write-Host "No running containers."
    }
}

switch ($Level) {
    "volumes" {
        Invoke-DockerStep "Removing stopped containers..." @("container", "prune", "-f")
        Invoke-DockerStep "Pruning unused volumes..." @("volume", "prune", "-f")
    }
    "all" {
        Invoke-DockerStep "Pruning unused Docker data (images, volumes, cache)..." @(
            "system", "prune", "-a", "--volumes", "-f"
        )
    }
}

Show-DockerDiskUsage

Write-Host "Done."
if ($Level -eq "volumes") {
    Write-Host "For deeper cleanup (unused images + build cache), run:"
    Write-Host "  scripts\maintenance\prune-docker-disk.bat -Level all"
}
Write-Host ""

exit 0
