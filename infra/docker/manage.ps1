# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

param (
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Tool,

    [Parameter(Mandatory = $true, Position = 1)]
    [ValidateSet("start", "stop", "status")]
    [string]$Action
)

$ComposeFile = Join-Path $PSScriptRoot "$Tool/docker-compose.yml"

if (-Not (Test-Path $ComposeFile)) {
    Write-Error "docker-compose.yml not found for tool '$Tool'"
    exit 1
}

switch ($Action) {
    "start" {
        Write-Host "Starting $Tool container..."
        docker compose -f $ComposeFile up -d
    }
    "stop" {
        Write-Host "Stopping $Tool container..."
        docker compose -f $ComposeFile down
    }
    "status" {
        Write-Host "$Tool container status:"
        docker compose -f $ComposeFile ps
    }
}