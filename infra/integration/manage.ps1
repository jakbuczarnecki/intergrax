# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

param (
    [Parameter(Mandatory = $true)]
    [ValidateSet("start", "stop", "status")]
    [string]$Action
)

$ComposeFile = Join-Path $PSScriptRoot "docker-compose.yml"

if (-Not (Test-Path $ComposeFile)) {
    Write-Error "docker-compose.yml not found in $PSScriptRoot"
    exit 1
}

switch ($Action) {
    "start" {
        Write-Host "Starting Integration stack..."
        docker compose -f $ComposeFile up -d
    }
    "stop" {
        Write-Host "Stopping Integration stack..."
        docker compose -f $ComposeFile down
    }
    "status" {
        Write-Host "Integration stack status:"
        docker compose -f $ComposeFile ps
    }
}