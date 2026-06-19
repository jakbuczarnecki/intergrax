# © Artur Czarnecki. All rights reserved.

param (
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet("start", "stop", "status", "build")]
    [string]$Action,

    [Parameter(Position = 1)]
    [ValidateSet("default", "minimal", "core", "queue", "rag", "data", "secrets", "observability", "cloud", "heavy", "vllm", "p6", "all")]
    [string]$Profile = "default"
)

$ComposeFile = Join-Path $PSScriptRoot "docker-compose.yml"

if (-Not (Test-Path $ComposeFile)) {
    Write-Error "docker-compose.yml not found in $PSScriptRoot"
    exit 1
}

function Get-ProfileFlags {
    param ([string]$Name)
    switch ($Name) {
        "all" { return @("--profile", "core", "--profile", "queue", "--profile", "rag", "--profile", "data", "--profile", "secrets", "--profile", "observability", "--profile", "cloud", "--profile", "heavy", "--profile", "vllm", "--profile", "all") }
        "default" { return @("--profile", "core", "--profile", "queue", "--profile", "rag", "--profile", "data", "--profile", "secrets") }
        "minimal" { return @("--profile", "core") }
        "p6" { return @("--profile", "core", "--profile", "p6") }
        default { return @("--profile", $Name) }
    }
}

$flags = Get-ProfileFlags -Name $Profile

switch ($Action) {
    "start" {
        Write-Host "Starting Integration stack (profile=$Profile)..."
        docker compose -f $ComposeFile @flags up -d
    }
    "stop" {
        Write-Host "Stopping Integration stack (profile=$Profile)..."
        docker compose -f $ComposeFile @flags down
    }
    "status" {
        Write-Host "Integration stack status (profile=$Profile):"
        docker compose -f $ComposeFile @flags ps -a
    }
    "build" {
        Write-Host "Building custom images (docling)..."
        docker compose -f $ComposeFile @flags build docling
    }
}
