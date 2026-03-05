# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

param (
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Tool,

    [Parameter(Mandatory = $true, Position = 1)]
    [ValidateSet("start", "stop", "status", "build")]
    [string]$Action
)

function Build-Tool {
    param (
        [Parameter(Mandatory = $true)]
        [string]$ToolName
    )

    $ToolDir = Join-Path $PSScriptRoot $ToolName
    $ComposeFile = Join-Path $ToolDir "docker-compose.yml"
    $Dockerfile = Join-Path $ToolDir "Dockerfile"

    if (-Not (Test-Path $Dockerfile)) {
        Write-Host "Skipping $ToolName (no Dockerfile)"
        return
    }

    if (-Not (Test-Path $ComposeFile)) {
        Write-Host "Skipping $ToolName (no docker-compose.yml)"
        return
    }

    Write-Host "Building $ToolName..."
    docker compose -f $ComposeFile build
}

switch ($Action) {

    "build" {

        if ($Tool -eq "all") {

            $Dirs = Get-ChildItem -Path $PSScriptRoot -Directory

            foreach ($Dir in $Dirs) {
                Build-Tool -ToolName $Dir.Name
            }

            exit 0
        }

        Build-Tool -ToolName $Tool
        exit 0
    }

    default {

        if ($Tool -eq "all") {
            Write-Error "Tool 'all' is only supported for action 'build'."
            exit 1
        }

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
    }
}