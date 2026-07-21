# © Artur Czarnecki. All rights reserved.
# Thin Windows PowerShell launcher for the shared LKW interaction client.

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Message,

    [string]$BaseUrl = "",

    [string]$Capability = "",

    [string]$TenantId = "default",

    [string]$UserId = "windows-user",

    [string]$SessionId = "",

    [string]$InteractionId = "",

    [string]$MetadataJson = "{}",

    [int]$TimeoutSeconds = 60
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

try {
    [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false
}
catch {
}
$OutputEncoding = New-Object System.Text.UTF8Encoding $false

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ClientScript = Join-Path $ScriptDir "invoke-lkw-interaction.py"

if (-not (Test-Path -LiteralPath $ClientScript)) {
    Write-Error "Missing shared interaction client: $ClientScript"
    exit 1
}

$python = $null
if ($env:VIRTUAL_ENV) {
    $candidate = Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"
    if (Test-Path -LiteralPath $candidate) {
        $python = $candidate
    }
}
if (-not $python) {
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($null -ne $pythonCmd) {
        $python = $pythonCmd.Source
    }
}
if (-not $python) {
    $pyCmd = Get-Command py -ErrorAction SilentlyContinue
    if ($null -ne $pyCmd) {
        $python = $pyCmd.Source
    }
}
if (-not $python) {
    Write-Error "Python interpreter was not found on PATH."
    exit 1
}

$argumentList = @(
    $ClientScript,
    "--os-family", "windows",
    "--adapter-id", "lkw.windows_powershell",
    "--source", "windows_powershell",
    "--wrapper-runtime", "windows_powershell",
    "--message", $Message,
    "--base-url", $BaseUrl,
    "--capability", $Capability,
    "--tenant-id", $TenantId,
    "--user-id", $UserId,
    "--session-id", $SessionId,
    "--interaction-id", $InteractionId,
    "--metadata-json", $MetadataJson,
    "--timeout-seconds", "$TimeoutSeconds"
)

& $python @argumentList
exit $LASTEXITCODE
