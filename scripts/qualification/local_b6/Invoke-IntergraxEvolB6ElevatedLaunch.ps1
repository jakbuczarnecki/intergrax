# © Artur Czarnecki. All rights reserved.
<#
.SYNOPSIS
  Launch elevated EVOL-X B6 bootstrap via UAC (RunAs). Non-secret results -> ResultPath JSON.
#>
[CmdletBinding()]
param(
    [string] $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path,
    [string] $ResultPath = (Join-Path $RepoRoot ".tmp\session\evol-x-local-b6-unattended\elevated-result.json")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$implementer = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$bootstrap = Join-Path $PSScriptRoot "Invoke-IntergraxEvolB6UnattendedBootstrap.ps1"
$resultDir = Split-Path -Parent $ResultPath
if (-not (Test-Path $resultDir)) { New-Item -ItemType Directory -Force -Path $resultDir | Out-Null }

$launchMeta = @{
    schema_version = "evol_x_local_b6_elevated_launch.v1"
    implementer = $implementer
    elevated_attempted = $true
    timestamp_utc = (Get-Date).ToUniversalTime().ToString("o")
}

if (-not (Test-Path $bootstrap)) {
    $launchMeta.disposition = "BOOTSTRAP_SCRIPT_MISSING"
    $launchMeta | ConvertTo-Json | Set-Content $ResultPath -Encoding UTF8
    exit 2
}

$argList = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", "`"$bootstrap`"",
    "-RepoRoot", "`"$RepoRoot`"",
    "-ResultPath", "`"$ResultPath`"",
    "-ImplementerAccount", "`"$implementer`""
)

try {
    $proc = Start-Process -FilePath "powershell.exe" -Verb RunAs -ArgumentList ($argList -join " ") -Wait -PassThru -ErrorAction Stop
    $launchMeta.elevated_exit_code = $proc.ExitCode
} catch {
    $launchMeta.disposition = "UAC_ELEVATION_DENIED_OR_FAILED"
    $launchMeta.error = $_.Exception.Message
    $launchMeta | ConvertTo-Json | Set-Content (Join-Path $resultDir "elevated-launch.json") -Encoding UTF8
    if (-not (Test-Path $ResultPath)) {
        $launchMeta | ConvertTo-Json | Set-Content $ResultPath -Encoding UTF8
    }
    exit 3
}

if (Test-Path $ResultPath) {
    Set-StrictMode -Off
    $existing = Get-Content $ResultPath -Raw | ConvertFrom-Json
    $kp = $existing.key_ceremony.key_path
    Set-StrictMode -Version Latest
    if ($null -ne $kp -and "$kp".Length -gt 0) {
        $keyPath = [string]$kp
        try {
            [System.IO.File]::OpenRead($keyPath).Close()
            $launchMeta.implementer_can_read_private_key = $true
        } catch {
            $launchMeta.implementer_can_read_private_key = $false
        }
    }
    $launchMeta | ConvertTo-Json | Set-Content (Join-Path $resultDir "elevated-launch.json") -Encoding UTF8
    exit [int]$launchMeta.elevated_exit_code
}
$launchMeta.disposition = "NO_RESULT_AFTER_ELEVATION"
$launchMeta | ConvertTo-Json | Set-Content $ResultPath -Encoding UTF8
exit 4
