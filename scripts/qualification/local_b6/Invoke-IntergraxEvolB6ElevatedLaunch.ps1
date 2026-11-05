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

$lockFile = Join-Path $resultDir "elevated-bootstrap.lock"
if (Test-Path $lockFile) {
    @{
        schema_version = "evol_x_local_b6_elevated_launch.v1"
        disposition = "BOOTSTRAP_ALREADY_RUNNING"
        timestamp_utc = (Get-Date).ToUniversalTime().ToString("o")
    } | ConvertTo-Json | Set-Content (Join-Path $resultDir "elevated-launch.json") -Encoding UTF8
    exit 6
}
New-Item -ItemType File -Force -Path $lockFile | Out-Null

$launchMeta = @{
    schema_version = "evol_x_local_b6_elevated_launch.v1"
    implementer = $implementer
    elevated_attempted = $true
    timestamp_utc = (Get-Date).ToUniversalTime().ToString("o")
}

try {
    if (-not (Test-Path $bootstrap)) {
        $launchMeta.disposition = "BOOTSTRAP_SCRIPT_MISSING"
        $launchMeta | ConvertTo-Json | Set-Content $ResultPath -Encoding UTF8
        exit 2
    }

    $proc = Start-Process -FilePath "powershell.exe" -Verb RunAs -ArgumentList @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $bootstrap,
        "-RepoRoot", $RepoRoot,
        "-ResultPath", $ResultPath,
        "-ImplementerAccount", $implementer
    ) -PassThru -ErrorAction Stop
    $deadline = (Get-Date).AddMinutes(15)
    while (-not $proc.HasExited -and (Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 2
    }
    if (-not $proc.HasExited) {
        try { $proc.Kill() } catch { }
        $launchMeta.disposition = "UAC_APPROVAL_TIMEOUT"
        $launchMeta | ConvertTo-Json | Set-Content (Join-Path $resultDir "elevated-launch.json") -Encoding UTF8
        exit 5
    }
    $launchMeta.elevated_exit_code = $proc.ExitCode

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
} catch {
    $launchMeta.disposition = "UAC_ELEVATION_DENIED_OR_FAILED"
    $launchMeta.error = $_.Exception.Message
    $launchMeta | ConvertTo-Json | Set-Content (Join-Path $resultDir "elevated-launch.json") -Encoding UTF8
    if (-not (Test-Path $ResultPath)) {
        $launchMeta | ConvertTo-Json | Set-Content $ResultPath -Encoding UTF8
    }
    exit 3
} finally {
    Remove-Item -Path $lockFile -Force -ErrorAction SilentlyContinue
}
