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

# Build argv explicitly. Never pass optional flag names without values —
# PowerShell splatting drops empty strings, which breaks Python argparse.
$argumentList = @(
    $ClientScript,
    "--os-family", "windows",
    "--adapter-id", "lkw.windows_powershell",
    "--source", "windows_powershell",
    "--wrapper-runtime", "windows_powershell",
    "--message", $Message,
    "--timeout-seconds", "$TimeoutSeconds"
)

if (-not [string]::IsNullOrWhiteSpace($BaseUrl)) {
    $argumentList += @("--base-url", $BaseUrl)
}

if (-not [string]::IsNullOrWhiteSpace($Capability)) {
    $argumentList += @("--capability", $Capability)
}

if (-not [string]::IsNullOrWhiteSpace($TenantId)) {
    $argumentList += @("--tenant-id", $TenantId)
}

if (-not [string]::IsNullOrWhiteSpace($UserId)) {
    $argumentList += @("--user-id", $UserId)
}

if (-not [string]::IsNullOrWhiteSpace($SessionId)) {
    $argumentList += @("--session-id", $SessionId)
}

if (-not [string]::IsNullOrWhiteSpace($InteractionId)) {
    $argumentList += @("--interaction-id", $InteractionId)
}

if (-not [string]::IsNullOrWhiteSpace($MetadataJson)) {
    $argumentList += @("--metadata-json", $MetadataJson)
}

# Test-only argv dump (unit tests); never used by live proof paths.
if (-not [string]::IsNullOrWhiteSpace($env:LKW_PS1_ARGV_DUMP)) {
    $jsonItems = foreach ($arg in $argumentList) {
        $escaped = ([string]$arg).Replace('\', '\\').Replace('"', '\"')
        '"' + $escaped + '"'
    }
    ("[" + ($jsonItems -join ",") + "]") |
        Out-File -FilePath $env:LKW_PS1_ARGV_DUMP -Encoding utf8
}

function ConvertTo-WindowsArgumentString {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    # Escape for CreateProcess / CommandLineToArgvW so JSON quotes survive.
    # Do not use `& native @args` — Windows PowerShell re-encodes and strips quotes.
    $parts = foreach ($arg in $Arguments) {
        $value = [string]$arg
        if ($value.Length -eq 0) {
            '""'
            continue
        }
        $needsQuotes = ($value.IndexOfAny([char[]]@(' ', "`t", '"')) -ge 0)
        if (-not $needsQuotes) {
            $value
            continue
        }
        $sb = New-Object System.Text.StringBuilder
        [void]$sb.Append('"')
        $backslashes = 0
        foreach ($ch in $value.ToCharArray()) {
            if ($ch -eq [char]'\' ) {
                $backslashes++
                continue
            }
            if ($ch -eq [char]'"') {
                if ($backslashes -gt 0) {
                    [void]$sb.Append('\', $backslashes * 2)
                    $backslashes = 0
                }
                [void]$sb.Append('\"')
                continue
            }
            if ($backslashes -gt 0) {
                [void]$sb.Append('\', $backslashes)
                $backslashes = 0
            }
            [void]$sb.Append($ch)
        }
        if ($backslashes -gt 0) {
            [void]$sb.Append('\', $backslashes * 2)
        }
        [void]$sb.Append('"')
        $sb.ToString()
    }
    return ($parts -join ' ')
}

$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $python
$psi.Arguments = ConvertTo-WindowsArgumentString -Arguments $argumentList
$psi.UseShellExecute = $false
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
$psi.RedirectStandardInput = $true
$psi.CreateNoWindow = $true
$psi.WorkingDirectory = (Get-Location).Path

$process = New-Object System.Diagnostics.Process
$process.StartInfo = $psi
[void]$process.Start()
$stdout = $process.StandardOutput.ReadToEnd()
$stderr = $process.StandardError.ReadToEnd()
$process.WaitForExit()

if (-not [string]::IsNullOrEmpty($stdout)) {
    [Console]::Out.Write($stdout)
}
if (-not [string]::IsNullOrEmpty($stderr)) {
    [Console]::Error.Write($stderr)
}

exit $process.ExitCode
