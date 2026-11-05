# © Artur Czarnecki. All rights reserved.
#Requires -RunAsAdministrator
<#
.SYNOPSIS
  Idempotent EVOL-X local B6 custody bootstrap (administrator).

  Creates custody tree, Windows principals, ACLs, key ceremony, attestor/custodian
  pipeline smoke. Writes non-secret structured results to ResultPath.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $RepoRoot,
    [Parameter(Mandatory = $true)]
    [string] $ResultPath,
    [Parameter(Mandatory = $true)]
    [string] $ImplementerAccount,
    [string] $CustodyRoot = "C:\ProgramData\Intergrax\evol-b6-custody",
    [string] $AttestorName = "IntergraxEvolAttst",
    [string] $CustodianName = "IntergraxEvolCustn",
    [string] $AttestorIdentity = "intergrax-evol-attestor-local",
    [string] $CustodianIdentity = "intergrax-evol-custodian-local",
    [string] $KeyId = "k-prod-local-1",
    [string] $TrustRootId = "tr-intergrax-local-b6-1"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Result {
    param([hashtable] $Data)
    $dir = Split-Path -Parent $ResultPath
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
    $Data["timestamp_utc"] = (Get-Date).ToUniversalTime().ToString("o")
    $Data | ConvertTo-Json -Depth 8 | Set-Content -Path $ResultPath -Encoding UTF8
}

function New-LocalPrincipal {
    param([string] $Name)
    $existing = Get-LocalUser -Name $Name -ErrorAction SilentlyContinue
    if ($existing) {
        return @{ created = $false; enabled = [bool]$existing.Enabled }
    }
    Add-Type -AssemblyName System.Web
    $plain = [System.Web.Security.Membership]::GeneratePassword(28, 6)
    $sec = ConvertTo-SecureString $plain -AsPlainText -Force
    $plain = $null
    New-LocalUser -Name $Name -Password $sec -PasswordNeverExpires -UserMayNotChangePassword | Out-Null
    $sec = $null
    return @{ created = $true; enabled = $true }
}

function Invoke-AsLocalUser {
        param(
            [string] $UserName,
            [string] $PlainPassword,
            [string] $Repo,
            [hashtable] $EnvExtra,
            [string[]] $PythonArgs
        )
        $python = Join-Path $Repo ".venv\Scripts\python.exe"
        if (-not (Test-Path $python)) {
            throw "venv python missing at $python"
        }
        $wrapperDir = Join-Path $Repo ".tmp\session\evol-x-local-b6-unattended"
        if (-not (Test-Path $wrapperDir)) { New-Item -ItemType Directory -Force -Path $wrapperDir | Out-Null }
        $taskName = "IntergraxEvolB6_$([guid]::NewGuid().ToString('N').Substring(0,8))"
        $argList = ($PythonArgs | ForEach-Object { "'" + ($_ -replace "'", "''") + "'" }) -join ", "
        $envLines = @("`$env:PYTHONPATH='$($Repo -replace "'", "''")'")
        foreach ($k in $EnvExtra.Keys) {
            $v = $EnvExtra[$k] -replace "'", "''"
            $envLines += "`$env:$k='$v'"
        }
        $psCmd = "Set-Location '$($Repo -replace "'", "''")'; $($envLines -join '; '); & '$($python -replace "'", "''")' $argList; exit `$LASTEXITCODE"
        $action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -Command `"$psCmd`""
        Register-ScheduledTask -TaskName $taskName -Action $action -User $UserName -Password $PlainPassword -Force | Out-Null
        Start-ScheduledTask -TaskName $taskName
        $deadline = (Get-Date).AddMinutes(10)
        while (((Get-ScheduledTask -TaskName $taskName).State -eq "Running") -and ((Get-Date) -lt $deadline)) {
            Start-Sleep -Seconds 2
        }
        $exitCode = (Get-ScheduledTaskInfo -TaskName $taskName).LastTaskResult
        Unregister-ScheduledTask -TaskName $taskName -Confirm:$false | Out-Null
        return @{ exit = [int]$exitCode }
    }

$result = @{
    schema_version = "evol_x_local_b6_unattended_bootstrap.v1"
    elevated = $true
    custody_root = $CustodyRoot
    implementer_account = $ImplementerAccount
    install = @{ status = "pending" }
    principals = @{}
    acl = @{}
    key_ceremony = @{}
    attestor = @{}
    custodian = @{}
    evaluator = @{}
    acl_probes = @{}
    disposition = "IN_PROGRESS"
}

try {
    $installScript = Join-Path $RepoRoot "scripts\qualification\local_b6\Install-IntergraxEvolB6Custody.ps1"
    & $installScript
    $result.install.status = "OK"

    $att = New-LocalPrincipal -Name $AttestorName
    $cus = New-LocalPrincipal -Name $CustodianName
    $result.principals.attestor = $att
    $result.principals.custodian = $cus

    $implementerRef = $ImplementerAccount
    if ($implementerRef -notmatch '\\') {
        $implementerRef = "$env:COMPUTERNAME\$ImplementerAccount"
    }
    $result.principals.sam_account_note = "SAM<=20: $AttestorName / $CustodianName (logical IntergraxEvolAttestor / IntergraxEvolCustodian)"

    $privateDir = Join-Path $CustodyRoot "private"
    $publicDir = Join-Path $CustodyRoot "public"
    $pendingDir = Join-Path $CustodyRoot "pending"
    $signedDir = Join-Path $CustodyRoot "signed"
    $ledgerDir = Join-Path $CustodyRoot "ledger"

    foreach ($d in @($privateDir, $publicDir, $pendingDir, $signedDir, $ledgerDir)) {
        if (-not (Test-Path $d)) { New-Item -ItemType Directory -Force -Path $d | Out-Null }
    }

    function Set-CustodyIcacls {
        param(
            [string] $Path,
            [string[]] $Grant,
            [string[]] $Deny = @()
        )
        & icacls $Path /inheritance:r | Out-Null
        if ($Grant.Count -gt 0) {
            & icacls $Path /grant:r ($Grant -join " ") | Out-Null
        }
        foreach ($d in $Deny) {
            & icacls $Path /deny $d | Out-Null
        }
    }

    $impDeny = "${implementerRef}:(OI)(CI)(W,D,WDAC,WO)"
    Set-CustodyIcacls -Path $privateDir -Grant @(
        "SYSTEM:(OI)(CI)F",
        "Administrators:(OI)(CI)F",
        "${CustodianName}:(OI)(CI)F"
    ) -Deny @($impDeny)
    Set-CustodyIcacls -Path $ledgerDir -Grant @(
        "SYSTEM:(OI)(CI)F",
        "Administrators:(OI)(CI)F",
        "${CustodianName}:(OI)(CI)F"
    ) -Deny @($impDeny)
    Set-CustodyIcacls -Path $pendingDir -Grant @(
        "SYSTEM:(OI)(CI)F",
        "Administrators:(OI)(CI)F",
        "${AttestorName}:(OI)(CI)M",
        "${CustodianName}:(OI)(CI)RX"
    ) -Deny @($impDeny)
    Set-CustodyIcacls -Path $signedDir -Grant @(
        "SYSTEM:(OI)(CI)F",
        "Administrators:(OI)(CI)F",
        "${CustodianName}:(OI)(CI)M",
        "${AttestorName}:(OI)(CI)RX"
    ) -Deny @($impDeny)
    Set-CustodyIcacls -Path $publicDir -Grant @(
        "SYSTEM:(OI)(CI)F",
        "Administrators:(OI)(CI)F",
        "${AttestorName}:(OI)(CI)RX",
        "${CustodianName}:(OI)(CI)RX",
        "${implementerRef}:(OI)(CI)RX"
    )
    $result.acl.status = "CONFIGURED"

    foreach ($principal in @($AttestorName, $CustodianName)) {
        & icacls (Join-Path $RepoRoot ".venv") /grant "${principal}:(OI)(CI)RX" | Out-Null
        & icacls (Join-Path $RepoRoot "scripts") /grant "${principal}:(OI)(CI)RX" | Out-Null
        & icacls (Join-Path $RepoRoot ".tmp") /grant "${principal}:(OI)(CI)RX" | Out-Null
    }
    $result.acl.repo_execute_grants = "RX on .venv, scripts, .tmp for attestor/custodian principals"

    Add-Type -AssemblyName System.Web
    $custPwdPlain = [System.Web.Security.Membership]::GeneratePassword(28, 6)
    $attPwdPlain = [System.Web.Security.Membership]::GeneratePassword(28, 6)
    if ($cus.created) {
        Set-LocalUser -Name $CustodianName -Password (ConvertTo-SecureString $custPwdPlain -AsPlainText -Force)
    } else {
        Set-LocalUser -Name $CustodianName -Password (ConvertTo-SecureString $custPwdPlain -AsPlainText -Force)
    }
    if ($att.created) {
        Set-LocalUser -Name $AttestorName -Password (ConvertTo-SecureString $attPwdPlain -AsPlainText -Force)
    } else {
        Set-LocalUser -Name $AttestorName -Password (ConvertTo-SecureString $attPwdPlain -AsPlainText -Force)
    }
    $custSec = ConvertTo-SecureString $custPwdPlain -AsPlainText -Force
    $attSec = ConvertTo-SecureString $attPwdPlain -AsPlainText -Force

    $keyPath = Join-Path $privateDir "$KeyId.ed25519"
    $trustPath = Join-Path $publicDir "trust_root.v1.json"
    $ceremony = Invoke-AsLocalUser -UserName $CustodianName -PlainPassword $custPwdPlain -Repo $RepoRoot -EnvExtra @{
        EVOL_B6_CUSTODIAN_MODE = "1"
        EVOL_B6_CUSTODIAN_IDENTITY = $CustodianIdentity
    } -PythonArgs @(
        "scripts/qualification/local_b6/evol_x_b6_local_key_ceremony.py",
        "--private-key", $keyPath,
        "--trust-root", $trustPath,
        "--key-id", $KeyId,
        "--trust-root-id", $TrustRootId,
        "--attestor-identity", $AttestorIdentity
    )
    $result.key_ceremony.exit = $ceremony.exit
    $result.key_ceremony.key_path = $keyPath
    $result.key_ceremony.trust_root_path = $trustPath
    if ($ceremony.exit -ne 0) {
        throw "key ceremony failed exit $($ceremony.exit)"
    }

    $r2Report = Join-Path $RepoRoot ".tmp\session\evol-x-r2-final-reconciliation\run-2026-11-04\independent_verification_report.json"
    if (-not (Test-Path $r2Report)) { throw "missing R2 verification report at $r2Report" }
    $unsignedPath = Join-Path $pendingDir "unsigned.json"
    $attRun = Invoke-AsLocalUser -UserName $AttestorName -PlainPassword $attPwdPlain -Repo $RepoRoot -EnvExtra @{
        EVOL_B6_ATTESTOR_READONLY = "1"
        EVOL_B6_ATTESTOR_IDENTITY = $AttestorIdentity
    } -PythonArgs @(
        "scripts/qualification/evol_x_b6_local_attestor.py",
        "--verification-report", $r2Report,
        "--trust-root-ref", $TrustRootId,
        "--key-id", $KeyId,
        "--output", $unsignedPath,
        "--credential-class", "PRODUCTION_INDEPENDENT",
        "--formal-disposition", "ACCEPTED_AUTOMATED_B6"
    )
    $result.attestor.exit = $attRun.exit
    if ($attRun.exit -ne 0) {
        throw "attestor failed exit $($attRun.exit)"
    }

    $signedPath = Join-Path $signedDir "attestation.json"
    $ledgerPath = Join-Path $ledgerDir "replay_ledger.v1.json"
    $signRun = Invoke-AsLocalUser -UserName $CustodianName -PlainPassword $custPwdPlain -Repo $RepoRoot -EnvExtra @{
        EVOL_B6_CUSTODIAN_MODE = "1"
        EVOL_B6_CUSTODIAN_IDENTITY = $CustodianIdentity
    } -PythonArgs @(
        "scripts/qualification/evol_x_b6_custodian_sign.py",
        "--unsigned", $unsignedPath,
        "--private-key", $keyPath,
        "--replay-ledger", $ledgerPath,
        "--output", $signedPath
    )
    $result.custodian.exit = $signRun.exit
    if ($signRun.exit -ne 0) {
        throw "custodian sign failed exit $($signRun.exit)"
    }

    $evalOut = Join-Path $RepoRoot ".tmp\session\evol-x-local-b6-unattended\evaluator-result.json"
    $evalDir = Split-Path $evalOut -Parent
    if (-not (Test-Path $evalDir)) { New-Item -ItemType Directory -Force -Path $evalDir | Out-Null }
    $python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
    $env:PYTHONPATH = $RepoRoot
    $evalProc = Start-Process -FilePath $python -ArgumentList @(
        "scripts/qualification/evol_x_b6_evaluate_signed_attestation.py",
        "--attestation", $signedPath,
        "--trust-root", $trustPath,
        "--replay-ledger", $ledgerPath,
        "--output", $evalOut
    ) -WorkingDirectory $RepoRoot -Wait -PassThru -NoNewWindow
    $result.evaluator.exit = $evalProc.ExitCode
    if (Test-Path $evalOut) {
        $ev = Get-Content $evalOut -Raw | ConvertFrom-Json
        $result.evaluator.formal_acceptance_eligibility = $ev.formal_acceptance_eligibility
        $result.evaluator.blockers = @($ev.formal_acceptance_blockers)
    }

    $probePy = "import sys`ntry:`n open(r'$($keyPath -replace '\\','\\')','rb').read(1)`n sys.exit(0)`nexcept OSError:`n sys.exit(1)"
    $attProbeRun = Invoke-AsLocalUser -UserName $AttestorName -PlainPassword $attPwdPlain -Repo $RepoRoot -EnvExtra @{} -PythonArgs @(
        "-c", $probePy
    )
    $result.acl_probes.attestor_can_read_private_key = ($attProbeRun.exit -eq 0)
    $result.acl_probes.implementer_can_read_private_key = $null
    $result.acl_probes.implementer_probe_note = "deferred_to_non_elevated_launcher"

    $custPwdPlain = $null
    $attPwdPlain = $null

    if ($result.acl_probes.attestor_can_read_private_key) {
        $result.disposition = "ACL_PROBE_FAILED"
    } elseif ($result.evaluator.formal_acceptance_eligibility -eq "ELIGIBLE") {
        $result.disposition = "TECHNICAL_PIPELINE_OK_ELIGIBLE"
    } else {
        $result.disposition = "TECHNICAL_PIPELINE_OK_PENDING_AUTHORITY"
    }
    Write-Result -Data $result
    exit 0
} catch {
    $result.disposition = "FAILED"
    $result.error = $_.Exception.Message
    Write-Result -Data $result
    exit 1
}
