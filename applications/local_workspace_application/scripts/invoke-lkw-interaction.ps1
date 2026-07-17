# © Artur Czarnecki. All rights reserved.
# LKW Windows PowerShell interaction adapter (LKW.6C).
# Thin localhost client for POST /v1/interactions/intake (lab_json payload).

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

$script:AdapterSchemaVersion = "local_workspace.windows_interaction_adapter_result.v1"
$script:AdapterId = "lkw.windows_powershell"
$script:IntakeEndpoint = "/v1/interactions/intake"

function Write-AdapterFailure {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ErrorId,

        [int]$ExitCode,

        [object]$HttpStatus = $null
    )

    $payload = [ordered]@{
        schema_version = $script:AdapterSchemaVersion
        adapter_id     = $script:AdapterId
        result         = "FAIL"
        error_id       = $ErrorId
    }
    if ($null -ne $HttpStatus) {
        $payload["http_status"] = [int]$HttpStatus
    }
    $json = ($payload | ConvertTo-Json -Compress -Depth 6)
    [Console]::Error.WriteLine($json)
    exit $ExitCode
}

function Resolve-BaseUrl {
    param([string]$RawBaseUrl)

    $resolved = $RawBaseUrl
    if ([string]::IsNullOrWhiteSpace($resolved)) {
        $resolved = [string]$env:LOCAL_WORKSPACE_BACKEND_BASE_URL
    }
    if ([string]::IsNullOrWhiteSpace($resolved)) {
        $resolved = "http://127.0.0.1:8020"
    }
    $resolved = $resolved.Trim()
    while ($resolved.EndsWith("/")) {
        $resolved = $resolved.Substring(0, $resolved.Length - 1)
    }
    return $resolved
}

if ([string]::IsNullOrWhiteSpace($Message)) {
    Write-AdapterFailure -ErrorId "invalid_adapter_input" -ExitCode 2
}
if ([string]::IsNullOrWhiteSpace($TenantId)) {
    Write-AdapterFailure -ErrorId "invalid_adapter_input" -ExitCode 2
}
if ([string]::IsNullOrWhiteSpace($UserId)) {
    Write-AdapterFailure -ErrorId "invalid_adapter_input" -ExitCode 2
}
if ($TimeoutSeconds -le 0) {
    Write-AdapterFailure -ErrorId "invalid_adapter_input" -ExitCode 2
}

$metadataObject = $null
try {
    if ([string]::IsNullOrWhiteSpace($MetadataJson)) {
        $MetadataJson = "{}"
    }
    $metadataObject = $MetadataJson | ConvertFrom-Json
}
catch {
    Write-AdapterFailure -ErrorId "invalid_adapter_input" -ExitCode 2
}
if ($null -eq $metadataObject -or $metadataObject -is [System.Array] -or $metadataObject -is [string] -or $metadataObject -is [ValueType]) {
    Write-AdapterFailure -ErrorId "invalid_adapter_input" -ExitCode 2
}

$resolvedBaseUrl = Resolve-BaseUrl -RawBaseUrl $BaseUrl
$encodedTenant = [System.Uri]::EscapeDataString($TenantId.Trim())
$uri = "{0}{1}?execute=true&tenant={2}" -f $resolvedBaseUrl, $script:IntakeEndpoint, $encodedTenant

$body = [ordered]@{
    tenant_id = $TenantId.Trim()
    user_id   = $UserId.Trim()
    message   = $Message
    source    = "windows_powershell"
    metadata  = $metadataObject
}
if (-not [string]::IsNullOrWhiteSpace($Capability)) {
    $body["capability"] = $Capability.Trim()
}
if (-not [string]::IsNullOrWhiteSpace($SessionId)) {
    $body["session_id"] = $SessionId.Trim()
}
if (-not [string]::IsNullOrWhiteSpace($InteractionId)) {
    $body["interaction_id"] = $InteractionId.Trim()
}

$bodyJson = $body | ConvertTo-Json -Compress -Depth 32

try {
    $response = Invoke-RestMethod `
        -Method Post `
        -Uri $uri `
        -ContentType "application/json; charset=utf-8" `
        -Headers @{ Accept = "application/json" } `
        -Body $bodyJson `
        -TimeoutSec $TimeoutSeconds
}
catch {
    $statusCode = $null
    try {
        if ($null -ne $_.Exception.Response -and $null -ne $_.Exception.Response.StatusCode) {
            $statusCode = [int]$_.Exception.Response.StatusCode
        }
    }
    catch {
        $statusCode = $null
    }
    Write-AdapterFailure -ErrorId "interaction_request_failed" -ExitCode 3 -HttpStatus $statusCode
}

$result = [ordered]@{
    schema_version = $script:AdapterSchemaVersion
    adapter_id     = $script:AdapterId
    endpoint       = $script:IntakeEndpoint
    execute        = $true
    response       = $response
}
Write-Output ($result | ConvertTo-Json -Compress -Depth 32)
exit 0
