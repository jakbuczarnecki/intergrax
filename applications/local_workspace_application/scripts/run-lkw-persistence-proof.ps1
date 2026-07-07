# © Artur Czarnecki. All rights reserved.
# Reviewer-friendly LKW persistent vector storage proof (non-destructive restart).

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$appDir = Resolve-Path (Join-Path $scriptDir "..")
$repoRoot = Resolve-Path (Join-Path $appDir "..\..")
$dockerDir = Resolve-Path (Join-Path $appDir "docker")
$sampleDocsDir = Join-Path $appDir "sample_docs"
$baseCompose = Join-Path $dockerDir "docker-compose.yml"

$tenantId = "lkw-persistence-proof"
$workspaceId = "lkw-persistence-proof"
$collectionId = "lkw-persistence-proof"

$baseUrl = if ([string]::IsNullOrWhiteSpace($env:LOCAL_WORKSPACE_BACKEND_BASE_URL)) {
    "http://127.0.0.1:8020"
} else {
    $env:LOCAL_WORKSPACE_BACKEND_BASE_URL.Trim()
}
$baseUrl = $baseUrl.TrimEnd("/")
$healthUrl = "$baseUrl/health"
$runUrl = "$baseUrl/v1/local_workspace/run"

$markerTimestamp = Get-Date -Format "yyyyMMddHHmmss"
$marker = "LKW_PERSISTENCE_PROOF_$markerTimestamp"
$proofFileName = "lkw_persistence_proof_$markerTimestamp.txt"
$containerSourcePath = "/data/user_docs/$proofFileName"

Set-Location $repoRoot

$composeArgs = @("-f", $baseCompose)
Get-ChildItem -Path $dockerDir -Filter "docker-compose.*.yml" | Sort-Object FullName | ForEach-Object {
    $composeArgs += @("-f", $_.FullName)
}

function Write-ProofFail {
    param(
        [string]$Phase,
        [string]$Reason
    )

    Write-Host ""
    Write-Host "proof_result=FAIL"
    Write-Host "proof_kind=persistent_vector_storage"
    Write-Host "failing_phase=$Phase"
    Write-Host "reason=$Reason"
    exit 1
}

function Wait-LkwHealth {
    param(
        [int]$TimeoutSeconds = 120
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    do {
        try {
            $response = Invoke-RestMethod -Method Get -Uri $healthUrl -TimeoutSec 5
            if ($response.status -eq "ok") {
                Write-Host "lkw_health=ok"
                return
            }
        } catch {
            Start-Sleep -Seconds 2
        }
        Start-Sleep -Seconds 2
    } while ((Get-Date) -lt $deadline)

    throw "LKW health check did not pass before timeout"
}

function Invoke-LkwRun {
    param(
        [hashtable]$Body
    )

    $json = $Body | ConvertTo-Json -Depth 8 -Compress
    return Invoke-RestMethod -Method Post -Uri $runUrl -ContentType "application/json" -Body $json
}

function Get-IndexSignalCount {
    param(
        $Response
    )

    $evidence = $Response.metadata."lkw_evidence.v1"
    if ($null -eq $evidence) {
        return 0
    }

    $indexSummary = $evidence.diagnostics."lkw.index_summary.v1"
    if ($null -eq $indexSummary) {
        return 0
    }

    foreach ($field in @("ingested_count", "chunk_count", "accepted_count")) {
        $value = $indexSummary.$field
        if ($null -ne $value) {
            $parsed = [int]$value
            if ($parsed -gt 0) {
                return $parsed
            }
        }
    }

    return 0
}

function Get-SearchResultCount {
    param(
        $Response
    )

    $evidence = $Response.metadata."lkw_evidence.v1"
    if ($null -eq $evidence) {
        return 0
    }

    $searchSummary = $evidence.diagnostics."lkw.search_summary.v1"
    if ($null -eq $searchSummary) {
        return 0
    }

    foreach ($field in @("evidence_count", "num_results")) {
        $value = $searchSummary.$field
        if ($null -ne $value) {
            $parsed = [int]$value
            if ($parsed -gt 0) {
                return $parsed
            }
        }
    }

    return 0
}

function Invoke-SearchProof {
    param(
        [string]$PhaseLabel
    )

    $body = @{
        tenant_id    = $tenantId
        workspace_id = $workspaceId
        message      = $marker
        capability   = "local.workspace.search"
        metadata     = @{
            collection_id = $collectionId
            query         = $marker
            top_k         = 5
        }
    }

    try {
        $response = Invoke-LkwRun -Body $body
    } catch {
        Write-ProofFail -Phase $PhaseLabel -Reason "search_request_failed"
    }

    $count = Get-SearchResultCount -Response $response
    if ($count -le 0) {
        Write-ProofFail -Phase $PhaseLabel -Reason "search_results_missing"
    }

    return $count
}

Write-Host "LKW persistent vector storage proof"
Write-Host "Repository root: $repoRoot"
Write-Host "LKW base URL: $baseUrl"
Write-Host "marker=$marker"
Write-Host ""

if (-not (Test-Path $sampleDocsDir)) {
    New-Item -ItemType Directory -Path $sampleDocsDir | Out-Null
}

$proofDocPath = Join-Path $sampleDocsDir $proofFileName
@"
Intergrax LKW persistence proof document.
Unique marker: $marker
This document verifies indexed local knowledge survives a non-destructive restart.
"@ | Set-Content -LiteralPath $proofDocPath -Encoding utf8 -NoNewline

Write-Host "proof_document_host_path=$proofDocPath"
Write-Host "proof_document_container_path=$containerSourcePath"
Write-Host ""

Write-Host "Phase 1/5: waiting for LKW health before indexing..."
try {
    Wait-LkwHealth
} catch {
    Write-ProofFail -Phase "health_before" -Reason "health_check_failed"
}

Write-Host ""
Write-Host "Phase 2/5: indexing proof document..."
$indexBody = @{
    tenant_id    = $tenantId
    workspace_id = $workspaceId
    message      = "index persistence proof document"
    capability   = "local.workspace.index"
    metadata     = @{
        source_paths  = @($containerSourcePath)
        collection_id = $collectionId
    }
}

try {
    $indexResponse = Invoke-LkwRun -Body $indexBody
} catch {
    Write-ProofFail -Phase "index" -Reason "index_request_failed"
}

$indexSignal = Get-IndexSignalCount -Response $indexResponse
if ($indexSignal -le 0) {
    Write-ProofFail -Phase "index" -Reason "index_not_ingested"
}

Write-Host "index_signal_count=$indexSignal"

Write-Host ""
Write-Host "Phase 3/5: searching before non-destructive restart..."
$beforeCount = Invoke-SearchProof -PhaseLabel "search_before_restart"
Write-Host "before_restart_results=$beforeCount"

Write-Host ""
Write-Host "Phase 4/5: non-destructive restart of local_workspace and qdrant..."
& docker compose @composeArgs restart local_workspace qdrant
if ($LASTEXITCODE -ne 0) {
    Write-ProofFail -Phase "restart" -Reason "docker_compose_restart_failed"
}

Write-Host "restart_mode=non_destructive"
Write-Host "volumes_removed=false"

Write-Host ""
Write-Host "Phase 5/5: waiting for LKW health after restart..."
try {
    Wait-LkwHealth
} catch {
    Write-ProofFail -Phase "health_after" -Reason "health_check_failed"
}

Write-Host ""
Write-Host "Searching again without reindexing..."
$afterCount = Invoke-SearchProof -PhaseLabel "search_after_restart"
Write-Host "after_restart_results=$afterCount"

Write-Host ""
Write-Host "proof_result=PASS"
Write-Host "proof_kind=persistent_vector_storage"
Write-Host "restart_mode=non_destructive"
Write-Host "volumes_removed=false"
Write-Host "tenant_id=$tenantId"
Write-Host "workspace_id=$workspaceId"
Write-Host "collection_id=$collectionId"
Write-Host "marker=$marker"
Write-Host "before_restart_results=$beforeCount"
Write-Host "after_restart_results=$afterCount"
Write-Host "reindexed_after_restart=false"
