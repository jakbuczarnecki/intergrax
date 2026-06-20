# © Artur Czarnecki. All rights reserved.
# Start llama.cpp stack (if needed), wait for health, run local-only E2E tests.

$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
$ComposeChat = Join-Path $Root "infra\docker\llama-cpp\docker-compose.yml"
$ComposeEmbed = Join-Path $Root "infra\docker\llama-cpp-embed\docker-compose.yml"
$ChatUrl = if ($env:INTERGRAX_DEFAULT_LLAMA_CPP_BASE_URL) { $env:INTERGRAX_DEFAULT_LLAMA_CPP_BASE_URL } else { "http://127.0.0.1:8102/v1" }
$EmbedUrl = if ($env:INTERGRAX_DEFAULT_LLAMA_CPP_EMBED_BASE_URL) { $env:INTERGRAX_DEFAULT_LLAMA_CPP_EMBED_BASE_URL } else { "http://127.0.0.1:8103/v1" }
$ChatModels = ($ChatUrl.TrimEnd("/")) + "/models"
$EmbedModels = ($EmbedUrl.TrimEnd("/")) + "/models"
$MaxWaitSec = if ($env:LLAMA_CPP_VERIFY_MAX_WAIT_SEC) { [int]$env:LLAMA_CPP_VERIFY_MAX_WAIT_SEC } else { 900 }
$PollSec = if ($env:LLAMA_CPP_VERIFY_POLL_SEC) { [int]$env:LLAMA_CPP_VERIFY_POLL_SEC } else { 10 }

$env:INTERGRAX_DEFAULT_LLAMA_CPP_BASE_URL = $ChatUrl
if (-not $env:INTERGRAX_DEFAULT_LLAMA_CPP_MODEL) { $env:INTERGRAX_DEFAULT_LLAMA_CPP_MODEL = "default" }
$env:INTERGRAX_DEFAULT_LLAMA_CPP_EMBED_BASE_URL = $EmbedUrl
if (-not $env:INTERGRAX_DEFAULT_LLAMA_CPP_EMBED_MODEL) { $env:INTERGRAX_DEFAULT_LLAMA_CPP_EMBED_MODEL = "default" }
$env:INTERGRAX_LLAMA_CPP_VERIFY = "1"

function Test-UrlReady {
    param([string]$Url)
    try {
        $null = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 5
        return $true
    } catch {
        return $false
    }
}

function Wait-Url {
    param([string]$Label, [string]$Url)
    $elapsed = 0
    Write-Host "Waiting for $Label at $Url (max ${MaxWaitSec}s)..."
    while ($elapsed -lt $MaxWaitSec) {
        if (Test-UrlReady -Url $Url) {
            Write-Host "$Label is ready."
            return
        }
        Start-Sleep -Seconds $PollSec
        $elapsed += $PollSec
    }
    throw "ERROR: $Label not ready after ${MaxWaitSec}s"
}

if (-not (Test-UrlReady -Url $ChatModels)) {
    Write-Host "Starting standalone llama.cpp chat + embed containers..."
    docker compose -f $ComposeChat up -d
    docker compose -f $ComposeEmbed up -d
}

Wait-Url -Label "llama.cpp chat" -Url $ChatModels
Wait-Url -Label "llama.cpp embed" -Url $EmbedModels

Set-Location $Root
Write-Host "Running llama.cpp E2E tests (excluded from GitHub CI)..."
uv run pytest tests/e2e/llama_cpp/ -m "e2e and no_ci" -q --tb=short
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "llama.cpp verify: OK"
