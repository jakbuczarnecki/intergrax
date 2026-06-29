# Commit INTEGRATIONS-2E cutover in reviewable waves.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path (Split-Path $PSScriptRoot))

function Commit-Wave {
    param([string[]]$Paths, [string]$Message)
    if (-not $Paths) { return }
    git add @Paths
    $status = git diff --cached --name-only
    if (-not $status) {
        Write-Host "SKIP (empty): $Message"
        return
    }
    git commit -m $Message
    Write-Host "OK: $Message ($(( $status | Measure-Object -Line ).Lines) files)"
}

# 1 — tooling
Commit-Wave @(
    "intergrax/integrations/_shared/runtime_cutover_templates.py",
    "scripts/maintenance/cutover_provider_runtime_integrations.py",
    "scripts/maintenance/fix_runtime_delegation.py"
) "chore(integrations): add runtime cutover maintenance tooling"

# 2 — guard tests
Commit-Wave @(
    "tests/unit/integrations/providers/test_provider_runtime_cutover.py"
) "test(integrations): extend runtime cutover guards to full catalog"

$categoryWaves = @(
    @{ Cat = "vector_store"; Msg = "refactor(integrations): cut over vector store provider runtime integrations" },
    @{ Cat = "observability_backend"; Msg = "refactor(integrations): cut over observability backend runtime integrations" },
    @{ Cat = "relational_store"; Msg = "refactor(integrations): cut over relational store runtime integrations" },
    @{ Cat = "object_storage"; Msg = "refactor(integrations): cut over object storage runtime integrations" },
    @{ Cat = "document_store"; Msg = "refactor(integrations): cut over document store runtime integrations" },
    @{ Cat = "message_bus"; Msg = "refactor(integrations): cut over message bus runtime integrations" },
    @{ Cat = "notification_channel"; Msg = "refactor(integrations): cut over notification channel runtime integrations" },
    @{ Cat = "search_provider"; Msg = "refactor(integrations): cut over search provider runtime integrations" },
    @{ Cat = "issue_tracker"; Msg = "refactor(integrations): cut over issue tracker runtime integrations" },
    @{ Cat = "document_parser"; Msg = "refactor(integrations): cut over document parser runtime integrations" },
    @{ Cat = "ci_cd"; Msg = "refactor(integrations): cut over ci_cd runtime integrations" },
    @{ Cat = "secrets_store"; Msg = "refactor(integrations): cut over secrets store runtime integrations" },
    @{ Cat = "cloud_platform"; Msg = "refactor(integrations): cut over cloud platform runtime integrations" },
    @{ Cat = "browser_automation"; Msg = "refactor(integrations): cut over browser automation runtime integrations" },
    @{ Cat = "key_value_cache"; Msg = "refactor(integrations): cut over key value cache runtime integrations" },
    @{ Cat = "workflow_orchestrator"; Msg = "refactor(integrations): cut over workflow orchestrator runtime integrations" },
    @{ Cat = "wiki_knowledge"; Msg = "refactor(integrations): cut over wiki knowledge runtime integrations" },
    @{ Cat = "interaction_surface"; Msg = "refactor(integrations): cut over interaction surface runtime integrations" },
    @{ Cat = "graph_store"; Msg = "refactor(integrations): cut over graph store runtime integrations" },
    @{ Cat = "feature_flag"; Msg = "refactor(integrations): cut over feature flag runtime integrations" },
    @{ Cat = "collaboration_suite"; Msg = "refactor(integrations): cut over collaboration suite runtime integrations" },
    @{ Cat = "identity_provider"; Msg = "refactor(integrations): cut over identity provider runtime integrations" },
    @{ Cat = "security_scanner"; Msg = "refactor(integrations): cut over security scanner runtime integrations" },
    @{ Cat = "sandbox_host"; Msg = "refactor(integrations): cut over sandbox host runtime integrations" },
    @{ Cat = "speech_provider"; Msg = "refactor(integrations): cut over speech provider runtime integrations" },
    @{ Cat = "rerank_provider"; Msg = "refactor(integrations): cut over rerank provider runtime integrations" },
    @{ Cat = "billing_meter"; Msg = "refactor(integrations): cut over billing meter runtime integrations" },
    @{ Cat = "ml_inference_host"; Msg = "refactor(integrations): cut over ml inference host runtime integrations" },
    @{ Cat = "vision_serving"; Msg = "refactor(integrations): cut over vision serving runtime integrations" },
    @{ Cat = "crm"; Msg = "refactor(integrations): cut over crm runtime integrations" }
)

foreach ($wave in $categoryWaves) {
    $cat = $wave.Cat
    $path = "intergrax/integrations/providers/$cat"
    if (Test-Path $path) {
        Commit-Wave @($path) $wave.Msg
    }
}

# leftover provider tests
Commit-Wave @(
    "tests/unit/integrations/providers",
    "tests/unit/applications/test_lab_integration_wiring.py"
) "test(integrations): align provider tests with runtime cutover adapters"

$remaining = git status --short
if ($remaining) {
    Write-Host "REMAINING UNCOMMITTED:"
    Write-Host $remaining
    exit 1
}
Write-Host "All cutover waves committed."
