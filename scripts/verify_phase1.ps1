$ErrorActionPreference = 'Stop'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..'))

$ids = @{}
foreach ($service in @('db', 'api', 'dashboard')) {
    $id = (docker compose ps -q $service).Trim()
    if ($LASTEXITCODE -ne 0 -or !$id) { throw "No MarketOps container found: $service" }
    $health = docker inspect --format '{{.State.Health.Status}}' $id
    if ($LASTEXITCODE -ne 0 -or $health.Trim() -ne 'healthy') {
        throw "MarketOps $service is not healthy: $health"
    }
    Write-Host "MarketOps ${service}: healthy"
}
$apiAddress = (docker compose port api 8000).Trim()
$dashAddress = (docker compose port dashboard 8501).Trim()
if (!$apiAddress -or !$dashAddress) { throw 'Published ports not available' }
$apiUrl = 'http://' + $apiAddress
$dashUrl = 'http://' + $dashAddress
$live = Invoke-RestMethod -Uri "$apiUrl/health/live" -TimeoutSec 10
if ($live.status -ne 'ok' -or $live.service -ne 'marketops-api') { throw 'Wrong API response' }
$ready = Invoke-RestMethod -Uri "$apiUrl/health/ready" -TimeoutSec 10
if ($ready.status -ne 'ready' -or $ready.dependencies.postgres -ne 'up') {
    throw 'PostgreSQL readiness failed'
}
$example = Invoke-RestMethod -Uri "$apiUrl/portfolio/example" -TimeoutSec 10
if ($example.source -ne 'synthetic_example' -or $example.holdings.Count -ne 2) {
    throw 'Synthetic example portfolio failed'
}
$dashboard = Invoke-WebRequest -Uri "$dashUrl/_stcore/health" -TimeoutSec 15 -UseBasicParsing
if ($dashboard.StatusCode -ne 200) { throw 'Dashboard health check failed' }
Write-Host "MarketOps foundation passed. UI: $dashUrl"
