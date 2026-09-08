param(
    [string]$BaseUrl = "http://127.0.0.1:8000",
    [switch]$ApproveMaintenanceRequest
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($env:FACTORY_AGENT_API_KEY)) {
    throw "FACTORY_AGENT_API_KEY must be configured in the environment."
}

$headers = @{
    "X-API-Key" = $env:FACTORY_AGENT_API_KEY
}

Write-Host "[1/5] Liveness"
Invoke-RestMethod "$BaseUrl/health" | ConvertTo-Json -Depth 10

Write-Host "[2/5] Readiness"
Invoke-RestMethod "$BaseUrl/ready" | ConvertTo-Json -Depth 10

Write-Host "[3/5] Current state and historical incident analysis"
$analysisBody = @{
    message = (
        "Robot-01 현재 상태를 확인하고 과거 유사 장애를 찾아서 " +
        "가능성 높은 원인과 정비 방향을 알려줘."
    )
    session_id = "day27-demo"
} | ConvertTo-Json
$analysis = Invoke-RestMethod `
    "$BaseUrl/agent/chat" `
    -Method Post `
    -Headers $headers `
    -ContentType "application/json" `
    -Body $analysisBody
$analysis | ConvertTo-Json -Depth 10

Write-Host "[4/5] Maintenance request approval"
$maintenanceBody = @{
    message = "Robot-01 정비 요청을 만들어줘."
} | ConvertTo-Json
$pending = Invoke-RestMethod `
    "$BaseUrl/tools/chat" `
    -Method Post `
    -Headers $headers `
    -ContentType "application/json" `
    -Body $maintenanceBody
$pending | ConvertTo-Json -Depth 10

$approvalId = $pending.approval_request.approval_id
if ([string]::IsNullOrWhiteSpace($approvalId)) {
    throw "The maintenance request did not return an approval_id."
}

if (-not $ApproveMaintenanceRequest) {
    Write-Host (
        "Approval remains pending. Re-run with " +
        "-ApproveMaintenanceRequest to execute the safe demo action."
    )
    exit 0
}

Write-Host "[5/5] Approve the pending maintenance request"
$approved = Invoke-RestMethod `
    "$BaseUrl/agent/approvals/$approvalId/approve" `
    -Method Post `
    -Headers $headers
$approved | ConvertTo-Json -Depth 10
