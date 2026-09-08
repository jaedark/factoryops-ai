param(
    [string]$Region = "asia-northeast3",
    [string]$Repository = "factory-agent",
    [string]$ServiceName = "factory-agent",
    [string]$RuntimeServiceAccount = "factory-agent-runtime",
    [string]$SecretName = "factory-agent-gemini-api-key"
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    throw "gcloud CLI is not installed or is not available on PATH."
}

$activeAccount = gcloud auth list --filter="status:ACTIVE" --format="value(account)"
if (-not $activeAccount) {
    throw "No active gcloud account is configured."
}

$projectId = gcloud config get-value project 2>$null
if (-not $projectId -or $projectId -eq "(unset)") {
    throw "No active Google Cloud project is configured."
}

$requiredApis = @(
    "run.googleapis.com",
    "cloudbuild.googleapis.com",
    "artifactregistry.googleapis.com",
    "secretmanager.googleapis.com",
    "logging.googleapis.com",
    "iam.googleapis.com",
    "cloudresourcemanager.googleapis.com"
)
gcloud services enable $requiredApis --project=$projectId

gcloud artifacts repositories describe $Repository `
    --location=$Region `
    --project=$projectId *> $null
if ($LASTEXITCODE -ne 0) {
    gcloud artifacts repositories create $Repository `
        --repository-format=docker `
        --location=$Region `
        --description="Factory Agent container images" `
        --project=$projectId
}

$runtimeServiceAccountEmail = (
    "$RuntimeServiceAccount@$projectId.iam.gserviceaccount.com"
)
gcloud iam service-accounts describe $runtimeServiceAccountEmail `
    --project=$projectId *> $null
if ($LASTEXITCODE -ne 0) {
    gcloud iam service-accounts create $RuntimeServiceAccount `
        --display-name="Factory Agent Cloud Run runtime" `
        --project=$projectId
}

gcloud secrets describe $SecretName --project=$projectId *> $null
if ($LASTEXITCODE -ne 0) {
    if (-not $env:GEMINI_API_KEY) {
        throw "Secret does not exist and GEMINI_API_KEY is not set locally."
    }
    gcloud secrets create $SecretName `
        --replication-policy=automatic `
        --project=$projectId
    $secretFile = New-TemporaryFile
    try {
        [System.IO.File]::WriteAllText(
            $secretFile.FullName,
            $env:GEMINI_API_KEY
        )
        gcloud secrets versions add $SecretName `
            --data-file=$secretFile.FullName `
            --project=$projectId
    }
    finally {
        Remove-Item -LiteralPath $secretFile.FullName -Force
    }
}

gcloud secrets add-iam-policy-binding $SecretName `
    --member="serviceAccount:$runtimeServiceAccountEmail" `
    --role="roles/secretmanager.secretAccessor" `
    --project=$projectId *> $null

$secretVersionName = gcloud secrets versions list $SecretName `
    --filter="state=ENABLED" `
    --sort-by="~createTime" `
    --limit=1 `
    --format="value(name)" `
    --project=$projectId
if (-not $secretVersionName) {
    throw "No enabled secret version is available."
}
$secretVersion = ($secretVersionName -split "/")[-1]
$imageTag = (git rev-parse --short=12 HEAD).Trim()

gcloud builds submit . `
    --config=cloudbuild.yaml `
    --region=$Region `
    --project=$projectId `
    --substitutions="_REGION=$Region,_REPOSITORY=$Repository,_IMAGE_NAME=$ServiceName,_IMAGE_TAG=$imageTag,_SERVICE_NAME=$ServiceName,_RUNTIME_SERVICE_ACCOUNT=$RuntimeServiceAccount,_SECRET_NAME=$SecretName,_SECRET_VERSION=$secretVersion"

$serviceUrl = gcloud run services describe $ServiceName `
    --region=$Region `
    --project=$projectId `
    --format="value(status.url)"
$identityToken = gcloud auth print-identity-token
$health = Invoke-RestMethod `
    -Uri "$serviceUrl/health" `
    -Headers @{ Authorization = "Bearer $identityToken" }

Write-Output "Cloud Run service: $serviceUrl"
Write-Output "Health status: $($health.status)"
