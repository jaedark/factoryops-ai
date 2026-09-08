param(
    [string]$Region = "asia-northeast3",
    [string]$Repository = "factory-agent",
    [string]$ServiceName = "factory-agent",
    [string]$RuntimeServiceAccount = "factory-agent-runtime",
    [string]$SecretName = "factory-agent-gemini-api-key",
    [string]$ApiSecretName = "factory-agent-api-key"
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

function Get-OrCreateSecretVersion {
    param(
        [string]$Name,
        [string]$EnvironmentName
    )

    gcloud secrets describe $Name --project=$projectId *> $null
    if ($LASTEXITCODE -ne 0) {
        $secretValue = [Environment]::GetEnvironmentVariable($EnvironmentName)
        if (-not $secretValue) {
            throw "Secret '$Name' does not exist and $EnvironmentName is not set locally."
        }
        gcloud secrets create $Name `
            --replication-policy=automatic `
            --project=$projectId | Out-Null
        $secretFile = New-TemporaryFile
        try {
            [System.IO.File]::WriteAllText(
                $secretFile.FullName,
                $secretValue
            )
            gcloud secrets versions add $Name `
                --data-file=$secretFile.FullName `
                --project=$projectId | Out-Null
        }
        finally {
            Remove-Item -LiteralPath $secretFile.FullName -Force
        }
    }

    gcloud secrets add-iam-policy-binding $Name `
        --member="serviceAccount:$runtimeServiceAccountEmail" `
        --role="roles/secretmanager.secretAccessor" `
        --project=$projectId *> $null

    $versionName = gcloud secrets versions list $Name `
        --filter="state=ENABLED" `
        --sort-by="~createTime" `
        --limit=1 `
        --format="value(name)" `
        --project=$projectId
    if (-not $versionName) {
        throw "No enabled secret version is available for '$Name'."
    }
    return ($versionName -split "/")[-1]
}

$secretVersion = Get-OrCreateSecretVersion `
    -Name $SecretName `
    -EnvironmentName "GEMINI_API_KEY"
$apiSecretVersion = Get-OrCreateSecretVersion `
    -Name $ApiSecretName `
    -EnvironmentName "FACTORY_AGENT_API_KEY"
$imageTag = (git rev-parse --short=12 HEAD).Trim()

gcloud builds submit . `
    --config=cloudbuild.yaml `
    --region=$Region `
    --project=$projectId `
    --substitutions="_REGION=$Region,_REPOSITORY=$Repository,_IMAGE_NAME=$ServiceName,_IMAGE_TAG=$imageTag,_SERVICE_NAME=$ServiceName,_RUNTIME_SERVICE_ACCOUNT=$RuntimeServiceAccount,_SECRET_NAME=$SecretName,_SECRET_VERSION=$secretVersion,_API_SECRET_NAME=$ApiSecretName,_API_SECRET_VERSION=$apiSecretVersion"

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
