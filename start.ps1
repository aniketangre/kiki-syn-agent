# start.ps1 — Start the PostgreSQL container then launch the Streamlit app.
# Usage: .\start.ps1

$projectDir = $PSScriptRoot

# Step 1: Ensure the Docker container is running.
# 'docker compose up -d' is a no-op if the container is already up.
Write-Host "Starting rag_postgres container..." -ForegroundColor Cyan
docker compose -f "$projectDir\docker-compose.yml" up -d

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to start Docker container. Is Docker Desktop running?" -ForegroundColor Red
    exit 1
}

Write-Host "rag_postgres is ready." -ForegroundColor Green

# Step 2: Launch the Streamlit app
Write-Host "Launching Streamlit app..." -ForegroundColor Cyan
Set-Location $projectDir
& ".\.venv\Scripts\streamlit.exe" run app.py
