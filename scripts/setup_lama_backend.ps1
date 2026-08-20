param(
    [string]$BasePython = "python",
    [string]$PackageRoot = "",
    [string]$ModelCache = "",
    [string]$ConfigPath = ""
)

$ErrorActionPreference = "Stop"
$repositoryRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$arguments = @((Join-Path $PSScriptRoot "setup_lama_backend.py"))
if ($PackageRoot) { $arguments += @("--package-root", $PackageRoot) }
if ($ModelCache) { $arguments += @("--model-cache", $ModelCache) }
if ($ConfigPath) { $arguments += @("--config", $ConfigPath) }

& $BasePython @arguments
if ($LASTEXITCODE -ne 0) {
    throw "LaMa setup failed with exit code $LASTEXITCODE"
}
Write-Host "HPID completion configuration is ready."
