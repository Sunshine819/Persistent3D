[CmdletBinding()]
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$PipelineArgs)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $RepoRoot
python scripts\run_pipeline.py configs\scan.json @PipelineArgs
exit $LASTEXITCODE
