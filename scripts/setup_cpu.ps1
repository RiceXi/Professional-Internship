#Requires -Version 5.1
<#
.SYNOPSIS
    Build the CPU-only environment and run the coverage-gated test suite.
.DESCRIPTION
    Windows counterpart of scripts/setup_cpu.sh. It creates .venv-cpu,
    installs the locked coverage version and runs test/run_cpu.py --coverage.
    The simulator itself only needs the standard library and Python 3.10+.
.PARAMETER Python
    Interpreter used to create the virtual environment. Defaults to $env:PYTHON,
    then to the py launcher, python or python3 on PATH.
.PARAMETER Venv
    Virtual environment directory, relative to the repository root.
.PARAMETER SkipTests
    Only prepare the environment; do not run the test suite.
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/setup_cpu.ps1
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/setup_cpu.ps1 -Python C:\Python312\python.exe
#>
[CmdletBinding()]
param(
    [string]$Python = $env:PYTHON,
    [string]$Venv = '.venv-cpu',
    [switch]$SkipTests
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = Split-Path -Parent $PSScriptRoot

function Get-PythonCommand {
    param([string]$Requested)
    if ($Requested) {
        $resolved = Get-Command $Requested -ErrorAction SilentlyContinue
        if ($resolved) { return , @($resolved.Source) }
        if (Test-Path -LiteralPath $Requested) { return , @((Resolve-Path -LiteralPath $Requested).Path) }
        throw "Python interpreter not found: $Requested"
    }
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3 --version *> $null
        if ($LASTEXITCODE -eq 0) { return , @('py', '-3') }
    }
    foreach ($name in 'python', 'python3') {
        $candidate = Get-Command $name -ErrorAction SilentlyContinue
        if ($candidate) { return , @($candidate.Source) }
    }
    throw 'Python 3.10+ was not found. Install it from https://www.python.org/downloads/ or pass -Python <path>.'
}

function Invoke-PythonStep {
    param([string[]]$Command, [string[]]$Arguments, [string]$Label)
    Write-Host "==> $Label"
    $executable = $Command[0]
    $arguments = @($Command | Select-Object -Skip 1) + $Arguments
    & $executable @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "step failed ($Label): $executable $($arguments -join ' ')"
    }
}

Push-Location $Root
try {
    $basePython = Get-PythonCommand -Requested $Python
    $version = & $basePython[0] @($basePython | Select-Object -Skip 1) -c "import sys; print('%d.%d' % sys.version_info[:2])"
    if ($LASTEXITCODE -ne 0) { throw 'failed to query the Python version' }
    if ([version]$version -lt [version]'3.10') {
        throw "Python 3.10+ is required, found $version"
    }
    Write-Host "python: $($basePython -join ' ') ($version)"

    $venvPython = Join-Path $Root (Join-Path $Venv 'Scripts\python.exe')
    if (Test-Path -LiteralPath $venvPython) {
        Write-Host "==> reuse existing virtual environment: $Venv"
    }
    else {
        Invoke-PythonStep -Command $basePython -Arguments @('-m', 'venv', $Venv) -Label "create $Venv"
    }

    Invoke-PythonStep -Command @($venvPython) -Arguments @('-m', 'pip', 'install', '-r', 'requirements-cpu.lock') -Label 'install locked test dependency'

    if ($SkipTests) {
        Write-Host '==> tests skipped (-SkipTests)'
    }
    else {
        Invoke-PythonStep -Command @($venvPython) -Arguments @('test/run_cpu.py', '--coverage') -Label 'run CPU tests with coverage'
        Write-Host "==> CPU test gate passed; summary in experiments/local/test_summary.json"
    }
}
finally {
    Pop-Location
}
