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
. (Join-Path $PSScriptRoot 'python_env.ps1')

Push-Location $Root
try {
    $basePython = Get-PythonCommand -Requested $Python
    $probe = Invoke-NativeCommand -Command $basePython `
        -Arguments @('-c', "import sys; print('%d.%d' % sys.version_info[:2])")
    $version = [string]($probe.Output | Select-Object -Last 1)
    $version = $version.Trim()
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
