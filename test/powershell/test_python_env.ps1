#Requires -Version 5.1
<#
.SYNOPSIS
    Tests for scripts/python_env.ps1 interpreter discovery.
.DESCRIPTION
    Fake launchers are placed in front of PATH so every branch of the probe can
    be exercised without touching the machine's Python installation. The last
    case runs scripts/setup_cpu.ps1 end to end under Windows PowerShell to prove
    that a failing py launcher no longer aborts the script.
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File test\powershell\test_python_env.ps1
#>
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
. (Join-Path $Root 'scripts\python_env.ps1')

$WindowsPowerShell = (Get-Command powershell -ErrorAction SilentlyContinue | Select-Object -First 1).Source
$script:Failures = 0

function Assert-Condition {
    param([bool]$Condition, [string]$Message)
    if ($Condition) {
        Write-Host "  ok   $Message"
    }
    else {
        Write-Host "  FAIL $Message" -ForegroundColor Red
        $script:Failures++
    }
}

function Assert-Contains {
    param([string]$Text, [string]$Needle, [string]$Message)
    Assert-Condition ($Text -like "*$Needle*") $Message
}

function New-FakeLauncher {
    param(
        [Parameter(Mandatory = $true)][string]$Directory,
        [Parameter(Mandatory = $true)][string]$Name,
        [int]$ExitCode = 0,
        [string]$Message = ''
    )
    $lines = @('@echo off')
    if ($Message) {
        $lines += "echo $Message 1>&2"
    }
    $lines += "exit /b $ExitCode"
    $path = Join-Path $Directory "$Name.cmd"
    Set-Content -LiteralPath $path -Value $lines -Encoding ASCII
    return $path
}

function New-TempDirectory {
    $name = 'python-env-' + [guid]::NewGuid().ToString('N').Substring(0, 8)
    $path = Join-Path ([System.IO.Path]::GetTempPath()) $name
    New-Item -ItemType Directory -Path $path -Force | Out-Null
    return $path
}

function Invoke-WithPath {
    param(
        [Parameter(Mandatory = $true)][string]$PathValue,
        [Parameter(Mandatory = $true)][scriptblock]$Body
    )
    $previous = $env:PATH
    try {
        $env:PATH = $PathValue
        & $Body
    }
    finally {
        $env:PATH = $previous
    }
}

$temp = New-TempDirectory
$directories = @($temp)
try {
    Write-Host 'explicit interpreter wins'
    $explicit = New-FakeLauncher -Directory $temp -Name 'python-ok'
    $resolved = Get-PythonCommand -Requested $explicit
    Assert-Condition ($resolved.Count -eq 1 -and $resolved[0] -eq $explicit) '-Requested path is returned unchanged'
    $missing = Join-Path $temp 'does-not-exist.exe'
    $threw = $false
    try { Get-PythonCommand -Requested $missing | Out-Null } catch { $threw = $true }
    Assert-Condition $threw '-Requested path that does not exist throws'

    Write-Host 'py launcher present but "py -3" fails'
    $broken = New-TempDirectory
    $directories += $broken
    New-FakeLauncher -Directory $broken -Name 'py' -ExitCode 1 -Message 'No installed Python found!' | Out-Null
    $fallback = New-FakeLauncher -Directory $broken -Name 'python' -ExitCode 0 -Message '3.12.14'
    $resolved = Invoke-WithPath -PathValue $broken -Body { Get-PythonCommand }
    Assert-Condition ($resolved.Count -eq 1 -and $resolved[0] -eq $fallback) 'falls back to python on PATH instead of aborting'

    Write-Host 'py launcher works'
    $healthy = New-TempDirectory
    $directories += $healthy
    New-FakeLauncher -Directory $healthy -Name 'py' -ExitCode 0 -Message 'Python 3.12.14' | Out-Null
    New-FakeLauncher -Directory $healthy -Name 'python' -ExitCode 0 -Message '3.12.14' | Out-Null
    $resolved = Invoke-WithPath -PathValue $healthy -Body { Get-PythonCommand }
    Assert-Condition ($resolved.Count -eq 2 -and $resolved[0] -eq 'py' -and $resolved[1] -eq '-3') 'returns the py launcher when it works'

    Write-Host 'first PATH candidate is broken'
    $second = New-TempDirectory
    $directories += $second
    New-FakeLauncher -Directory $second -Name 'py' -ExitCode 1 -Message 'No installed Python found!' | Out-Null
    New-FakeLauncher -Directory $second -Name 'python' -ExitCode 9009 -Message 'not found' | Out-Null
    $usable = New-FakeLauncher -Directory $second -Name 'python3' -ExitCode 0 -Message '3.12.14'
    $resolved = Invoke-WithPath -PathValue $second -Body { Get-PythonCommand }
    Assert-Condition ($resolved.Count -eq 1 -and $resolved[0] -eq $usable) 'skips a python that cannot start and accepts python3'

    Write-Host 'no interpreter at all'
    $empty = New-TempDirectory
    $directories += $empty
    $message = ''
    try { Invoke-WithPath -PathValue $empty -Body { Get-PythonCommand } | Out-Null } catch { $message = $_.Exception.Message }
    Assert-Contains $message 'Python 3.10+ was not found' 'reports the actionable message'

    Write-Host 'setup_cpu.ps1 end to end with a failing launcher'
    if (-not $WindowsPowerShell) {
        Write-Host '  skip  Windows PowerShell is not available on this platform'
    }
    else {
        $script = Join-Path $Root 'scripts\setup_cpu.ps1'
        $result = Invoke-WithPath -PathValue $empty -Body {
            Invoke-NativeCommand -Command @($WindowsPowerShell) -IgnoreExitCode -Arguments @(
                '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $script, '-SkipTests')
        }
        $text = [string]($result.Output -join "`n")
        Assert-Condition ($result.ExitCode -ne 0) 'entry point fails when no interpreter exists'
        Assert-Contains $text 'Python 3.10+ was not found' 'entry point reports the actionable message'
        Assert-Condition ($text -notlike '*NativeCommandError*') 'entry point no longer aborts with NativeCommandError'
    }
}
finally {
    $tempRoot = [System.IO.Path]::GetTempPath()
    foreach ($directory in $directories) {
        if (-not $directory) { continue }
        $full = [System.IO.Path]::GetFullPath($directory)
        if (-not $full.StartsWith($tempRoot, [System.StringComparison]::OrdinalIgnoreCase)) { continue }
        if ((Split-Path -Leaf $full) -notlike 'python-env-*') { continue }
        if (Test-Path -LiteralPath $full) {
            Remove-Item -LiteralPath $full -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}

if ($script:Failures -gt 0) {
    Write-Host "$script:Failures check(s) failed" -ForegroundColor Red
    exit 1
}
Write-Host 'all interpreter discovery checks passed'
exit 0
