#Requires -Version 5.1
<#
.SYNOPSIS
    Interpreter discovery and tolerant native invocation for the Windows entry points.
.DESCRIPTION
    Dot-source this file from a script that needs to locate Python 3.10+ or run a
    native command whose failure is expected.

    Windows PowerShell turns a native command's stderr into a terminating
    NativeCommandError while $ErrorActionPreference is 'Stop'. Interpreter
    probing relies on commands that fail by design (a py launcher with no
    registered interpreter, a broken python shim), so those calls run with a
    relaxed preference and report the exit code instead of aborting.
#>

function Invoke-NativeCommand {
    <#
    .SYNOPSIS
        Run a native command and return its exit code and output.
    .DESCRIPTION
        Stderr is merged into the returned output. A non-zero exit raises an
        error unless -IgnoreExitCode is given, in which case the caller decides
        what the code means.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string[]]$Command,
        [string[]]$Arguments = @(),
        [switch]$IgnoreExitCode
    )
    $executable = $Command[0]
    $all = @($Command | Select-Object -Skip 1) + $Arguments
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $output = & $executable @all 2>&1
        $code = $LASTEXITCODE
    }
    catch {
        $output = @($_.Exception.Message)
        $code = -1
    }
    finally {
        $ErrorActionPreference = $previous
    }
    if (-not $IgnoreExitCode -and $code -ne 0) {
        throw "command failed: $executable $($all -join ' ') (exit code $code)"
    }
    return [pscustomobject]@{ ExitCode = $code; Output = @($output) }
}

function Test-PythonCommand {
    <#
    .SYNOPSIS
        Return $true when the command starts and reports a Python version.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory = $true)][string[]]$Command)
    $probe = Invoke-NativeCommand -Command $Command -Arguments @('-c', 'import sys') -IgnoreExitCode
    return $probe.ExitCode -eq 0
}

function Get-PythonCommand {
    <#
    .SYNOPSIS
        Resolve the interpreter used to build the CPU virtual environment.
    .DESCRIPTION
        Order: the -Requested value, then the py launcher, then python and
        python3 on PATH. Candidates are probed before being accepted, so a
        launcher or shim that exists but cannot start a Python interpreter falls
        through to the next candidate instead of aborting the caller.
    #>
    [CmdletBinding()]
    param([string]$Requested)

    if ($Requested) {
        $resolved = Get-Command $Requested -ErrorAction SilentlyContinue
        if ($resolved) { return , @($resolved.Source) }
        if (Test-Path -LiteralPath $Requested) { return , @((Resolve-Path -LiteralPath $Requested).Path) }
        throw "Python interpreter not found: $Requested"
    }

    if (Get-Command py -ErrorAction SilentlyContinue) {
        if (Test-PythonCommand -Command @('py', '-3')) { return , @('py', '-3') }
        Write-Verbose 'py launcher found but "py -3" cannot start an interpreter; trying PATH candidates'
    }

    foreach ($name in 'python', 'python3') {
        $candidate = Get-Command $name -ErrorAction SilentlyContinue
        if (-not $candidate) { continue }
        if (Test-PythonCommand -Command @($candidate.Source)) { return , @($candidate.Source) }
    }

    throw 'Python 3.10+ was not found. Install it from https://www.python.org/downloads/ or pass -Python <path>.'
}

function Invoke-PythonStep {
    <#
    .SYNOPSIS
        Run a step of the environment setup, keeping its output visible.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string[]]$Command,
        [string[]]$Arguments = @(),
        [Parameter(Mandatory = $true)][string]$Label
    )
    Write-Host "==> $Label"
    $executable = $Command[0]
    $all = @($Command | Select-Object -Skip 1) + $Arguments
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $executable @all
        $code = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previous
    }
    if ($code -ne 0) {
        throw "step failed ($Label): $executable $($all -join ' ') (exit code $code)"
    }
}
