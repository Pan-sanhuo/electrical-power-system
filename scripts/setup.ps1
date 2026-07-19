$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Host "正在创建 Python 虚拟环境..."
    python -m venv (Join-Path $root ".venv")
}

Write-Host "正在安装经过验证的依赖版本..."
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r (Join-Path $root "requirements.txt")
& $venvPython -m pip install pytest

Write-Host "正在检查环境..."
& $venvPython -m pfagent doctor
Write-Host "安装完成。请在 VS Code 中运行任务：2. 运行完整演示"
