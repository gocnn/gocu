# Set console output encoding to UTF-8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Initialize result collection
$results = [System.Collections.Generic.List[PSObject]]::new()

# Function to add result
function Add-Result {
    param($Check, $Status, $Details)
    $results.Add([PSCustomObject]@{
            Check   = $Check
            Status  = $Status
            Details = $Details
        })
}

# Function to display styled messages
function Write-Message {
    param($Message, $Type)
    $symbols = @{ Success = "OK"; Info = "INFO"; Warning = "WARN"; Error = "ERR" }
    $colors = @{ Success = "Green"; Info = "Cyan"; Warning = "Yellow"; Error = "Red" }
    Write-Host "[$($symbols[$Type])] $Message" -ForegroundColor $colors[$Type]
}

# Function to display banner
function Show-Banner {
    Write-Host ""
    Write-Host "          _____                   _______                   _____                    _____" -ForegroundColor Cyan
    Write-Host "         /\    \                 /::\    \                 /\    \                  /\    \" -ForegroundColor Cyan
    Write-Host "        /::\    \               /::::\    \               /::\    \                /::\____\" -ForegroundColor Cyan
    Write-Host "       /::::\    \             /::::::\    \             /::::\    \              /:::/    /" -ForegroundColor Cyan
    Write-Host "      /::::::\    \           /::::::::\    \           /::::::\    \            /:::/    /" -ForegroundColor Cyan
    Write-Host "     /:::/\:::\    \         /:::/~~\:::\    \         /:::/\:::\    \          /:::/    /" -ForegroundColor Cyan
    Write-Host "    /:::/  \:::\    \       /:::/    \:::\    \       /:::/  \:::\    \        /:::/    /" -ForegroundColor Cyan
    Write-Host "   /:::/    \:::\    \     /:::/    / \:::\    \     /:::/    \:::\    \      /:::/    /" -ForegroundColor Cyan
    Write-Host "  /:::/    / \:::\    \   /:::/____/   \:::\____\   /:::/    / \:::\    \    /:::/    /      _____" -ForegroundColor Cyan
    Write-Host " /:::/    /   \:::\ ___\ |:::|    |     |:::|    | /:::/    /   \:::\    \  /:::/____/      /\    \" -ForegroundColor Cyan
    Write-Host "/:::/____/  ___\:::|    ||:::|____|     |:::|    |/:::/____/     \:::\____\|:::|    /      /::\____\" -ForegroundColor Cyan
    Write-Host "\:::\    \ /\  /:::|____| \:::\    \   /:::/    / \:::\    \      \::/    /|:::|____\     /:::/    /" -ForegroundColor Cyan
    Write-Host " \:::\    /::\ \::/    /   \:::\    \ /:::/    /   \:::\    \      \/____/  \:::\    \   /:::/    /" -ForegroundColor Cyan
    Write-Host "  \:::\   \:::\ \/____/     \:::\    /:::/    /     \:::\    \               \:::\    \ /:::/    /" -ForegroundColor Cyan
    Write-Host "   \:::\   \:::\____\        \:::\__/:::/    /       \:::\    \               \:::\    /:::/    /" -ForegroundColor Cyan
    Write-Host "    \:::\  /:::/    /         \::::::::/    /         \:::\    \               \:::\__/:::/    /" -ForegroundColor Cyan
    Write-Host "     \:::\/:::/    /           \::::::/    /           \:::\    \               \::::::::/    /" -ForegroundColor Cyan
    Write-Host "      \::::::/    /             \::::/    /             \:::\    \               \::::::/    /" -ForegroundColor Cyan
    Write-Host "       \::::/    /               \::/____/               \:::\____\               \::::/    /" -ForegroundColor Cyan
    Write-Host "        \::/____/                 ~~                      \::/    /                \::/____/" -ForegroundColor Cyan
    Write-Host "                                                           \/____/                  ~~" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "                                   CUDA Toolkit Setup for GOCU" -ForegroundColor Yellow
    Write-Host "                                      github.com/gocnn/gocu" -ForegroundColor DarkGray
    Write-Host "                                           BSD 3-Clause" -ForegroundColor DarkGray
    Write-Host ""
}

# Function to display step
function Show-Step {
    param($Step, $Description)
    Write-Host "[$Step] $Description" -ForegroundColor Yellow
}

# Function to display colored summary
function Show-Summary {
    Write-Host ("`n" + "-" * 49) -ForegroundColor DarkGray
    Write-Host "CUDA Environment Validation Report:" -ForegroundColor Yellow
    Write-Host ""
    foreach ($result in $results) {
        $statusColor = switch ($result.Status) {
            "Success" { "Green" }
            "Failed" { "Red" }
            "Warning" { "Yellow" }
            default { "White" }
        }
        Write-Host ("{0,-20} {1,-10} {2}" -f $result.Check, $result.Status, $result.Details) -ForegroundColor $statusColor
    }
}

# Display banner
Show-Banner

# Ensure admin privileges
if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Start-Process powershell -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`"" -Verb RunAs
    exit
}

# Step 1: Check NVIDIA GPU
Show-Step "1" "Checking for NVIDIA GPU"
$nvidiaGpu = Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match "NVIDIA" -and $_.AdapterCompatibility -match "NVIDIA" }
if (-not $nvidiaGpu) {
    Write-Message "No NVIDIA GPU detected." -Type Error
    Add-Result "NVIDIA GPU" "Failed" "No NVIDIA GPU found."
    Write-Host "`nTutorial: Ensure you have an NVIDIA GPU installed. Download and install the latest drivers from https://www.nvidia.com/Download/index.aspx. After installation, rerun this script." -ForegroundColor Yellow
    Show-Summary
    pause
    exit 1
}
Write-Message "NVIDIA GPU detected: $($nvidiaGpu.Name)" -Type Success
Add-Result "NVIDIA GPU" "Success" $nvidiaGpu.Name

# Define paths
$cudaBasePath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
$linkPath = "C:\cuda"

# Step 2: Check CUDA installation
Show-Step "2" "Checking CUDA Toolkit Installation"
if (-not (Test-Path $cudaBasePath)) {
    Write-Message "CUDA Toolkit not found at $cudaBasePath." -Type Error
    Add-Result "CUDA Installation" "Failed" "CUDA Toolkit not found."
    Write-Host "`nTutorial: Download and install CUDA Toolkit from https://developer.nvidia.com/cuda-downloads. Choose the appropriate version for your system. After installation, rerun this script." -ForegroundColor Yellow
    Show-Summary
    pause
    exit 1
}
Write-Message "CUDA Toolkit found." -Type Success
Add-Result "CUDA Installation" "Success" "Found at $cudaBasePath."

# Get CUDA versions
$versions = Get-ChildItem -Path $cudaBasePath -Directory | Where-Object { $_.Name -match '^v\d+\.\d+$' } | Select-Object -ExpandProperty Name
if (-not $versions) {
    Write-Message "No CUDA versions found." -Type Error
    Add-Result "CUDA Versions" "Failed" "No versions found in $cudaBasePath."
    Write-Host "`nTutorial: Ensure CUDA is properly installed in $cudaBasePath. Reinstall if necessary from https://developer.nvidia.com/cuda-downloads. Rerun the script after." -ForegroundColor Yellow
    Show-Summary
    pause
    exit 1
}
Write-Message "Found CUDA version(s): $($versions -join ', ')" -Type Success
Add-Result "CUDA Versions" "Success" ($versions -join ', ')

# Step 3: Select CUDA version
Show-Step "3" "Selecting CUDA Version"
$selectedVersion = $versions | Out-GridView -Title "Select CUDA Toolkit Version" -PassThru
if (-not $selectedVersion) {
    Write-Message "No version selected." -Type Warning
    Add-Result "Version Selection" "Cancelled" "User cancelled selection."
    Show-Summary
    pause
    exit 0
}
Write-Message "Selected CUDA version: $selectedVersion" -Type Success
Add-Result "Version Selection" "Success" $selectedVersion

# Step 4: Create symbolic link
Show-Step "4" "Creating Symbolic Link"
$selectedPath = Join-Path $cudaBasePath $selectedVersion
if (-not (Test-Path $selectedPath)) {
    Write-Message "Selected version path does not exist: $selectedPath." -Type Error
    Add-Result "Symbolic Link" "Failed" "Invalid path: $selectedPath."
    Write-Host "`nTutorial: Verify the selected CUDA version directory exists. Reinstall CUDA if missing. Rerun the script." -ForegroundColor Yellow
    Show-Summary
    pause
    exit 1
}
if (Test-Path $linkPath) {
    Remove-Item $linkPath -Force -Recurse
    Write-Message "Removed existing symbolic link." -Type Info
}
try {
    New-Item -ItemType SymbolicLink -Path $linkPath -Target $selectedPath -Force | Out-Null
    Write-Message "Symbolic link created: $linkPath -> $selectedPath" -Type Success
    Add-Result "Symbolic Link" "Success" "$linkPath -> $selectedPath"
}
catch {
    Write-Message "Failed to create symbolic link: $_" -Type Error
    Add-Result "Symbolic Link" "Failed" $_.Exception.Message
    Write-Host "`nTutorial: Ensure you have admin rights and no file locks. Manually create the link with 'mklink /D C:\cuda `"$selectedPath`"' in an admin Command Prompt. Rerun the script." -ForegroundColor Yellow
    Show-Summary
    pause
    exit 1
}

# Step 5: Check CUDA Compiler (nvcc)
Show-Step "5" "Checking CUDA Compiler (nvcc)"
$nvccPath = Join-Path $linkPath "bin\nvcc.exe"
if (-not (Test-Path $nvccPath)) {
    Write-Message "CUDA compiler (nvcc) not found." -Type Error
    Add-Result "CUDA Compiler" "Failed" "nvcc not found at $nvccPath."
    Write-Host "`nTutorial: Reinstall CUDA Toolkit ensuring the compiler is included. Download from https://developer.nvidia.com/cuda-downloads. Rerun the script." -ForegroundColor Yellow
    Show-Summary
    pause
    exit 1
}
Write-Message "CUDA compiler found." -Type Success
Add-Result "CUDA Compiler" "Success" "nvcc found at $nvccPath."

# Step 6: Check/update PATH
Show-Step "6" "Checking System PATH for CUDA"
$binPath = Join-Path $linkPath "bin"
$systemPath = [Environment]::GetEnvironmentVariable("Path", "Machine")
if ($systemPath -notmatch [regex]::Escape($binPath)) {
    try {
        $newPath = "$systemPath;$binPath".Trim(';')
        [Environment]::SetEnvironmentVariable("Path", $newPath, "Machine")
        Write-Message "Added $binPath to system PATH." -Type Success
        Add-Result "System PATH" "Success" "Added $binPath."
        Write-Message "Restart your terminal to apply PATH changes." -Type Info
    }
    catch {
        Write-Message "Failed to update PATH: $_" -Type Error
        Add-Result "System PATH" "Failed" $_.Exception.Message
        Write-Host "`nTutorial: Manually add $binPath to System Environment Variables > Path. Search 'Edit the system environment variables' in Windows Search. Restart terminal and rerun script." -ForegroundColor Yellow
        Show-Summary
        pause
        exit 1
    }
}
else {
    Write-Message "CUDA bin directory already in PATH." -Type Success
    Add-Result "System PATH" "Success" "Already includes $binPath."
}

# Step 7: Check GCC
Show-Step "7" "Checking GCC Installation"
$gccPath = "gcc"  # Assume in PATH; adjust if needed
try {
    $gccOutput = & $gccPath --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        $versionLine = $gccOutput[0]
        Write-Message "GCC is installed and functional." -Type Success
        Add-Result "GCC Installation" "Success" $versionLine
    }
    else {
        throw "GCC execution failed: $gccOutput"
    }
}
catch {
    Write-Message "GCC not found or not functional: $_" -Type Error
    Add-Result "GCC Installation" "Failed" $_
    Write-Host "`nTutorial: Install MinGW from https://www.mingw-w64.org/. Add the bin directory (e.g., C:\mingw-w64\x86_64-8.1.0-win32-seh-rt_v6-rev0\mingw64\bin) to your system PATH. Verify with 'gcc --version' in a new terminal. Rerun the script." -ForegroundColor Yellow
    Show-Summary
    pause
    exit 1
}

# Display final summary
Show-Summary
pause