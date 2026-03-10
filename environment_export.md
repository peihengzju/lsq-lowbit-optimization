# Environment Export

Generated: 2026-03-06T23:08:03-05:00

## System
Linux PerryLaptop 6.6.87.2-microsoft-standard-WSL2 #1 SMP PREEMPT_DYNAMIC Thu Jun  5 18:30:46 UTC 2025 x86_64 x86_64 x86_64 GNU/Linux

PRETTY_NAME="Ubuntu 24.04.3 LTS"
NAME="Ubuntu"
VERSION_ID="24.04"
VERSION="24.04.3 LTS (Noble Numbat)"
VERSION_CODENAME=noble
ID=ubuntu
ID_LIKE=debian
HOME_URL="https://www.ubuntu.com/"
SUPPORT_URL="https://help.ubuntu.com/"
BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"
PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"
UBUNTU_CODENAME=noble
LOGO=ubuntu-logo

## Python
Python 3.12.3
pip 24.0 from /home/yph3738/projects/cuda_optimization/.venv/lib/python3.12/site-packages/pip (python 3.12)

## PyTorch
torch_version= 2.2.2+cu121
torch_cuda= 12.1
cuda_available= False

## CUDA Toolkit
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Mon_Apr__3_17:16:06_PDT_2023
Cuda compilation tools, release 12.1, V12.1.105
Build cuda_12.1.r12.1/compiler.32688072_0

## Compilers
gcc-12 (Ubuntu 12.4.0-2ubuntu1~24.04.1) 12.4.0
g++-12 (Ubuntu 12.4.0-2ubuntu1~24.04.1) 12.4.0

## CUDA Env Vars
CUDA_HOME=/usr/local/cuda-12.1
PATH=/home/yph3738/.codex/tmp/arg0/codex-arg0fyypv4:/home/yph3738/.vscode-server/bin/0870c2a0c7c0564e7631bfed2675573a94ba4455/bin/remote-cli:/usr/local/cuda-12.1/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/usr/lib/wsl/lib:/mnt/c/Users/yph37/AppData/Local/Programs/Microsoft VS Code:/mnt/c/Program Files/Microsoft/jdk-21.0.8.9-hotspot/bin:/mnt/c/WINDOWS/system32:/mnt/c/WINDOWS:/mnt/c/WINDOWS/System32/Wbem:/mnt/c/WINDOWS/System32/WindowsPowerShell/v1.0/:/mnt/c/WINDOWS/System32/OpenSSH/:/mnt/c/Program Files/MATLAB/R2017a/runtime/win64:/mnt/c/Program Files/MATLAB/R2017a/bin:/mnt/c/Program Files/Git/cmd:/mnt/c/Program Files/dotnet/:/mnt/c/Program Files/NVIDIA Corporation/NVIDIA App/NvDLISR:/mnt/c/Program Files (x86)/NVIDIA Corporation/PhysX/Common:/mnt/c/Program Files (x86)/Arm GNU Toolchain arm-none-eabi/14.3 rel1/bin:/mnt/c/Program Files/qemu:/mnt/c/mingw64/bin:/mnt/c/Program Files/Tailscale/:/mnt/c/Users/yph37/.local/bin:/mnt/c/Users/yph37/AppData/Local/Microsoft/WindowsApps:/mnt/c/Users/yph37/AppData/Local/Programs/Microsoft VS Code/bin:/snap/bin:/home/yph3738/.vscode-server/extensions/openai.chatgpt-26.304.20706-linux-x64/bin/linux-x86_64
LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:/usr/local/cuda/lib64:

## NVIDIA SMI
Fri Mar  6 23:08:05 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.105.07             Driver Version: 581.80         CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3070 ...    On  |   00000000:01:00.0  On |                  N/A |
| N/A   49C    P5             23W /  140W |    1426MiB /   8192MiB |     52%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A              33      G   /Xwayland                             N/A      |
+-----------------------------------------------------------------------------------------+

## Notes
- Full Python dependency lock is saved in requirements.lock.txt
