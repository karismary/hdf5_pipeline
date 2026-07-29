import subprocess, platform

def pick_folder() -> str:
    """跨平台文件夹选择器，返回选中路径或空字符串。

    调用系统原生的文件夹选择弹窗，不依赖任何 GUI 框架。
    支持 macOS（osascript）、Windows（PowerShell）、Linux（zenity）。

    Returns:
        str: 选中的文件夹绝对路径。用户取消时返回空字符串。
    """
    s = platform.system()

    if s == "Darwin":
        cmd = ["osascript", "-e", 'POSIX path of (choose folder)']
    elif s == "Windows":
        cmd = ["powershell", "-Command",
               'Add-Type -AssemblyName System.Windows.Forms;'
               '$f=New-Object System.Windows.Forms.FolderBrowserDialog;'
               'if($f.ShowDialog()-eq"OK"){$f.SelectedPath}']
    else:
        cmd = ["zenity", "--file-selection", "--directory"]

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout.strip():
        return

    folder = r.stdout.strip()

    return folder
