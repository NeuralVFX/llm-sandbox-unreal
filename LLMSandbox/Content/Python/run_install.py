import subprocess
import sys
import os

real_python_exe = sys.argv[1]
target_path = sys.argv[2]

libs_to_install = ["fastcore==1.11.3", "flask", "ipython", "lisette"]

os.makedirs(target_path, exist_ok=True)
print(f"--- Installing to: {target_path} ---")
print("Installing:", ", ".join(libs_to_install))

subprocess.check_call([
    real_python_exe, "-m", "pip", "install",
    "--upgrade",
    "--force-reinstall",
    "--no-cache-dir",
    "--target=" + target_path,
    *libs_to_install,
    "--no-user",
])

print("--- Complete. Restart Unreal Engine. ---")
input("Press Enter to close...")