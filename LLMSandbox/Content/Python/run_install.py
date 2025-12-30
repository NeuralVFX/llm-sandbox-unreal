import subprocess
import sys
import os


real_python_exe = sys.argv[1]
target_path = sys.argv[2]

try:
    os.makedirs(target_path, exist_ok=True)
    libs_to_install = ["flask", "ipython", "lisette"]

    print(f"--- Installing to: {target_path} ---")

    for lib in libs_to_install:
        print(f"Installing {lib}...")
        try:
            subprocess.check_call([
                real_python_exe, '-m', 'pip', 'install',
                '--target=' + target_path,
                lib, '--no-user'
            ])
            print(f"SUCCESS: {lib} installed.")
        except Exception as e:
            print(f"ERROR installing {lib}: {e}")
except Exception as e:
    print(f"Error: {e}")

print("--- Complete. Restart Unreal Engine. ---")
input("Press Enter to close...")