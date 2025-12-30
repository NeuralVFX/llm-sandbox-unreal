import subprocess
import sys
import os
import unreal

engine_root = unreal.Paths.convert_relative_path_to_full(unreal.Paths.engine_dir())
real_python_exe = os.path.normpath(os.path.join(engine_root, "Binaries", "ThirdParty", "Python3", "Win64", "python.exe"))
script_dir = os.path.dirname(__file__)

project_content = unreal.Paths.project_content_dir()
target_path = os.path.normpath(os.path.join(project_content, "Python", "Lib", "site-packages"))

subprocess.Popen(
    [real_python_exe, os.path.join(script_dir, "run_install.py"), real_python_exe, target_path],
    creationflags=subprocess.CREATE_NEW_CONSOLE
)