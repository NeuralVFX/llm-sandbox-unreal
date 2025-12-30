import sys
import os
import unreal

project_content = unreal.Paths.project_content_dir()
site_packages = os.path.normpath(os.path.join(project_content, "Python", "Lib", "site-packages"))
if site_packages not in sys.path:
    sys.path.insert(0, site_packages)

def build_menu():
    menus = unreal.ToolMenus.get()
    main_menu = menus.find_menu("LevelEditor.MainMenu")
    
    sandbox_menu = main_menu.add_sub_menu("LLMSandbox",
                                          "LLM Sandbox",
                                          "LLM Sandbox",
                                          "LLM Sandbox Tools")
    
    # Start Server
    entry_start = unreal.ToolMenuEntry(type=unreal.MultiBlockType.MENU_ENTRY)
    entry_start.set_label("Start Server")
    entry_start.set_string_command(unreal.ToolMenuStringCommandType.PYTHON, "",
                                 "from server import register_callback; register_callback()")

    sandbox_menu.add_menu_entry("StartServer", entry_start)
    
    # Stop Server
    entry_stop = unreal.ToolMenuEntry(type=unreal.MultiBlockType.MENU_ENTRY)
    entry_stop.set_label("Stop Server")
    entry_stop.set_string_command(unreal.ToolMenuStringCommandType.PYTHON, "",
                                     "from server import cleanup; cleanup()")

    sandbox_menu.add_menu_entry("StopServer", entry_stop)
    
    # Install Dependencies
    entry_install = unreal.ToolMenuEntry(type=unreal.MultiBlockType.MENU_ENTRY)
    entry_install.set_label("Install Dependencies")
    entry_install.set_string_command(unreal.ToolMenuStringCommandType.PYTHON, "",
                 "from install_deps import install_sandbox_dependencies; install_sandbox_dependencies()")

    sandbox_menu.add_menu_entry("InstallDeps", entry_install)
    
    menus.refresh_all_widgets()

build_menu()
unreal.log("LLM Sandbox menu registered")
