# Unreal LLM Sandbox

<p align="center">
  <img src="assets/demo.gif" alt="Demo">
</p>

# This project has two components:
- Web-Interface: [unreal-llm-sandbox](https://github.com/NeuralVFX/unreal-llm-sandbox)
- Unreal Plugin: [unreal-llm-sandbox-plugin](https://github.com/NeuralVFX/unreal-llm-sandbox-plugin)  <--- You are here
  
# What is this?

A web-based coding/llm interface to Unreal Engine, which allows:
- Easy expirimentation with Python
- Review of code and code output/errors via LLM
- Creation and iteration of agentic tools directly from the interface
  
## Features
- **Code Execution** - All Code is executed directly in Unreal Engine
- **LLM Execution** - Ask an LLM for help, with your code/errors in context
- **Agentic Tool Use** - LLM can use tools directly in Unreal Engine
- **User Tools** - Build and register custom agentic tools instantly

# Installation

1. Copy the `LLMSandbox` folder to your project's `Plugins/` directory
2. Restart Unreal Engine
3. Go to `Edit → Plugins`, and Enable thes plugins:
   - `Python Editor Script Plugin`
   - `Python Foundation Packages`
   - `LLM Sandbox`
4. Restart Unreal Engine
5. From the `LLM Sandbox Tools` menu, click `Install Dependencies`
6. Restart Unreal Engine

# Web Interface

Install **[unreal-llm-sandbox](https://github.com/NeuralVFX/unreal-llm-sandbox)** to use `Web Browser` interface

## Usage

### Starting the Server

From the menu bar:
`LLM Sandbox → Start Server`
- The server runs at `http://127.0.0.1:8765`
  
### Starting the Web Interface
- Start `unreal-llm-sandbox` from command line ( outside of Unreal )
- Open `http://localhost:5001/notebook/notebook.ipynb` ( or any `ipynb` name )

The web app provides:
- **Notebook Interface** - Jupyter-style interface
- `Code Cells` - Write and executing Python code in Unreal
- `Markdown Cells` - Write notes in Markdown
- `LLM Prompt Cells` - Chat with LLMs that have full context of your notebook + agentic control of Unreal

# Registering Custom Agent Tools

Syntax to register a new agentic tool:
```python

@register_tool
def spawn_cube(location_x: float, location_y: float, location_z: float):
    """Spawn a cube at the specified world location.
    
    Args:
        location_x: X coordinate
        location_y: Y coordinate  
        location_z: Z coordinate
    
    Returns:
        Name of the spawned actor
    """
    import unreal
    # Your Unreal Python code here
    ...
```
#### Either:
- Run this in a `Code Cell`
- Or create a new python file in your project's `Content/Python/tools` directory
#### Tool Discovery:
- Tools registered in `Code Cells` are instantly availible to the LLM
- Tools added to `Content/Python/tools` are discovered on project restart
#### To use:
- Open a `Prmopt Cell`, Click 🛠️, and write a prompt!

# Requirements

- Unreal Engine 5.6
- A conda env for the web-server

