# Unreal LLM Sandbox
<p align="center">
  <img src="assets/demo.gif" alt="Demo">
</p>

## What is this?
A web interface to Unreal Engine, which allows:
- Easy expirimentation with Python
- Review of code and cout output/erros via LLM
- Creation and iteration of agentic tools directly from the interface
## Is the toolset complete?
A small set of working agentic tools is provided with this, but thats not the point. The idea is that this tool helps you easily build whatever custom agentic tools you need for your department, on your production! With feedback directly from the LLM along the way.

## Features

- **Code Execution** - All Code is executed directly in Unreal Engine
- **LLM Execution** - Ask an LLM for help, with your code/errros in context
- **Agentic Tool Use** - LLM can use tools directly in Unreal Engine
- **User Tools** - Build and register custom agentic tools

## Installation

1. Copy the `LLMSandbox` folder to your project's `Plugins/` directory
2. Restart Unreal Engine
3. Go to `Edit → Plugins`, and Enable thes plugins:
   - `Python Editor Script Plugin`
   - `Python Foundation Packages`
   - `LLM Sandbox`
4. Restart Unreal Engine
5. From the `LLM Sandbox Tools` menu, click `Install Dependencies`
6. Restart Unreal Engine

### Web Interface

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
- `Agent Cells` - Agentic code generation with automatic unit testing and iteration

### Registering Custom Agent Tools

Create a new python file in your project's `Content/Python/tools` directory:

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
Tools are automatically discovered by LLM, and may be used via a `Prompt Cell`

## Requirements

- Unreal Engine 5.6
- Python 3.11+ (bundled with UE)

## License

MIT
