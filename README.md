# LLM Sandbox

A plugin for Unreal Engine that provides a Flask server for LLM-powered code execution directly in the editor.

## Features

- **Flask API Server** — Execute Python code and tools via HTTP endpoints
- **IPython Integration** — Full interactive shell with history and rich tracebacks
- **Tool System** — Register custom tools that LLMs can discover and call
- **Main Thread Execution** — All Unreal API calls run safely on the main thread
- **Hot-Reloadable Tools** — Add tools to your project without modifying the plugin

## Installation

1. Copy the `LLMSandbox` folder to your project's `Plugins/` directory
2. Restart Unreal Engine
3. Go to **Edit → Plugins** and verify "LLM Sandbox" is enabled
4. From the **LLM Sandbox** menu, click **Install Dependencies**
5. Restart Unreal Engine

### Web Interface

Install **[unreal-llm-sandbox](https://github.com/NeuralVFX/unreal-llm-sandbox)** to use `Web Browser` interface

## Usage

### Starting the Server

From the menu bar:
`LLM Sandbox → Start Server`
- The server runs at `http://127.0.0.1:8765`
  
### Starting the Web Iterface
- Start `unreal-llm-sandbox` from command line ( outside of Unreal )
- Open `http://localhost:5001/notebook/notebook.ipynb` ( or any `ipynb` name )

The web app provides:
- **Notebook Interface** — Jupyter-style cells for writing and executing Python code in Unreal
- **LLM Prompt Cells** — Chat with LLMs that have full context of your notebook + agentic control of Unreal
- **Agent Cells** — Agentic code generation with automatic unit testing and iteration
- **Live Streaming** — Real-time output from code execution and LLM responses

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

- Unreal Engine 5.4+
- Python 3.11+ (bundled with UE)

## License

MIT
