# Unreal LLM Sandbox

<p align="center">
  <img src="assets/demo.gif" alt="Demo">
</p>

#### This project has two components:
- Web-Interface: [unreal-llm-sandbox](https://github.com/NeuralVFX/unreal-llm-sandbox)
- Unreal Plugin: [unreal-llm-sandbox-plugin](https://github.com/NeuralVFX/unreal-llm-sandbox-plugin)  <--- You are here
  
# What is this?

An Unreal Engine plugin that exposes a Python execution server, enabling:
- A web-based coding interface connected directly to Unreal
- LLM-assisted code review with full context of your code and errors
- Creation and deployment of agentic tools that operate inside the engine

## Features
- **Code Execution** - All Code is executed directly in Unreal Engine
- **LLM Execution** - Ask an LLM for help, with your code/errors in context
- **Agentic Tool Use** - LLM can use tools directly in Unreal Engine
- **User Tools** - Build and register custom agentic tools instantly

# Installation

1. Copy the `LLMSandbox` folder to your project's `Plugins/` directory
2. Restart Unreal Engine
3. Go to `Edit → Plugins`, and enable the plugins:
   - `Python Editor Script Plugin`
   - `Python Foundation Packages`
   - `LLM Sandbox`
4. Restart Unreal Engine
5. From the `LLM Sandbox Tools` menu, click `Install Dependencies`
   - Adds required python packages to the project (`flask`, `ipython`, `lisette`)
7. Restart Unreal Engine

# Web Interface

Install **[unreal-llm-sandbox](https://github.com/NeuralVFX/unreal-llm-sandbox)** to use `Web Browser` interface
- The web interface is a standalone Jupyter-style notebook server that connects to Unreal over HTTP.

# Usage

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

## Syntax to register a new agentic tool

#### Simple Tool:
```python
# Simple tool - no patches needed
@register_tool
def spawn_cube(
    x: float,  # X world coordinate
    y: float,  # Y world coordinate
    z: float   # Z world coordinate
):
    """Spawn a cube at the specified location."""
    # Your implementation here
    pass
```

#### Tool with specific Schema overrides: 
```python
# Tool with schema patch - enforces array constraints
ACTOR_PATHS_PATCH = {'type': 'array', 'items': {'type': 'string'}, 'minItems': 1}

@register_tool(patches={'actor_paths': ACTOR_PATHS_PATCH})
def delete_actors(
    actor_paths: List[str]  # List of actor paths to delete
):
    """Delete the specified actors from the level."""
    # Your implementation here
    pass
```

#### Either:
- Run this in a `Code Cell`
- Or create a new python file in your project's `Content/Python/tools` directory
#### Tool Discovery:
- Tools registered in `Code Cells` are instantly availible to the LLM
- Tools added to `Content/Python/tools` are discovered on project restart
#### To use:
- Open a `Prompt Cell`, Click the 🛠️ icon to activate Unreal tools, and write a prompt!

## View the tool Schema
Schemas are stored in a global variable `TOOL_SCHEMAS`, printing it should show something like:
```json
{'type': 'function',
  'function': {'name': 'move_actor_until_hit',
   'description': '\n    Drop actors onto surfaces below (or in any direction).\n    \n    USE THIS WHEN:\n    - User says "drop to floor", "place on ground", "snap to surface"\n    - Aligning objects to terrain or other geometry\n    - User wants objects to "sit" on something naturally\n    \n    Optionally rotates actor to match surface normal (good for organic placement).\n\n    Args:\n        actor_paths:\n            List of Actor UObject paths to raycast from and potentially move.\n\n        distance:\n            Trace length in centimeters.\n\n        buffer_distance:\n            Start the ray from N units behind the object ( usefull in case object is already slightly under terrain )\n\n        direction:\n            World-space direction vector [x,y,z]. Does not need to be normalized.\n\n        set_rotation:\n            If True, apply the computed quaternion to the actor rotation so that local +Z\n            aligns to the hit normal (twist stabilized using the actor\'s Y axis).\n            If False, only the actor location is updated.\n\n    Returns:\n        List[dict]: One entry per actor that was successfully moved (and possibly rotated):\n\n            {\n              "actor_path": str,\n              "actor_label": str,\n              "transform": {\n                  "location": [x,y,z],\n                  "quat": [qx,qy,qz,qw]\n              }\n            }\n    \n\nReturns:\n- type: object',
   'parameters': {'type': 'object',
    'properties': {'actor_paths': {'type': 'array',
      'description': 'REQUIRED. Non-empty list of Actor UObject paths (strings). Never pass an empty list.',
      'items': {'type': 'string'},
      'minItems': 1},
     'distance': {'type': 'number', 'description': '', 'default': 10000},
     'buffer_distance': {'type': 'number',
      'description': '',
...
```


# Requirements

- Unreal Engine 5.6
- A conda env for the web-server

