# LLM Sandbox — Custom Agentic Tools

This document explains how to register custom tools that an LLM can invoke inside Unreal Engine.

Tools can be registered in two ways:
- By placing Python files in a discovery directory (persistent tools)
- By registering tools dynamically from a notebook code cell (temporary tools)

In both cases, the `@register_tool` decorator is required for the LLM to see the tool.

---

## Registering Tools via Search Directories (Persistent)

Place a Python file containing your tools in one of the following locations:

- `PluginDir/Content/Python/default_tools`
- `ProjectDir/Content/Python/tools`

Tools in these directories:
- Are discovered when the Unreal server starts
- Persist across sessions
- Are suitable for production or shared tools

---

## Registering Tools from a Code Cell (Temporary)

Tools can also be registered by executing code in a **Code Cell**.

- Tools become available to the LLM immediately
- Registration lasts until the Unreal server is restarted
- This is the recommended way to prototype and test tools

---

## Syntax to Register

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

## To use:
- Open a `Prompt Cell`, Click the 🛠️ icon to activate Unreal tools, and write a prompt!

## View the tool Schema
Schemas are stored in a global variable `TOOL_SCHEMAS`, printing it should show something like:
```python
[{'type': 'function',
  'function': {'name': 'move_actor_until_hit',
   'description': '\n    Drop actors onto surfaces below (or in any direction).\n ',
   'parameters': {'type': 'object',
    'properties': {'actor_paths': {'type': 'array',
      'description': 'REQUIRED. Non-empty list of Actor UObject paths (strings). Never pass an empty list.',
      'items': {'type': 'string'},
      'minItems': 1},
     'distance': {'type': 'number', 'description': '', 'default': 10000},
     'buffer_distance': {'type': 'number',
      'description': '',
```

