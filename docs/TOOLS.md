# LLM Sandbox - Custom Agentic Tools


Agentic tools allow the LLM to actively manipulate the Unreal scene.
Tools can be registered either dynamically at runtime or permanently via a search directory.

# Adding Agentic tools
### Search Directory (Permanent Tools)

- Put a Python file containing your tools inside:
  - `PluginDir/Content/Python/default_tools`, or
  - `ProjectDir/Content/Python/tools`
- Tools in the search directory are loaded automatically
- This is the best option for stable, reusable tools

### Code Cell (Sandbox Tools)

- Execute the tool definition directly in a **Code Cell**
- Tools registered in Code Cells are instantly available to the LLM
- Registration lasts until the server is restarted
- This is the recommended way to prototype and test tools before making them permanent

  
![tool registraion](https://raw.githubusercontent.com/NeuralVFX/llm-sandbox-unreal/main/assets//tool_register.gif)

# Tool Registration

## Simple Tool:
Tool registration defines the callable interface exposed to the LLM.

- The parameters are automatically parsed
- Docstrings are automatically picked up
- Notes on the same line as a parameter declaration are automatically picked up
  
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

## Tool with specific Schema overrides
Schema patches allow you to apply additional constraints to tool parameters.

This is analogous to validating function arguments:
- Enforcing non-empty arrays
- Restricting value ranges
- Narrowing accepted input shapes

These constraints guide the LLM toward valid calls and prevent malformed requests.

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

# Tool Schema
All registered tools are converted into JSON schemas and stored in the global `TOOL_SCHEMAS` variable.

Inspecting this output is useful for:
- Debugging tool registration 
- Verifying parameter constraints
- Understanding how the LLM perceives the tool interface
  
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

