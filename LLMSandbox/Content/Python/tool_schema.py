import os
import sys
import glob
import copy
import importlib.util
from queue import Queue
from typing import List, Dict, Any, Optional

# Your custom schema generator
from lisette import lite_mk_func 

# Try to import unreal, fail gracefully if testing outside editor
try:
    import unreal
except ImportError:
    unreal = None
    
####################################
# Special Tool Schema
####################################
from typing import List, Dict, Any, Optional
import copy

# Reusable Vector Types
VECTOR3 = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3, "description": "Vector3 [x, y, z]"}
VECTOR4 = {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4, "description": "Vector4 [x, y, z, w]"}

class ToolSchema:
    def __init__(self, func):
        self.func = func
        # Proxy standard function attributes so it looks like the original
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__
        
        # Generate base schema immediately
        self.raw_schema = lite_mk_func(func)
        self.schema = copy.deepcopy(self.raw_schema)

    def __call__(self, *args, **kwargs):
        """Allows the wrapper to be called just like the original function."""
        return self.func(*args, **kwargs)

    def define_list_of_objects(self, param_name: str, item_schema: Dict[str, Any], required_fields: Optional[List[str]] = None):
        """Forces a parameter to be a Typed Array of Objects."""
        props = self.schema['function']['parameters']['properties']
        
        if param_name not in props:
            print(f"Warning: Param '{param_name}' not found in schema.")
            return self

        # Create the object definition
        obj_def = {
            "type": "object",
            "properties": item_schema,
            "required": required_fields if required_fields else list(item_schema.keys())
        }

        # Force Array type
        props[param_name]["type"] = "array"
        props[param_name]["items"] = obj_def
        props[param_name].pop("$ref", None) # Clean up artifacts
        
        return self # Return self to allow chaining or registration

    def set_param_type(self, param_name, type_def):
        """Helper to manually overwrite any simple param type."""
        props = self.schema['function']['parameters']['properties']
        if param_name in props:
            props[param_name] = type_def
        return self

def refine_schema(param_name, item_schema, required_fields=None):
    """
    A decorator that wraps the function in a ToolSchema and applies the fix.
    """
    def decorator(func):
        # 1. Ensure we are working with a ToolSchema wrapper
        if hasattr(func, 'schema'):
            wrapper = func
        else:
            wrapper = ToolSchema(func)
            
        # 2. Apply the fix
        wrapper.define_list_of_objects(
            param_name, 
            item_schema, 
            required_fields
        )
        return wrapper
    return decorator
