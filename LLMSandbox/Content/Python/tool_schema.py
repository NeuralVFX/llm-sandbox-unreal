import litellm
    
####################################
# Special Tool Schema
####################################


def patch_schema(schema, patches):
    """
    Merge additional JSON Schema properties into a function tool schema.
    
    Args:
        schema: A tool schema dict from lite_mk_func (has ['function']['parameters']['properties'])
        patches: Dict mapping parameter names to schema additions, e.g. {'actor_paths': {'minItems': 1}}
    
    Returns:
        Modified copy of the schema with patches applied
    """
    properties = schema['function']['parameters']['properties']

    for add_prop in patches:
        if add_prop in properties:
            properties[add_prop].update(patches[add_prop])


def schema_unit_test(schema):
    """
    Validate a tool schema by sending it to the OpenAI API.
    
    Args:
        schema: A tool schema dict (with 'type': 'function' and 'function' keys)
    
    Returns:
        bool: True if schema is valid, False if API rejects it
    """
    try:
        response = litellm.completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hello"}],
            tools=[schema]
        )
        return True
    except Exception as e:
        print(f"Schema Error: {e}")
        return False
