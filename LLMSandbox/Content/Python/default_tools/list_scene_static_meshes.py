import unreal

def _get_static_mesh_from_component(static_mesh_component):
    """
    Return the StaticMesh asset assigned to a StaticMeshComponent, or None.

    Args:
        static_mesh_component (unreal.StaticMeshComponent): Component to query.

    Returns:
        unreal.StaticMesh or None
    """
    if not static_mesh_component:
        return None
    return static_mesh_component.get_editor_property("static_mesh")


def _rotator_to_dict(rot):
    """
    Convert an unreal.Rotator to a JSON-friendly dict.

    Args:
        rot (unreal.Rotator): Rotator to convert.

    Returns:
        dict: {pitch,yaw,roll}
    """
    return {"pitch": float(rot.pitch), "yaw": float(rot.yaw), "roll": float(rot.roll)}


def _vector_to_dict(vec):
    """
    Convert an unreal.Vector to a JSON-friendly dict.

    Args:
        vec (unreal.Vector): Vector to convert.

    Returns:
        dict: {x,y,z}
    """
    return {"x": float(vec.x), "y": float(vec.y), "z": float(vec.z)}


def _transform_to_dict(xform: unreal.Transform):
    t = xform.translation
    q = xform.rotation  # unreal.Quat
    s = xform.scale3d
    return {
        "location": {"x": float(t.x), "y": float(t.y), "z": float(t.z)},
        "quat": {"x": float(q.x), "y": float(q.y), "z": float(q.z), "w": float(q.w)},
        "scale": {"x": float(s.x), "y": float(s.y), "z": float(s.z)},
    }


@register_tool
def get_static_mesh_scene_transforms(actor_name_filter_list=None, include_blueprint_static_mesh_components=True, max_actors=0, selected_only=True):
    """
    Collect world-space transform data for static meshes in the currently loaded editor level.

    Scans the current editor world and returns a dict describing the world position, rotation,
    and scale for each static mesh found.

    Args:
        actor_name_filter_list (list): Optional list of strings. If provided, only actors whose
            label or name contains any of these strings (case-insensitive) are included.
        include_blueprint_static_mesh_components (bool): If True, include StaticMeshComponents found
            on any actor (including Blueprints). If 0, only include StaticMeshActor instances.
        max_actors (int): If > 0, stop after collecting this many actor entries.
        selected_only (bool): If True, limit to only selected actors
    Returns:
        dict: Keyed by actor path name. Each value is a dict with:
            - actor_label (str)
            - actor_name (str)
            - actor_path (str)
            - actor_class (str)
            - actor_world (dict): {location, rotation, scale}
            - static_meshes (list of dicts):
                - component_name (str)
                - component_path (str)
                - mesh_name (str)
                - mesh_path (str)
                - component_world (dict): {location, rotation, scale}
    """
    results = {}

    filters = actor_name_filter_list if isinstance(actor_name_filter_list, list) else []
    filters_lower = [str(s).lower() for s in filters if str(s).strip()]



    # NEW: choose actor source
    if int(selected_only) == 1:
        actors = unreal.EditorLevelLibrary.get_selected_level_actors()
    else:
        actors = unreal.EditorLevelLibrary.get_all_level_actors()    
    
    count = 0

    for actor in actors:
        if max_actors and count >= int(max_actors):
            break

        actor_label = actor.get_actor_label()
        actor_name = actor.get_name()
        actor_path = actor.get_path_name()
        actor_class = actor.get_class().get_name()

        if filters_lower:
            hay = (str(actor_label) + " " + str(actor_name)).lower()
            matched = False
            for f in filters_lower:
                if f in hay:
                    matched = True
                    break
            if not matched:
                continue

        if int(include_blueprint_static_mesh_components) == 0 and (not actor.is_a(unreal.StaticMeshActor)):
            continue

        actor_world_xform = actor.get_actor_transform()

        results[actor_path] = {
            "actor_label": str(actor_label),
            "actor_name": str(actor_name),
            "actor_path": str(actor_path),
            "actor_class": str(actor_class),
            "actor_world": _transform_to_dict(actor_world_xform),
        }
        count += 1

    return results