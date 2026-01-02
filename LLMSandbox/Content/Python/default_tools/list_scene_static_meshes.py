import unreal
import math
import re
import unreal
from typing import Literal, List


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


def _transform_to_dict(xform: unreal.Transform) -> dict:
    t = xform.translation
    q = xform.rotation
    s = xform.scale3d
    return {
        "location": {"x": float(t.x), "y": float(t.y), "z": float(t.z)},
        "quat": {"x": float(q.x), "y": float(q.y), "z": float(q.z), "w": float(q.w)},
        "scale": {"x": float(s.x), "y": float(s.y), "z": float(s.z)},
    }


def _component_world_transform(component) -> unreal.Transform:
    if hasattr(component, "get_component_transform"):
        return component.get_component_transform()
    if hasattr(component, "get_world_transform"):
        return component.get_world_transform()
    return component.get_editor_property("world_transform")


# ----------------------------
# Tools
# ----------------------------

@register_tool
def find_actors(
    name: str = "",
    class_name: str = "",
    class_path: str = "",
    tag: str = "",
    selected_only: bool = False,
    include_hidden: bool = True,
    max_results: int = 500,
    match_mode: str = "contains",
    case_sensitive: bool = False,
    return_format: str = "paths",
) -> list:
    """
    Find actors in the current editor world using common filters.

    Args:
        name: Match against actor label OR object name.
        class_name: Match against actor class name (e.g. "StaticMeshActor").
        class_path: Match against actor class path (e.g. "/Script/Engine.StaticMeshActor").
        tag: Match against actor tags.
        selected_only: If True, search only selected actors.
        include_hidden: If False, exclude actors hidden in editor.
        max_results: Maximum number of results to return (0 means no cap).
        match_mode: "contains" | "equals" | "regex".
        case_sensitive: If False, matching is case-insensitive.
        return_format: "paths" | "labels" | "dict".

    Returns:
        list: Actor identifiers in the requested format.
    """
    actors = (unreal.EditorLevelLibrary.get_selected_level_actors()
              if selected_only else
              unreal.EditorLevelLibrary.get_all_level_actors())

    def norm(s: str) -> str:
        return s if case_sensitive else s.lower()

    def match(hay: str, needle: str) -> bool:
        if needle is None or needle == "":
            return True
        if hay is None:
            return False

        if match_mode == "regex":
            flags = 0 if case_sensitive else re.IGNORECASE
            return re.search(needle, hay, flags=flags) is not None

        hay_n = norm(hay)
        needle_n = norm(needle)

        if match_mode == "equals":
            return hay_n == needle_n
        # default: contains
        return needle_n in hay_n

    out = []
    for a in actors:
        if not a:
            continue

        if not include_hidden and a.is_hidden_ed():
            continue

        label = str(a.get_actor_label())
        obj_name = str(a.get_name())
        path = str(a.get_path_name())

        if name and not (match(label, name) or match(obj_name, name)):
            continue

        cls = a.get_class()
        cls_name_str = str(cls.get_name()) if cls else ""
        cls_path_str = str(cls.get_path_name()) if cls else ""

        if class_name and not match(cls_name_str, class_name):
            continue
        if class_path and not match(cls_path_str, class_path):
            continue

        if tag:
            tags = [str(t) for t in getattr(a, "tags", [])]
            if not any(match(t, tag) for t in tags):
                continue

        if return_format == "paths":
            out.append(path)
        elif return_format == "labels":
            out.append(label)
        else:
            out.append({
                "actor_path": path,
                "actor_label": label,
                "actor_name": obj_name,
                "actor_class": cls_name_str,
            })

        if max_results and len(out) >= int(max_results):
            break

    return out


@register_tool
def get_static_mesh_scene_transforms(
    include_blueprint_static_mesh_components: bool,
    selected_only: bool,
    actor_paths: list[str] = [],

) -> dict:
    """
    Retrieves world-space transform data for actors with Static Mesh Components.
    
    LOGIC HIERARCHY:
    1. If 'actor_paths' is provided (not empty), the tool ONLY processes those specific actors.
    2. If 'actor_paths' is empty, the tool searches the level based on 'selected_only'.
    
    Args:
        actor_paths: List of full Actor Path Names (e.g., ["PersistentLevel.Chair_2"]). 
                     Leave empty if you want to search based on selection or the whole level.
        include_blueprint_static_mesh_components: 
                     If True, finds meshes inside Blueprints/Classes. 
                     If False, only finds meshes on standard StaticMeshActors.
        selected_only: Only used if 'actor_paths' is empty. 
                       True = Process only selected actors. 
                       False = Process all actors in the level.
        include_actor_world: Whether to include the root Actor's world transform in the output.

    Returns:
            dict: A dictionary where keys are Actor Paths and values are metadata.
            
            Example Return Structure:
            {
                "PersistentLevel.Chair_22": {
                    "actor_label": "Office_Chair",
                    "is_selected": True,
                    "actor_class": "StaticMeshActor",
                    "actor_world": {
                        "translation": [100.0, 0.0, 0.0],
                        "rotation": [0.0, 0.0, 0.0, 1.0],
                        "scale": [1.0, 1.0, 1.0]
                    },
                    "static_meshes": [
                        {
                            "component_name": "CustomMesh",
                            "mesh_path": "/Game/Assets/ChairMesh.ChairMesh",
                            "component_world": {...}
                        }
                    ]
                }
            }
    """
    results = {}

    # Handle Input Logic
    if actor_paths and isinstance(actor_paths, list) and len(actor_paths) > 0:
        actors = []
        for p in actor_paths:
            # Note: unreal.find_object is safer for world actors than load_object
            a = unreal.find_object(None, p)
            if a:
                actors.append(a)
    else:
        actors = (unreal.EditorLevelLibrary.get_selected_level_actors()
                  if selected_only else
                  unreal.EditorLevelLibrary.get_all_level_actors())

    # Get the current selection set to compare against for 'is_selected'
    current_selection = unreal.EditorLevelLibrary.get_selected_level_actors()
    selection_paths = {a.get_path_name() for a in current_selection if a}

    count = 0
    for actor in actors:
        if not actor:
            continue

        # Class Filter
        if not include_blueprint_static_mesh_components and (not actor.is_a(unreal.StaticMeshActor)):
            continue

        actor_path = str(actor.get_path_name())
        
        # Core Actor Info
        entry = {
            "actor_label": str(actor.get_actor_label()),
            "actor_name": str(actor.get_name()),
            "actor_path": actor_path,
            "actor_class": str(actor.get_class().get_name()),
            "is_selected": actor_path in selection_paths, # <-- NEW SELECTION TRACKING
            "static_meshes": []
        }

        entry["actor_world"] = _transform_to_dict(actor.get_actor_transform())

        # Component Scouring
        comps = actor.get_components_by_class(unreal.StaticMeshComponent)
        for c in comps:
            if not c: continue
            sm = _get_static_mesh_from_component(c)
            comp_world = _component_world_transform(c)

            entry["static_meshes"].append({
                "component_name": str(c.get_name()),
                "component_path": str(c.get_path_name()),
                "mesh_name": str(sm.get_name()) if sm else "",
                "mesh_path": str(sm.get_path_name()) if sm else "",
                "component_world": _transform_to_dict(comp_world),
            })

        results[actor_path] = entry
        count += 1

    return results
    


@register_tool
def destroy_actors(
    actor_paths: List[str] = []
) -> dict:
    """
    Permanently deletes multiple actors from the world using their full path names.
    
    Args:
        actor_paths: A list of full path strings (e.g., ["PersistentLevel.Sphere_0"]).
    
    Returns:
        dict: Summary of the operation, including 'success_count' and 'failed_paths'.
        Example Return:
        {
            "success_count": 38,
            "failed_count": 2,
            "failed_paths": ["PersistentLevel.Invalid_Actor_99"]
        }
    """
    success_count = 0
    failed_paths = []

    for path in actor_paths:
        try:
            # 1. Resolve the actor in the world
            actor = unreal.find_object(None, path)
            
            if actor:
                # 2. Use the Editor library to ensure it handles undo/redo correctly
                success = unreal.EditorLevelLibrary.destroy_actor(actor)
                if success:
                    success_count += 1
                else:
                    failed_paths.append(path)
            else:
                failed_paths.append(path)
                
        except Exception as e:
            unreal.log_warning(f"Failed to destroy {path}: {str(e)}")
            failed_paths.append(path)

    return {
        "success_count": success_count,
        "failed_count": len(failed_paths),
        "failed_paths": failed_paths
    }