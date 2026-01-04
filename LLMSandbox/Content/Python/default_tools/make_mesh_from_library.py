from __future__ import annotations

import unreal
from typing import List, Dict
from typing import TypedDict, List, Tuple, Literal, NotRequired
from typing import Annotated, List, Dict, Any
from pydantic import BaseModel, Field
import numpy as np

# ----------------------------
# Tool
# ----------------------------


@register_tool
@refine_schema("actor_transforms", item_schema={
    "asset_path": {"type": "string"},
    "location": VECTOR3,
    "quaternion": VECTOR4,
    "scale": VECTOR3,
})
def spawn_actors(
    actor_transforms: list,
    base_name: str,
) -> List[str]:
    """
    Spawns multiple actors in a single execution using Quaternion rotation.
    
    Args:
        class_or_asset_path: Full asset path to spawn.
        transforms: List of dicts with:
         {
          'asset_path': library_asset_path,
            'location': [x,y,z], 
            'quaternion': [x,y,z,w],
            'scale': [x,y,z]
            }
        base_name: Prefix for the actor labels.


    Returns:
            List[str]: A list of the full path names for every successfully spawned actor.
            Example: ["PersistentLevel.BatchActor_0", "PersistentLevel.BatchActor_1"]
    """

    spawned = []
    for i, xf in enumerate(actor_transforms):

        class_or_asset_path= xf.get('asset_path')

        # 1. Try to load as a Class first (for Lights/Cameras)
        spawn_obj = unreal.load_class(None, class_or_asset_path)
        
        # 2. If not a class, try to load as an Asset (for Meshes)
        if not spawn_obj:
            spawn_obj = unreal.EditorAssetLibrary.load_asset(class_or_asset_path)

        if not spawn_obj:
            unreal.log_error(f"Could not find Class or Asset at: {class_or_asset_path}")
            continue

        loc = unreal.Vector(*xf.get('location', [0,0,0]))
        quat = unreal.Quat(*xf.get('quaternion', [0,0,0,1]))
        scale = unreal.Vector(*xf.get('scale', [1,1,1]))

        # This API handles both Classes and Assets correctly
        actor = unreal.EditorLevelLibrary.spawn_actor_from_object(spawn_obj, loc, quat.rotator())
        
        if actor:
            label = f"{base_name}_{i}"
            actor.set_actor_label(label)
            actor.set_actor_scale3d(scale)
            spawned.append({
                "actor_path": actor.get_path_name(),
                "actor_class": actor.get_class().get_name(),
            })
            
    return spawned


    
@register_tool
@refine_schema("actor_transform_updates", item_schema={
    "actor_path": {"type": "string"},
    "location": VECTOR3,
    "quaternion": VECTOR4,
    "scale": VECTOR3
})
def update_actors_transforms(
    actor_transform_updates: list
) -> List[str]:
    """
    Updates the location, rotation, and scale of multiple existing actors in one batch.
    
    This is the most efficient way to move groups of objects or align them to patterns.

        [
        {
            "actor_path": "PersistentLevel.Chair_2",
            "location": [100.0, 0.0, 50.0],
            "quat": [0.0, 0.0, 0.0, 1.0],
            "scale": [1.0, 1.0, 1.0]
        },
        {
            "actor_path": "PersistentLevel.Table_5",
            "location": [200.0, 10.0, 55.0],
            "quat": [0.0, 0.707, 0.0, 0.707],
            "scale": [1.2, 1.2, 1.2]
        }
        ]

    Args:
        updates: A list of dictionaries. Each MUST contain 'actor_path' (str), 
                 'location' [x,y,z], 'quat' [x,y,z,w], and 'scale' [x,y,z].

    Returns:
        Dict: Return Status, and  list of actor paths that were successfully updated.
    """

    #out_dict = {'return_status':"Failure: You must fill 'updates' param ie:\n "+UPDATES_EX ,'updated_paths':[]}
    updated_paths = []
    for i, xf in enumerate(actor_transform_updates):
        try:
            # 1. Resolve the Actor
            path = xf.get('actor_path', "")
            actor = unreal.find_object(None, path)
            
            if not actor:
                unreal.log_warning(f"Batch Update: Could not find actor at {path}")
                continue

            # 2. Extract and cast data
            l_vals = xf.get('location', [0.0, 0.0, 0.0])
            q_vals = xf.get('quaternion', [0.0, 0.0, 0.0, 1.0])
            s_vals = xf.get('scale', [1.0, 1.0, 1.0])

            loc = unreal.Vector(float(l_vals[0]), float(l_vals[1]), float(l_vals[2]))
            quat = unreal.Quat(float(q_vals[0]), float(q_vals[1]), float(q_vals[2]), float(q_vals[3]))
            scale = unreal.Vector(float(s_vals[0]), float(s_vals[1]), float(s_vals[2]))
            
            # 3. Apply Transform
            # Using set_actor_transform is cleaner for batching all three properties at once
            new_transform = unreal.Transform(loc, quat.rotator(), scale)
            actor.set_actor_transform(new_transform, False, True)
            
            updated_paths.append(path)
                
        except Exception as e:
            unreal.log_error(f"Failed to update batch item {i}: {str(e)}")
            continue
            
    return updated_paths

from typing import List, Dict


@register_tool
def space_apart_intersecting_actors(
    actor_paths: List[str],

) -> str:
    """
    Spaces out overlapping actors using a physics-like relaxation solver.
    
    USE THIS WHEN:
    - User says "separate these", "they're overlapping", "spread apart"
    - After scatter/duplication creates intersecting objects
    - Fixing clipping/z-fighting between nearby actors
    
    Args:
        actor_paths: List of full actor paths to de-overlap.
        
    Returns:
        List[dict]: Each moved actor with actor_path, actor_label, and new_location.
    """
    iterations = 50
    learning_rate = 0.5

    # 1. Resolve Paths to Actor Objects
    actors = []
    for path in actor_paths:
        actor = unreal.find_object(None, path)
        if actor:
            actors.append(actor)
            
    count = len(actors)
    if count < 2: 
        return "Skipped: Need at least 2 valid actors."

    # --- 2. PREPARE DATA ---
    positions = []
    radii = []
    
    for a in actors:
        pos = a.get_actor_location()
        positions.append([pos.x, pos.y, pos.z])
        
        origin, extent = a.get_actor_bounds(False) 
        avg_radius = (extent.x + extent.y + extent.z) / 3.0
        radii.append(avg_radius)

    P = np.array(positions)
    P += np.random.uniform(-0.1, 0.1, P.shape)  # now this works
    R = np.array(radii).reshape(-1, 1)

    # Helper to calculate max overlap for reporting
    def get_max_overlap(Pos, Rad):
        diff = Pos[:, np.newaxis, :] - Pos[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        overlap = (Rad + Rad.T) - dist
        return np.max(np.maximum(overlap, 0))

    # Calculate Starting Stats
    start_overlap = get_max_overlap(P, R)
    unreal.log(f"Starting Relaxation. Max Overlap: {start_overlap:.2f}")

    # --- 3. OPTIMIZATION LOOP ---
    final_overlap = start_overlap
    
    for i in range(iterations):
        # Pairwise differences and distances
        diff_matrix = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff_matrix, axis=2)
        np.fill_diagonal(dist_matrix, np.inf)

        # Overlap calculation
        req_dist_matrix = R + R.T
        overlap = np.maximum(req_dist_matrix - dist_matrix, 0)
        
        final_overlap = np.max(overlap)
        if final_overlap < 0.1:
            break
            
        # Compute forces
        directions = diff_matrix / (dist_matrix[:, :, np.newaxis] + 1e-6)
        forces = directions * overlap[:, :, np.newaxis]
        total_forces = np.sum(forces, axis=1)
        
        # Raw step
        step = total_forces * learning_rate
        
        # HARD CLAMP: magnitude can never exceed 1/5 of object's radius
        step_magnitude = np.linalg.norm(step, axis=1, keepdims=True)
        max_allowed = R * 0.1
        
        # Normalize direction, then scale to clamped magnitude
        step_direction = step / (step_magnitude + 1e-9)
        clamped_magnitude = np.minimum(step_magnitude, max_allowed)
        step = step_direction * clamped_magnitude
        
        P += step

    actors_moved = []

    # --- 4. APPLY & REPORT ---
    with unreal.ScopedEditorTransaction("Spacing Relaxation"):
        for i, actor in enumerate(actors):
            new_loc = [float(P[i][0]), float(P[i][1]), float(P[i][2])]
            

            actor.set_actor_location(unreal.Vector(*new_loc), False, True)
            
            actors_moved.append({
                "actor_path": actor.get_path_name(),
                "actor_label": actor.get_actor_label(),
                "new_location": [round(v, 2) for v in new_loc],
            })
    
    

    return actors_moved