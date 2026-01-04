import unreal
import math
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import random

# ----------------------------
# Vector / Math helpers
# ----------------------------


def _resolve_actor_paths_or_selected(
    actor_paths: Optional[List[str]] = None,
    selected_only: bool = False
) -> List[unreal.Actor]:
    """
    Resolve actors from paths, or fall back to selection/all actors.
    
    Args:
        actor_paths: If provided, resolve these specific paths.
        selected_only: When actor_paths is empty:
                       True = selected actors, False = all actors.
    """
    if actor_paths:
        actors = []
        for p in actor_paths:
            a = unreal.find_object(None, p)
            if a and isinstance(a, unreal.Actor):
                actors.append(a)
        return actors
    
    if selected_only:
        return [a for a in unreal.EditorLevelLibrary.get_selected_level_actors() if a]
    else:
        return [a for a in unreal.EditorLevelLibrary.get_all_level_actors() if a]


def _v3_to_np(v: unreal.Vector) -> np.ndarray:
    return np.array([v.x, v.y, v.z], dtype=np.float64)


def _camera_basis_np(cam_rot: unreal.Rotator):
    fwd = _v3_to_np(unreal.MathLibrary.get_forward_vector(cam_rot))
    rgt = _v3_to_np(unreal.MathLibrary.get_right_vector(cam_rot))
    up  = _v3_to_np(unreal.MathLibrary.get_up_vector(cam_rot))
    return fwd, rgt, up


def _as_vec3(v, default=None) -> unreal.Vector:
    if default is None:
        default = unreal.Vector(0.0, 0.0, 0.0)
    if isinstance(v, unreal.Vector):
        return v
    if v is None:
        return default
    if isinstance(v, (list, tuple)) and len(v) >= 3:
        return unreal.Vector(float(v[0]), float(v[1]), float(v[2]))
    return default


def _dot(a: unreal.Vector, b: unreal.Vector) -> float:
    return float(a.x * b.x + a.y * b.y + a.z * b.z)


def _cross(a: unreal.Vector, b: unreal.Vector) -> unreal.Vector:
    return unreal.Vector(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    )


def _len(v: unreal.Vector) -> float:
    return math.sqrt(_dot(v, v))


def _safe_normal(v: unreal.Vector, fallback: unreal.Vector) -> unreal.Vector:
    l = _len(v)
    if l < 1e-6:
        return fallback
    inv = 1.0 / l
    return unreal.Vector(v.x * inv, v.y * inv, v.z * inv)


def _build_quat_from_normal(normal: unreal.Vector, up_hint: unreal.Vector) -> unreal.Quat:
    z_axis = _safe_normal(normal, unreal.Vector(0.0, 0.0, 1.0))
    upn = _safe_normal(up_hint, unreal.Vector(0.0, 1.0, 0.0))

    if abs(_dot(upn, z_axis)) > 0.98:
        upn = unreal.Vector(0.0, 1.0, 0.0)
        if abs(_dot(upn, z_axis)) > 0.98:
            upn = unreal.Vector(1.0, 0.0, 0.0)

    x_axis = _safe_normal(_cross(upn, z_axis), unreal.Vector(1.0, 0.0, 0.0))
    y_axis = _safe_normal(_cross(z_axis, x_axis), unreal.Vector(0.0, 1.0, 0.0))

    rot = unreal.MathLibrary.make_rotation_from_axes(x_axis, y_axis, z_axis)
    return rot.quaternion()


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


def _component_world_transform(component) -> unreal.Transform:
    if hasattr(component, "get_component_transform"):
        return component.get_component_transform()
    if hasattr(component, "get_world_transform"):
        return component.get_world_transform()
    return component.get_editor_property("world_transform")

# ----------------------------
# HitResult parsing (UE 5.6 Python binding: use to_tuple())
# Observed tuple layout (len=18):
# 0 blocking_hit(bool)
# 4 location(Vector)
# 5 impact_point(Vector)
# 6 normal(Vector)
# 7 impact_normal(Vector)
# 9 actor(Actor)
# 10 component(Component)
# 16 trace_start(Vector)
# 17 trace_end(Vector)
# ----------------------------

def _hitresult_tuple(hit) -> Optional[tuple]:
    if hit is None:
        return None
    if hasattr(hit, "to_tuple"):
        return hit.to_tuple()
    return None

def _hitresult_blocking(hit) -> bool:
    t = _hitresult_tuple(hit)
    if not t or len(t) < 1:
        return False
    return bool(t[0])

def _hitresult_location(hit) -> Optional[unreal.Vector]:
    t = _hitresult_tuple(hit)
    if not t:
        return None
    for idx in (4, 5):
        if len(t) > idx and isinstance(t[idx], unreal.Vector):
            return t[idx]
    return None

def _hitresult_normal(hit) -> Optional[unreal.Vector]:
    t = _hitresult_tuple(hit)
    if not t:
        return None
    for idx in (6, 7):
        if len(t) > idx and isinstance(t[idx], unreal.Vector):
            return t[idx]
    return None

def _hitresult_actor(hit):
    t = _hitresult_tuple(hit)
    if not t:
        return None
    if len(t) > 9 and isinstance(t[9], unreal.Actor):
        return t[9]
    if len(t) > 10 and t[10] and hasattr(t[10], "get_owner"):
        return t[10].get_owner()
    return None

# ----------------------------
# Collision / trace helpers
# ----------------------------

def _sanitize_ignore_actors(ignore_actors) -> List[unreal.Actor]:
    clean = []
    if not ignore_actors:
        return clean
    for a in ignore_actors:
        if a and isinstance(a, unreal.Actor):
            clean.append(a)
    return clean

def _resolve_trace_type_query(trace_channel: str):
    tc = (trace_channel or "Visibility").strip().lower()

    cc = None
    if tc == "camera":
        cc = getattr(unreal.CollisionChannel, "ECC_CAMERA", None)
    elif tc == "worldstatic":
        cc = getattr(unreal.CollisionChannel, "ECC_WORLD_STATIC", None)
    elif tc == "worlddynamic":
        cc = getattr(unreal.CollisionChannel, "ECC_WORLD_DYNAMIC", None)
    else:
        cc = getattr(unreal.CollisionChannel, "ECC_VISIBILITY", None)

    if cc is not None and hasattr(unreal.SystemLibrary, "convert_to_trace_type"):
        return unreal.SystemLibrary.convert_to_trace_type(cc)

    if hasattr(unreal.TraceTypeQuery, "TRACE_TYPE_QUERY1"):
        return unreal.TraceTypeQuery.TRACE_TYPE_QUERY1

    for name in dir(unreal.TraceTypeQuery):
        if name.startswith("TRACE_TYPE_QUERY"):
            return getattr(unreal.TraceTypeQuery, name)

    return None

def _force_collision_refresh(actor: unreal.Actor) -> bool:
    if not actor:
        return False
    smc = actor.get_component_by_class(unreal.StaticMeshComponent)
    if not smc:
        return False

    smc.set_collision_profile_name("BlockAll")
    smc.set_collision_enabled(unreal.CollisionEnabled.QUERY_ONLY)
    if hasattr(smc, "set_generate_overlap_events"):
        smc.set_generate_overlap_events(False)

    if hasattr(smc, "recreate_physics_state"):
        smc.recreate_physics_state()
    if hasattr(smc, "reregister_component"):
        smc.reregister_component()
    if hasattr(smc, "update_bounds"):
        smc.update_bounds()
    if hasattr(smc, "mark_render_state_dirty"):
        smc.mark_render_state_dirty()

    return True

# ----------------------------
# Main tool
# ----------------------------

def ray_cast_array(
    rays: List[dict] = [],
    ignore_actor_paths: List[str] = [],
    distance: float = 10000,
    direction:List[float] = [0,0,-1],
) -> List[dict]:
    """
    Batch line-trace (raycast) in the current Unreal **Editor World** and return a
    per-ray hit result plus a transform (location + quaternion) at the hit point.

    Notes (UE 5.6 Python):
        `unreal.SystemLibrary.line_trace_single` returns a `HitResult` whose fields are not
        readable as normal Python attributes in this environment. This function extracts
        hit data via `HitResult.to_tuple()` using helper functions.

    Args:
        rays (List[dict]):
            List of ray specs. Each ray dict supports:

            Required keys:
                - "start": [x, y, z]
                    World-space start position in centimeters.

        direction:List[float] : [x, y, z]
            World-space direction vector. Does not need to be normalized.

        distance : float
            Trace length in centimeters.

        ignore_actor_paths (List[str]):
            List of Actor UObject paths to ignore during the trace. Each entry should be a
            valid actor path string (e.g. "/Game/Map.Map:PersistentLevel.Actor_12").
            These paths are resolved to `unreal.Actor` objects and passed to the trace as
            `actors_to_ignore`.

    Returns:
        List[dict]: One result dict per input ray, in the same order:

            {
              "hit": bool,
              "hit_location": [x, y, z] or None,
              "hit_normal": [x, y, z] or None,
              "hit_actor_path": str or None,
              "transform": {
                  "location": [x, y, z],
                  "quat": [qx, qy, qz, qw]
              } or None
            }

            - If "hit" is False, all hit fields and "transform" will be None.
            - If "hit" is True, "transform.location" matches "hit_location", and
              "transform.quat" aligns local +Z to the surface normal (twist stabilized by
              "up_hint").
    """
    debug_lifetime = 2.0
    draw_debug = True
    trace_complex = True

    if rays is None:
        rays = []

    world = unreal.EditorLevelLibrary.get_editor_world()
    tquery = _resolve_trace_type_query("Visibility")

    ignore_actors = []
    for p in ignore_actor_paths:
        a = unreal.find_object(None, p)
        if a and isinstance(a, unreal.Actor):
            ignore_actors.append(a)
    ignore_actors = _sanitize_ignore_actors(ignore_actors)

    draw_type = unreal.DrawDebugTrace.NONE
    if draw_debug:
        draw_type = unreal.DrawDebugTrace.FOR_DURATION

    results = []
    for r in rays:
        start = _as_vec3(r.get("start", [0.0, 0.0, 0.0]))
        up_hint = unreal.Vector(0.0, 1.0, 0.0)
        dir_vec = _as_vec3(direction, unreal.Vector(0,0,-1))
        dirn = _safe_normal(dir_vec, unreal.Vector(0,0,-1))
        end = start + (dirn * distance)

        out = {
            "hit": False,
            "hit_location": None,
            "hit_normal": None,
            #"hit_actor_path": None,
            #"transform": None
        }

        if world is None or tquery is None:
            results.append(out)
            continue

        hit = unreal.SystemLibrary.line_trace_single(
            world,
            start,
            end,
            tquery,
            bool(trace_complex),
            ignore_actors,
            draw_type,
            False,
            unreal.LinearColor(1.0, 0.0, 0.0, 1.0),
            unreal.LinearColor(0.0, 1.0, 0.0, 1.0),
            float(debug_lifetime)
        )

        if hit is None:
            results.append(out)
            continue

        if not _hitresult_blocking(hit):
            results.append(out)
            continue

        loc = _hitresult_location(hit)
        nrm = _hitresult_normal(hit)
        #actor_obj = _hitresult_actor(hit)

        if loc is None or nrm is None:
            results.append(out)
            continue

       # q = _build_quat_from_normal(nrm, up_hint)
        #actor_path = actor_obj.get_path_name() if actor_obj else None

        out["hit"] = True
        out["hit_location"] = [float(loc.x), float(loc.y), float(loc.z)]
        out["hit_normal"] = [float(nrm.x), float(nrm.y), float(nrm.z)]
        #out["hit_actor_path"] = actor_path
        #out["transform"] = {
        #    "location": [float(loc.x), float(loc.y), float(loc.z)],
        #    "quat": [float(q.x), float(q.y), float(q.z), float(q.w)]
        #}

        results.append(out)

    return results


@register_tool
def move_actor_until_hit(
    actor_paths: List[str],
    selected_only: bool = False, 
    distance: float = 10000,
    buffer_distance: float = 5000.0,
    direction: List[float] = [0, 0, -1],
    set_rotation: bool = True, 
) -> List[dict]:
    """
    Drop actors onto surfaces below (or in any direction).
    
    USE THIS WHEN:
    - User says "drop to floor", "place on ground", "snap to surface"
    - Aligning objects to terrain or other geometry
    - User wants objects to "sit" on something naturally
    
    Optionally rotates actor to match surface normal (good for organic placement).

    Args:
        actor_paths:
            List of Actor UObject paths to raycast from and potentially move.

        distance:
            Trace length in centimeters.

        buffer_distance:
            Start the ray from N units behind the object ( usefull in case object is already slightly under terrain )

        direction:
            World-space direction vector [x,y,z]. Does not need to be normalized.

        set_rotation:
            If True, apply the computed quaternion to the actor rotation so that local +Z
            aligns to the hit normal (twist stabilized using the actor's Y axis).
            If False, only the actor location is updated.

    Returns:
        List[dict]: One entry per actor that was successfully moved (and possibly rotated):

            {
              "actor_path": str,
              "actor_label": str,
              "transform": {
                  "location": [x,y,z],
                  "quat": [qx,qy,qz,qw]
              }
            }
    """
    debug_lifetime = 2.0
    draw_debug = True
    trace_complex = True

    world = unreal.EditorLevelLibrary.get_editor_world()
    tquery = _resolve_trace_type_query("Visibility")
    if world is None or tquery is None:
        return []

    # Resolve ignore actors
    actors = _resolve_actor_paths_or_selected(actor_paths, selected_only)

    ignore_actors = _sanitize_ignore_actors(actors)

    draw_type = unreal.DrawDebugTrace.FOR_DURATION if draw_debug else unreal.DrawDebugTrace.NONE

    dir_vec = _as_vec3(direction, unreal.Vector(0, 0, -1))
    dirn = _safe_normal(dir_vec, unreal.Vector(0, 0, -1))

    moved = []

    with unreal.ScopedEditorTransaction("Drop Actors to Surface"): 

        for actor in actors: 
            if not actor:
                continue
                

            start = actor.get_actor_location() - (dirn*buffer_distance)
            end = start + (dirn * float(distance))

            per_trace_ignore = _sanitize_ignore_actors(ignore_actors + [actor])

            hit = unreal.SystemLibrary.line_trace_single(
                world,
                start,
                end,
                tquery,
                bool(trace_complex),
                per_trace_ignore,
                draw_type,
                False,
                unreal.LinearColor(1.0, 0.0, 0.0, 1.0),
                unreal.LinearColor(0.0, 1.0, 0.0, 1.0),
                float(debug_lifetime)
            )

            if hit is None or not _hitresult_blocking(hit):
                continue

            loc = _hitresult_location(hit)
            nrm = _hitresult_normal(hit)
            if loc is None or nrm is None:
                continue

            up_hint = actor.get_actor_right_vector()
            q = _build_quat_from_normal(nrm, up_hint)

            # Apply transform
            if set_rotation:
                # Set both location + rotation in one call (more consistent)
                rot = q.rotator()  # Quat -> Rotator
                new_xform = unreal.Transform(
                    location=loc,
                    rotation=rot,   # MUST be Rotator in this binding
                    scale=actor.get_actor_scale3d()
                )
                actor.set_actor_transform(new_xform, False, False)
            else:
                actor.set_actor_location(loc, False, False)

            moved.append({
                "actor_path": actor.get_path_name(),
                "actor_label": actor.get_actor_label(),
                "transform": {
                    "location": [float(loc.x), float(loc.y), float(loc.z)],
                    "quat": [float(q.x), float(q.y), float(q.z), float(q.w)]
                }
            })

    return moved

    
def _gather_actor_bounds(
    world,
    actor_paths: List[str],
    include_child_actors: bool
) -> Tuple[List[unreal.Actor], np.ndarray, np.ndarray]:
    """
    Gather one AABB per Actor (via Actor.get_actor_bounds).

    Narrowing behavior:
      - If actor_paths is a non-empty list: resolve those actors by path and only use those.
      - If actor_paths is empty (or None): gather *all* actors in the level.

    Args:
        world: editor world
        actor_paths: list of full actor object paths (actor.get_path_name()).
                     If empty/None => all actors.
        only_colliding: passed to get_actor_bounds
        include_child_actors: passed to get_actor_bounds

    Returns:
        actors: list[unreal.Actor]
        centers: (N,3) float64
        extents: (N,3) float64
    """
    if actor_paths:
        actors: List[unreal.Actor] = []
        for p in actor_paths:
            a = unreal.find_object(None, p)
            if not a:
                unreal.log_warning(f"_gather_actor_bounds: Could not resolve actor: {p}")
                continue
            # ensure it's an Actor
            if not isinstance(a, unreal.Actor):
                unreal.log_warning(f"_gather_actor_bounds: Object is not an Actor: {p} ({a.get_class().get_name()})")
                continue
            actors.append(a)
    else:
        actors = unreal.GameplayStatics.get_all_actors_of_class(world, unreal.Actor)

    centers = np.empty((len(actors), 3), dtype=np.float64)
    extents = np.empty((len(actors), 3), dtype=np.float64)

    for i, a in enumerate(actors):
        c, e = a.get_actor_bounds(False, include_child_actors)
        centers[i] = _v3_to_np(c)
        extents[i] = _v3_to_np(e)

    return actors, centers, extents


def _aabb_corners_from_center_extent(centers, extents):
    signs = np.array([
        [-1,-1,-1], [-1,-1, 1], [-1, 1,-1], [-1, 1, 1],
        [ 1,-1,-1], [ 1,-1, 1], [ 1, 1,-1], [ 1, 1, 1],
    ], dtype=np.float64)
    return centers[:, None, :] + extents[:, None, :] * signs[None, :, :]


def _transform_to_dict(xform: unreal.Transform) -> dict:
    t = xform.translation
    q = xform.rotation
    s = xform.scale3d
    return {
        "location": {"x": float(t.x), "y": float(t.y), "z": float(t.z)},
        "quat": {"x": float(q.x), "y": float(q.y), "z": float(q.z), "w": float(q.w)},
        "scale": {"x": float(s.x), "y": float(s.y), "z": float(s.z)},
    }


################################
# Look at actual scene geomotry
################################

@register_tool
def look_at_scene_depth(
   # forwards to ray_cast_array by temporarily toggling inside it? (see note below)
) -> List[Dict[str, Any]]:
    """
    Shoots a grid of rays from the Editor camera to detect scene geometry.
    
    USE THIS WHEN:
    - User asks "what's in front of me" or "what can I see"
    - Need to understand the 3D layout/depth of a scene
    - Detecting surfaces, floors, walls from camera perspective
    - Finding empty space vs occupied space in view
    
    NOT FOR: Finding specific actors by name/type (use find_actors instead) Actual Gemoetry - Detects Actual Geomtric Shape of Scene*

    Returns per-sample results similar to ray_cast_array, and also includes:
      - ndc_xy : [x,y] in [-1..1]
      - distance_from_camera_cm : float
      - deg_from_view_center    : float (0 at center ray)

    """
    grid_w = 10
    grid_h = 10
    hfov_deg = 90.0
    aspect =1.0
    distance = 1000000.0
    ignore_actor_paths = []
    jitter = False
    draw_debug = True    

    cam_loc, cam_rot = unreal.EditorLevelLibrary.get_level_viewport_camera_info()

    # YOUR existing helpers:
    C = _v3_to_np(cam_loc)                 # (3,)
    F, R, U = _camera_basis_np(cam_rot)    # each (3,)

    F_unit = F / (np.linalg.norm(F) + 1e-12)


    hfov = math.radians(float(hfov_deg))
    vfov = 2.0 * math.atan(math.tan(hfov * 0.5) / float(aspect))
    tan_h = math.tan(hfov * 0.5)
    tan_v = math.tan(vfov * 0.5)

    rng = None
    if jitter:
        import random
        rng = random.Random(1337)

    out: List[Dict[str, Any]] = []

    for j in range(int(grid_h)):
        for i in range(int(grid_w)):
            u = (i + 0.5) / float(grid_w)
            v = (j + 0.5) / float(grid_h)

            if rng is not None:
                du = (rng.random() - 0.5) / float(grid_w)
                dv = (rng.random() - 0.5) / float(grid_h)
                u = min(1.0, max(0.0, u + du))
                v = min(1.0, max(0.0, v + dv))

            ndc_x = -1.0 + 2.0 * u
            ndc_y = -1.0 + 2.0 * v

            # dir = F + (ndc_x*tan_h)*R + (ndc_y*tan_v)*U
            dir_world = F + (ndc_x * tan_h) * R + (ndc_y * tan_v) * U
            dir_unit = dir_world / (np.linalg.norm(dir_world) + 1e-12)  # FIX

            # Call your existing ray_cast_array (it normalizes direction internally)
            hit_list = ray_cast_array(
                rays=[{"start": [float(C[0]), float(C[1]), float(C[2])]}],
                ignore_actor_paths=ignore_actor_paths,
                distance=float(distance),
                direction=[float(dir_world[0]), float(dir_world[1]), float(dir_world[2])],
            )
            # angle from camera center ray (in degrees)
            dot_fd = float(np.clip(np.dot(F_unit, dir_unit), -1.0, 1.0))
            deg_from_center = float(math.degrees(math.acos(dot_fd)))

            hit = hit_list[0] if hit_list else None
            if not (hit and hit.get("hit", False)):
                continue

            hx, hy, hz = hit["hit_location"]
            H = np.array([hx, hy, hz], dtype=np.float64)
            V = H - C
            dist_cm = float(np.linalg.norm(V))
            

            hit["ndc_xy"] = [float(ndc_x), float(ndc_y)]
            #hit["ij"] = [int(i), int(j)]
            #hit["dir_world"] = [float(dir_world[0]), float(dir_world[1]), float(dir_world[2])]
            hit["dist"] = dist_cm
            hit["deg_from_view_center"] = deg_from_center
            out.append(hit)

    return out


##############################
# View Actors Screenspace
############################## 

@register_tool
def view_actors_screenspace(max_results: int = 5000) -> List[Dict[str, Any]]:
    """
    Get visible actors in the Editor viewport with full transform and mesh details.
    
    USE THIS WHEN:
    - User references objects by screen position ("the thing on the left")
    - Need to know which actors are currently visible to the user
    - Understanding spatial relationships from user's viewpoint
    - Need mesh/transform info for visible objects
    
    Args:
        max_results: Max number of visible actors to return (sorted by angle from center).

    Returns:
        List of dicts for each visible actor:
          - actor_path: Full path for unreal.find_object()
          - actor_label: Human-readable name
          - actor_class: e.g. "StaticMeshActor", "PointLight"
          - is_selected: bool
          - ndc_xy: [x,y] in [-1..1], (-1,-1)=bottom-left, (1,1)=top-right
          - depth: Distance along camera forward (cm)
          - angle_from_center_deg: 0 = dead center of screen
          - bounds_extent: [x,y,z] half-size of bounding box
          - actor_world: {location, quat, scale} transform dict
          - static_meshes: List of mesh components with paths and transforms
    """
    hfov_deg = 90.0
    aspect = 1.0
    near = 10.0
    far = 1.0e12
    expand = 0.0
    include_child_actors = False

    world = unreal.EditorLevelLibrary.get_editor_world()
    cam_loc, cam_rot = unreal.EditorLevelLibrary.get_level_viewport_camera_info()

    current_selection = unreal.EditorLevelLibrary.get_selected_level_actors()
    selection_paths = {a.get_path_name() for a in current_selection if a}

    actors, centers, extents = _gather_actor_bounds(world, [], include_child_actors)

    C = _v3_to_np(cam_loc)
    F, R, U = _camera_basis_np(cam_rot)

    # Frustum culling (vectorized)
    corners = _aabb_corners_from_center_extent(centers, extents)
    V = corners - C[None, None, :]
    x = V @ F
    y = V @ R
    z = V @ U

    hfov = np.deg2rad(float(hfov_deg))
    vfov = 2.0 * np.arctan(np.tan(hfov * 0.5) / float(aspect))
    tan_h = np.tan(hfov * 0.5) * (1.0 + float(expand))
    tan_v = np.tan(vfov * 0.5) * (1.0 + float(expand))

    outside_any = (
        np.all((x - near) < 0, axis=1) |
        np.all((far - x) < 0, axis=1) |
        np.all((y + x * tan_h) < 0, axis=1) |
        np.all((-y + x * tan_h) < 0, axis=1) |
        np.all((z + x * tan_v) < 0, axis=1) |
        np.all((-z + x * tan_v) < 0, axis=1)
    )
    visible = ~outside_any

    # NDC + depth (vectorized)
    centers_V = centers - C[None, :]
    depth = centers_V @ F
    right = centers_V @ R
    up = centers_V @ U

    eps = 1e-9
    safe_depth = np.maximum(depth, eps)
    ndc_x = (right / safe_depth) / tan_h
    ndc_y = (up / safe_depth) / tan_v
    angle_from_center_deg = np.degrees(np.arctan2(np.sqrt(right*right + up*up), depth))

    idx = np.nonzero(visible)[0]
    if len(idx) > max_results:
        idx = idx[:max_results]

    results = []
    for i in idx.tolist():
        a = actors[int(i)]
        actor_path = a.get_path_name()

        # Gather static mesh components
        static_meshes = []
        comps = a.get_components_by_class(unreal.StaticMeshComponent)
        for c in comps:
            if not c:
                continue
            sm = _get_static_mesh_from_component(c)
            comp_world = _component_world_transform(c)
            static_meshes.append({
                "mesh_path": str(sm.get_path_name()) if sm else "",
                "component_world": _transform_to_dict(comp_world),
            })

        results.append({
            # Identity
            "actor_path": actor_path,
            "actor_label": a.get_actor_label(),
            "actor_class": a.get_class().get_name(),
            "is_selected": actor_path in selection_paths,
            # Screen-space
            "ndc_xy": [float(ndc_x[int(i)]), float(ndc_y[int(i)])],
            "depth": float(depth[int(i)]),
            "angle_from_center_deg": float(angle_from_center_deg[int(i)]),
            # World-space
            "world_center_xyz": [float(centers[int(i),0]), float(centers[int(i),1]), float(centers[int(i),2])],
            "bounds_extent": [float(extents[int(i),0]), float(extents[int(i),1]), float(extents[int(i),2])],
            "actor_world": _transform_to_dict(a.get_actor_transform()),
            # Meshes
            "static_meshes": static_meshes,
        })

    results.sort(key=lambda r: r["angle_from_center_deg"])
    return results


############################
# Uber viewport view
############################

def analyze_viewport(max_actor_results: int = 5000) -> Dict[str, Any]:
    """
    Complete viewport analysis: what actors are visible AND what geometry is in view.
    
    USE THIS WHEN:
    - User asks "what do you see" or "describe the scene"
    - Need both actor-level info AND geometric depth/surface data
    - Starting point for understanding an unfamiliar scene
    - User wants comprehensive situational awareness
    
    This combines two analyses:
    1. ACTORS: Which actors are in the camera frustum (fast, bounding-box based)
    2. GEOMETRY: Ray-traced depth samples showing actual surfaces (slower, more precise)
    
    Use the individual tools if you only need one:
    - view_actors_screenspace() — just actors, faster
    - look_at_scene_geometry() — just geometry rays, slower
    
    Args:
        max_actor_results: Maximum actors to return (sorted by angle from view center).
    
    Returns:
        dict with two keys:
        
        "actors": List of visible actors, each with:
            - actor_path: Full path for unreal.find_object()
            - is_selected: bool
            - ndc_xy: [x, y] screen position in [-1..1]
            - depth: Distance along camera forward axis (cm)
            - angle_from_center_deg: How far from screen center (0 = dead center)
            
        "geometry": Grid of ray hits from camera, each with:
            - hit_location: [x, y, z] world position
            - hit_normal: [x, y, z] surface normal
            - hit_actor_path: What actor was hit (if any)
            - ndc_xy: [x, y] screen position of this ray
            - ij: [col, row] grid coordinates
            - distance_from_camera_cm: Depth to hit point
            - deg_from_view_center: Angle from screen center
            
    Example response interpretation:
        - actors with small angle_from_center_deg are near crosshair
        - geometry samples with similar distance_from_camera_cm form a surface
        - gaps in geometry grid indicate sky/empty space
    """
    actors_result = view_actors_screenspace(max_results=max_actor_results)
    geometry_result = look_at_scene_geometry()
    
    return {
        "actors": actors_result,
        "geometry": geometry_result,
    }

##############################
# Get Actor Transforms
##############################


@register_tool
def get_actor_transforms(
    selected_only: bool,
    actor_paths: list[str],

) -> dict:
    """
    Get world transforms and mesh info for actors.
    
    USE THIS WHEN:
    - Need precise position/rotation/scale of specific actors
    - User asks "where is X" or "how big is X"
    - Preparing data for move/rotate/scale operations
    - Inspecting what meshes an actor contains
    
    Tip: Use selected_only=True when user says "the selected objects"
    
    Args:
        actor_paths: List of full Actor Path Names (e.g., ["PersistentLevel.Chair_2"]). 
                     Leave empty if you want to search based on selection or the whole level.
        selected_only: Only used if 'actor_paths' is empty. 
                       True = Process only selected actors. 
                       False = Process all actors in the level.

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
                            "mesh_path": "/Game/Assets/ChairMesh.ChairMesh",
                            "component_world": {...}
                        }
                    ]
                }
            }
    """
    results = {}

    # Get camera
    cam_loc, cam_rot = unreal.EditorLevelLibrary.get_level_viewport_camera_info()
    C = _v3_to_np(cam_loc)
    F, R, U = _camera_basis_np(cam_rot)

    actors = _resolve_actor_paths_or_selected(actor_paths, selected_only)
    # Get the current selection set to compare against for 'is_selected'
    current_selection = unreal.EditorLevelLibrary.get_selected_level_actors()
    selection_paths = {a.get_path_name() for a in current_selection if a}

    count = 0
    for actor in actors:
        if not actor:
            continue

        # Camera deg calc
        P = _v3_to_np(actor.get_actor_location())
        V = P - C

        depth = float(V @ F)   # forward axis
        right = float(V @ R)
        up    = float(V @ U)

        deg_from_center = float(np.degrees(np.arctan2(np.sqrt(right*right + up*up), depth)))

        actor_path = str(actor.get_path_name())
        
        # Core Actor Info
        entry = {
            "actor_label": str(actor.get_actor_label()),
            "actor_path": actor_path,
            "actor_class": str(actor.get_class().get_name()),
            "is_selected": actor_path in selection_paths, 
            "deg_from_view_center": deg_from_center,
             "static_meshes": [],
        }

        entry["actor_world"] = _transform_to_dict(actor.get_actor_transform())

        # Component Scouring
        comps = actor.get_components_by_class(unreal.StaticMeshComponent)
        for c in comps:
            if not c: continue
            sm = _get_static_mesh_from_component(c)
            comp_world = _component_world_transform(c)

            entry["static_meshes"].append({
                "mesh_path": str(sm.get_path_name()) if sm else "",
                "component_world": _transform_to_dict(comp_world),
            })

        results[actor_path] = entry

        count += 1

    return results


@register_tool
def randomize_position(actor_paths: List[str],radius: float=200.0):
    """
    Randomize positions of existing actors within a spherical radius.
    
    USE THIS WHEN:
    - User says "scatter", "randomize positions", "spread out"
    - Making arrangements look more natural/organic
    - Breaking up grid-like patterns
    
    Does NOT duplicate — just moves existing actors. See scatter_replicate for duplication.

    Target selection:
      - If `actor_paths` is non-empty: only those actors are affected (paths are resolved
        with `unreal.find_object`).
      - If `actor_paths` is empty: the currently selected level actors are used.

    Args:
        actor_paths (List[str]):
            List of Actor UObject paths (e.g. "/Game/Map.Map:PersistentLevel.MyActor_2").
            If empty, uses selected actors in the editor.

        radius (float):
            Maximum offset distance in centimeters. Each actor will move by <= `radius`
            from its starting position for this operation.

    Returns:
        List[Dict[str, Any]]:
            One entry per moved actor, containing the actor path and new location:

            [
              {
                "actor_path": str,
                "new_location": [x, y, z]
              },
              ...
            ]

    Notes:
        - This tool changes actor locations in the *Editor* world.
        - Distribution detail: direction is randomized; distance is uniform in [0, radius]
          (which is simple and often good enough, but not perfectly uniform-by-volume in
          the sphere. If you need uniform volume density, use r = radius * (u ** (1/3)).)
    """
    rng = random.Random()
    results = []
    with unreal.ScopedEditorTransaction("Randomize Positions"):

        for a in (_resolve_actor_paths_or_selected(actor_paths) or []):

            d=unreal.Vector(rng.uniform(-1,1),rng.uniform(-1,1),rng.uniform(-1,1)).normal()
            new_loc = a.get_actor_location() + d * rng.uniform(0.0, radius)

            a.set_actor_location(new_loc, False, False)

            results.append({
                "actor_path": a.get_path_name(),
                "actor_label": a.get_actor_label(),
                "new_location": [float(new_loc.x), float(new_loc.y), float(new_loc.z)],
            })
        
    return results
    


@register_tool
def randomize_scale_percent(
    actor_paths: List[str],
    percent: float,
) -> List[Dict[str, Any]]:
    """
    Randomly perturb an actor’s **uniform** scale by a percentage.

    * ``percent`` – maximum deviation expressed as a **percentage**.
      The random factor is drawn from **[1‑p , 1 + p]** where
      ``p = percent / 100``.  Example: ``percent=30`` → factor ∈ [0.70, 1.30].

    * ``actor_paths`` – list of full UObject paths; if empty the currently
      selected actors are used.

    Returns
    -------
    list[dict]
        {
            "actor_path":  str,
            "actor_label": str,
            "new_scale":   [x, y, z]   # all three values identical
        }
    """
    if percent < 0.0:
        raise ValueError("percent must be non‑negative")

    # Convert to a fraction (e.g. 30 → 0.30)
    p = percent / 100.0

    rng = random.Random()
    results: List[Dict[str, Any]] = []

    actors = _resolve_actor_paths_or_selected(actor_paths, selected_only=False)
    if not actors:
        return results

    with unreal.ScopedEditorTransaction("Randomize Scales (percent)"):
        for actor in actors:
            # Current (average) uniform scale
            cur_vec: unreal.Vector = actor.get_actor_scale3d()
            base_scale = (cur_vec.x + cur_vec.y + cur_vec.z) / 3.0

            # Random deviation in [-p, +p]
            delta = rng.uniform(-p, p)          # e.g. -0.30 … +0.30
            factor = 1.0 + delta                # e.g. 0.70 … 1.30

            new_val = base_scale * factor
            new_vec = unreal.Vector(new_val, new_val, new_val)

            # Apply the new uniform scale
            actor.set_actor_scale3d(new_vec)

            results.append({
                "actor_path":  actor.get_path_name(),
                "actor_label": actor.get_actor_label(),
                "new_scale": [
                    float(new_vec.x), float(new_vec.y), float(new_vec.z)
                ],
            })

    return results
      
@register_tool
def randomize_rotation(
    actor_paths: List[str],
    roll_mult: float,
    pitch_mult: float,
    yaw_mult: float,
    space: str,
) -> List[Dict[str, Any]]:
    """
    Randomly perturb an actor’s rotation.

    * `roll_mult`, `pitch_mult`, `yaw_mult` – maximum *absolute* offset in **degrees**.
      The offset is drawn uniformly from [-multiplier, +multiplier] (so a value of 360
      can spin the actor a full turn or more).

    * `space` – ``"world"`` (adds the delta to the actor’s world rotation) or
      ``"object"`` (applies the delta in the actor’s local space).

    Returns a list of dicts with the actor path, label and the new rotation expressed
    as **Roll‑Pitch‑Yaw** (degrees).
    """
    space = space.lower()
    if space not in {"world", "object"}:
        raise ValueError("space must be either 'world' or 'object'")

    rng = random.Random()
    results: List[Dict[str, Any]] = []

    actors = _resolve_actor_paths_or_selected(actor_paths, selected_only=False)
    if not actors:
        return results

    with unreal.ScopedEditorTransaction("Randomize Rotations"):
        for actor in actors:
            # ------- random delta -------------------------------------------------
            d_roll  = rng.uniform(-1.0, 1.0) * roll_mult
            d_pitch = rng.uniform(-1.0, 1.0) * pitch_mult
            d_yaw   = rng.uniform(-1.0, 1.0) * yaw_mult
            delta_rot = unreal.Rotator(d_pitch, d_yaw, d_roll)   # (pitch, yaw, roll)

            if space == "world":
                # world‑space: just add the delta to the current world rot.
                cur_rot = actor.get_actor_rotation()
                new_rot = unreal.Rotator(
                    cur_rot.pitch + delta_rot.pitch,
                    cur_rot.yaw   + delta_rot.yaw,
                    cur_rot.roll  + delta_rot.roll,
                )
                # UE 5.6: two‑arg call – the second arg is the *teleport* flag.
                actor.set_actor_rotation(new_rot, True)   # True → teleport, no sweep

            else:   # object‑space
                # Build a transform that contains only the rotation delta.
                cur_xform   = actor.get_actor_transform()
                delta_xform = unreal.Transform(
                    unreal.Vector(0, 0, 0),   # no translation
                    delta_rot,
                    unreal.Vector(1, 1, 1)    # no scale change
                )
                # Compose the current world transform with the delta (local rotation)
                new_xform = unreal.MathLibrary.compose_transforms(cur_xform, delta_xform)
                actor.set_actor_transform(new_xform, False, True)   # keep location/scale, teleport
                new_rot = new_xform.rotation.rotator()

            # ------- record result ------------------------------------------------
            results.append({
                "actor_path": actor.get_path_name(),
                "actor_label": actor.get_actor_label(),
                "new_rotation_rpy": [
                    float(new_rot.roll),
                    float(new_rot.pitch),
                    float(new_rot.yaw),
                ],
            })

    return results


@register_tool
def scatter_replicate(
    actor_paths: List[str],
    duplicates_per_actor: int = 5,
    radius: float = 200.0,
) -> List[Dict[str, Any]]:
    """
    Duplicate actors and scatter all
    
    USE THIS WHEN:
    - User says "scatter copies", "duplicate and spread", "populate area"
    - Creating clusters of similar objects (trees, rocks, debris)
    - Filling an area with variations of existing objects
    
    Selection logic:
      - If actor_paths is non-empty: uses those actors.
      - If empty: uses currently selected actors.

    Returns a list of dicts describing the spawned duplicates.
    """
    rng = random.Random()
    world = unreal.EditorLevelLibrary.get_editor_world()
    targets = _resolve_actor_paths_or_selected(actor_paths) or []
    if not targets or duplicates_per_actor <= 0:
        return []

    results: List[Dict[str, Any]] = []

    with unreal.ScopedEditorTransaction("Duplicate and Randomize Positions"):
        for src in targets:
            if not src:
                continue

            src_loc = src.get_actor_location()

            # Duplicate in editor (keeps same class/components/settings)
            actor_sub = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)

            # Duplicate once
            dup = actor_sub.duplicate_actor(src) 

            # UE's duplicate_actor duplicates once; so repeat
            all_dupes = []
            if dup:
                # depending on binding, duplicate_actor can return Actor or list[Actor]
                if isinstance(dup, list):
                    all_dupes.extend([d for d in dup if d])
                else:
                    all_dupes.append(dup)

            # create remaining duplicates
            while len(all_dupes) < int(duplicates_per_actor):
                d = actor_sub.duplicate_actor(src) 
                if isinstance(d, list):
                    all_dupes.extend([x for x in d if x])
                elif d:
                    all_dupes.append(d)
                else:
                    break
            all_dupes.append(src)

            # Randomize all
            for i, a in enumerate(all_dupes):
                dvec = unreal.Vector(
                    rng.uniform(-1, 1),
                    rng.uniform(-1, 1),
                    rng.uniform(-1, 1),
                ).normal()

                new_loc = src_loc + dvec * rng.uniform(0.0, float(radius))
                a.set_actor_location(new_loc, False, False)

                results.append({
                    "source_actor_path": src.get_path_name(),
                    "duplicate_actor_path": a.get_path_name(),
                    "new_location": [float(new_loc.x), float(new_loc.y), float(new_loc.z)],
                })

    return results

@register_tool
def destroy_actors(
    actor_paths: List[str],
    selected_only: bool = False,  # <-- ADD

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
    actors = _resolve_actor_paths_or_selected(actor_paths, selected_only)
    
    success_count = 0
    failed_paths = []
    with unreal.ScopedEditorTransaction("Destroy Actors"): 
        for actor in actors:
            try:
                success = unreal.EditorLevelLibrary.destroy_actor(actor)
                if success:
                    success_count += 1
                else:
                    failed_paths.append(actor.get_path_name())
            except Exception as e:
                failed_paths.append(actor.get_path_name() if actor else "unknown")
        
    return {"success_count": success_count, "failed_count": len(failed_paths), "failed_paths": failed_paths}


def transform_actors(
    selected_only: bool,
    values: List[float],
    mode: str,
    space: str,
    actor_paths: List[str],
) -> List[Dict[str, Any]]:
    """
    Apply **one** transformation (translate, rotate or scale) to the supplied
    actors.  The function never mixes operations – only the operation indicated
    by ``mode`` is performed.

    Parameters
    ----------
    actor_paths : list[str] | None
        Explicit actor path names to operate on. If ``None`` the function falls
        back to either the selected actors (``selected_only=True``) or *all*
        actors in the level.

    selected_only : bool
        When ``actor_paths`` is ``None`` this flag decides whether to use the
        current selection (``True``) or every actor in the level (``False``).

    values : list[float] (length = 3)
        The three numbers that encode the operation:
        * **translate** – X/Y/Z offset in cm.
        * **rotate**    – Roll/Pitch/Yaw in degrees.
        * **scale**     – X/Y/Z scale multipliers (1 = no change).

    mode : str
        One of ``"translate"``, ``"rotate"``, ``"scale"`` (case‑insensitive).

    space : str
        ``"world"`` or ``"object"``.  Determines whether translation/rotation are
        applied in world space or the actor’s local space.  Ignored for scaling.

    Returns
    -------
    list[dict]
        For each processed actor a dict containing:
        ``actor_path``, ``actor_label``, ``new_location``, ``new_rotation_rpy``,
        ``new_scale``.
    """
    # ------------------------------------------------------------------ #
    # Validate inputs
    # ------------------------------------------------------------------ #
    if values is None or len(values) != 3:
        raise ValueError("`values` must be a list/tuple of three floats.")
    mode = mode.lower()
    if mode not in {"translate", "rotate", "scale"}:
        raise ValueError("`mode` must be one of: 'translate', 'rotate', 'scale'.")
    space = space.lower()
    if space not in {"world", "object"}:
        raise ValueError("`space` must be either 'world' or 'object'.")

    actors = _resolve_actor_paths_or_selected(actor_paths, selected_only)
    if not actors:
        return []  # nothing to do

    use_object_space = space == "object"
    results = []

    # ------------------------------------------------------------------ #
    # Build the delta transform for the chosen mode only
    # ------------------------------------------------------------------ #
    # Default to identity components
    delta_loc   = unreal.Vector(0.0, 0.0, 0.0)
    delta_rot   = unreal.Rotator(0.0, 0.0, 0.0)   # pitch, yaw, roll order in ctor
    delta_scale = unreal.Vector(1.0, 1.0, 1.0)

    if mode == "translate":
        delta_loc = unreal.Vector(float(values[0]), float(values[1]), float(values[2]))
    elif mode == "rotate":
        # Rotator ctor: (pitch, yaw, roll)
        delta_rot = unreal.Rotator(float(values[1]), float(values[2]), float(values[0]))
    elif mode == "scale":
        delta_scale = unreal.Vector(float(values[0]), float(values[1]), float(values[2]))

    delta_transform = unreal.Transform(delta_loc, delta_rot, delta_scale)

    # ------------------------------------------------------------------ #
    # Apply to each actor inside a scoped transaction (undo‑able)
    # ------------------------------------------------------------------ #
    with unreal.ScopedEditorTransaction(f"{mode.title()} Actors"):
        for actor in actors:
            if not actor:
                continue

            current_xform = actor.get_actor_transform()

            if mode in {"translate", "rotate"}:
                # Choose space handling
                if use_object_space:
                    # Object space → compose after current transform
                    new_xform = unreal.MathLibrary.compose_transforms(current_xform, delta_transform)
                else:
                    # World space → compose before current transform
                    new_xform = unreal.MathLibrary.compose_transforms(delta_transform, current_xform)
            else:  # scale – always local (object) space
                new_xform = unreal.MathLibrary.compose_transforms(current_xform, delta_transform)

            # Apply without sweep and with teleport (no physics interpolation)
            actor.set_actor_transform(new_xform, False, True)

            # Gather result info
            loc   = new_xform.translation
            rot   = new_xform.rotation.rotator()   # returns Rotator (pitch,yaw,roll)
            scale = new_xform.scale3d

            results.append({
                "actor_path": actor.get_path_name(),
                "actor_label": actor.get_actor_label(),
                "new_location": [float(loc.x), float(loc.y), float(loc.z)],
                # Convert back to Roll‑Pitch‑Yaw order for consistency with tests
                "new_rotation_rpy": [float(rot.roll), float(rot.pitch), float(rot.yaw)],
                "new_scale": [float(scale.x), float(scale.y), float(scale.z)],
            })

    return results