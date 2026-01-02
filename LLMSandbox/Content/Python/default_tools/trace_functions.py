import unreal
import math
from typing import List, Optional

# ----------------------------
# Vector / Math helpers
# ----------------------------

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

@register_tool
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
            "hit_actor_path": None,
            "transform": None
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
        actor_obj = _hitresult_actor(hit)

        if loc is None or nrm is None:
            results.append(out)
            continue

        q = _build_quat_from_normal(nrm, up_hint)
        actor_path = actor_obj.get_path_name() if actor_obj else None

        out["hit"] = True
        out["hit_location"] = [float(loc.x), float(loc.y), float(loc.z)]
        out["hit_normal"] = [float(nrm.x), float(nrm.y), float(nrm.z)]
        out["hit_actor_path"] = actor_path
        out["transform"] = {
            "location": [float(loc.x), float(loc.y), float(loc.z)],
            "quat": [float(q.x), float(q.y), float(q.z), float(q.w)]
        }

        results.append(out)

    return results


@register_tool
def move_actor_until_hit(
    actor_paths: List[str] = [],
    distance: float = 10000,
    direction: List[float] = [0, 0, -1],
    set_rotation: bool = True,  # NEW: optionally apply the computed quaternion
) -> List[dict]:
    """
    Cast a ray for each input actor (from the actor's current location) and, on hit,
    move that actor so its location becomes the hit point. Optionally rotate the actor
    so its local +Z aligns to the hit normal. ( Great for placing object on uneven terrain )

    Args:
        actor_paths:
            List of Actor UObject paths to raycast from and potentially move.

        ignore_actor_paths:
            List of Actor UObject paths to ignore during the trace (in addition to the
            actor being moved; the actor itself is always ignored for its own trace).

        distance:
            Trace length in centimeters.

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
    ignore_actors = []
    for p in (actor_paths or []):
        a = unreal.find_object(None, p)
        if a and isinstance(a, unreal.Actor):
            ignore_actors.append(a)

    ignore_actors = _sanitize_ignore_actors(ignore_actors)

    draw_type = unreal.DrawDebugTrace.FOR_DURATION if draw_debug else unreal.DrawDebugTrace.NONE

    dir_vec = _as_vec3(direction, unreal.Vector(0, 0, -1))
    dirn = _safe_normal(dir_vec, unreal.Vector(0, 0, -1))

    moved = []

    for path in (actor_paths or []):
        actor = unreal.find_object(None, path)
        if not actor or not isinstance(actor, unreal.Actor):
            continue

        start = actor.get_actor_location()
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