import unreal

# ----------------------------
# Small helpers (simple + UE-friendly)
# ----------------------------
def _normalize_mesh_object_path(asset_object_path: str) -> str:
    """
    Accepts:
      "/Game/Path/SM_Thing.SM_Thing"
      "/Game/Path/SM_Thing"  -> normalized to "/Game/Path/SM_Thing.SM_Thing"
    """
    if not isinstance(asset_object_path, str):
        raise TypeError("asset_object_path must be a string")

    s = asset_object_path.strip()
    if not s:
        raise ValueError("asset_object_path is empty")

    if "." not in s:
        name = s.rsplit("/", 1)[-1]
        s = f"{s}.{name}"
    return s


def _num(v, name: str) -> float:
    """Accept int/float or numeric strings; return float."""
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        vs = v.strip()
        if vs == "":
            raise ValueError(f"{name} is empty string")
        return float(vs)
    raise TypeError(f"{name} must be int/float or numeric string, got {type(v)}")


def _get_static_mesh_from_component(static_mesh_component):
    """Same pattern as your other cell: ask the component for its editor property."""
    if not static_mesh_component:
        return None
    return static_mesh_component.get_editor_property("static_mesh")

def _get_static_mesh_component(actor) -> unreal.StaticMeshComponent:
    """
    UE Python can be inconsistent about exposing `is_a()` on wrapped actor instances.
    So we avoid it and use a property-first approach (like your unit test code).
    """
    if not actor:
        return None

    # For StaticMeshActor this property exists and is the most reliable
    try:
        smc = actor.get_editor_property("static_mesh_component")
        if smc:
            return smc
    except Exception:
        pass

    # Fallback: find any StaticMeshComponent on the actor
    try:
        comps = actor.get_components_by_class(unreal.StaticMeshComponent)
        if comps:
            return comps[0]
    except Exception:
        pass

    # Fallback: root component if it happens to be a StaticMeshComponent
    try:
        root = actor.get_editor_property("root_component")
        if isinstance(root, unreal.StaticMeshComponent):
            return root
    except Exception:
        pass

    return None

# ----------------------------
# Tool
# ----------------------------


@register_tool
def spawn_static_mesh_from_library(
    asset_object_path: str,
    loc_x=0.0, loc_y=0.0, loc_z=0.0,
    quat_x=0.0, quat_y=0.0, quat_z=0.0, quat_w=1.0,
    scl_x=1.0, scl_y=1.0, scl_z=1.0,
    actor_label: str = "SpawnedStaticMesh",
):
    """
    Spawn a StaticMeshActor in the current editor level from a StaticMesh asset path,
    using world-space Location + Quaternion + Scale.

    Accepts numeric (int/float) or numeric strings for transform fields.

    Rotation is provided as a quaternion (x,y,z,w) to avoid Euler/Rotator ambiguity.
    """
    obj_path = _normalize_mesh_object_path(asset_object_path)
    mesh = unreal.EditorAssetLibrary.load_asset(obj_path)
    if not mesh or not isinstance(mesh, unreal.StaticMesh):
        return None

    loc = unreal.Vector(_num(loc_x, "loc_x"), _num(loc_y, "loc_y"), _num(loc_z, "loc_z"))
    scl = unreal.Vector(_num(scl_x, "scl_x"), _num(scl_y, "scl_y"), _num(scl_z, "scl_z"))

    quat = unreal.Quat(
        _num(quat_x, "quat_x"),
        _num(quat_y, "quat_y"),
        _num(quat_z, "quat_z"),
        _num(quat_w, "quat_w"),
    )

    actor = unreal.EditorLevelLibrary.spawn_actor_from_class(
        unreal.StaticMeshActor,
        loc,
        unreal.Rotator(0.0, 0.0, 0.0),
    )
    if not actor:
        return None

    if actor_label is not None and str(actor_label).strip():
        actor.set_actor_label(str(actor_label))

    smc = _get_static_mesh_component(actor)
    if not smc:
        unreal.EditorLevelLibrary.destroy_actor(actor)
        return None
    smc.set_editor_property("static_mesh", mesh)

    # IMPORTANT FIX: Transform(Location, Rotation, Scale)
    xf = unreal.Transform(loc, quat.rotator(), scl)
    actor.set_actor_transform(xf, False, True)

    return actor