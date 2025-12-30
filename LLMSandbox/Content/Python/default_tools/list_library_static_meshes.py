import unreal

def _asset_data_to_object_path(ad):
    package_name = str(ad.package_name)  # "/Game/Path/Asset"
    asset_name = str(ad.asset_name)      # "Asset"
    return f"{package_name}.{asset_name}"

def _get_static_mesh_from_component(smc):
    # UE 5.6 Python: prefer editor property access
    mesh = smc.get_editor_property("static_mesh")
    return mesh

def _set_static_mesh_on_component(smc, mesh):
    smc.set_editor_property("static_mesh", mesh)

@register_tool
def find_all_static_meshes_in_library(include_engine_content:bool=False, include_plugin_content:bool=False):
    """
    Find all StaticMesh assets available in the Unreal Asset Registry and return
    a structured dictionary describing them.

    This tool scans the project Content directory (and optionally Engine/Plugins)
    and queries the Asset Registry for assets of class `StaticMesh`.

    Args:
        include_engine_content (bool):
            If True, also scan `/Engine` and include Engine static meshes.
            Defaults to False.

        include_plugin_content (bool):
            If True, also scan `/Plugins` and include plugin static meshes.
            Defaults to False.

    Returns:
        dict[str, dict]:
            A dictionary keyed by StaticMesh object path (for example:
            `"/Game/Props/SM_Chair.SM_Chair"`). Each value is a dictionary:

            {
              "name": <asset name>,
              "object_path": <package.asset>,
              "package_name": <"/Game/Props/SM_Chair">,
              "package_path": <"/Game/Props">,
              "asset_class": <"StaticMesh">,
              "tags": { <tag_key>: <tag_value>, ... }
            }

            Notes:
            - `tags` comes from AssetRegistry metadata (if present).
            - `asset_class` is derived from `asset_class_path` when available
              (UE5+), otherwise it falls back to `asset_class`.

    Example:
        meshes = find_all_static_meshes_in_library()
        first_key = next(iter(meshes))
        print(first_key, meshes[first_key])

    """
    ar = unreal.AssetRegistryHelpers.get_asset_registry()

    paths = ["/Game"]
    if include_engine_content:
        paths.append("/Engine")
    if include_plugin_content:
        paths.append("/Plugins")

    ar.scan_paths_synchronous(paths, force_rescan=False)

    flt = unreal.ARFilter(
        class_names=["StaticMesh"],
        package_paths=paths,
        recursive_paths=True,
        recursive_classes=True
    )

    assets = ar.get_assets(flt)

    result = {}
    for ad in assets:
        name = str(ad.asset_name)
        package_name = str(ad.package_name)
        package_path = str(ad.package_path)

        if hasattr(ad, "asset_class_path"):
            asset_class = str(ad.asset_class_path.asset_name)
        else:
            asset_class = str(getattr(ad, "asset_class", "StaticMesh"))

        object_path = _asset_data_to_object_path(ad)

        tags = {}
        tag_map = getattr(ad, "tags_and_values", None)
        if tag_map:
            for k in tag_map.keys():
                tags[str(k)] = str(tag_map.get(k))

        result[object_path] = {
            "name": name,
            "object_path": object_path,
            "package_name": package_name,
            "package_path": package_path,
            "asset_class": asset_class,
            "tags": tags
        }

    return result