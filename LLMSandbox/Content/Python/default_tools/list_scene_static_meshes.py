import unreal
import math
import re
import unreal
from typing import Literal, List


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


