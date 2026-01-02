import unreal
import copy
from typing import List, Dict, Any, Optional


@register_tool
def get_actor_count():
    """Get the total number of actors in the current Unreal level."""
    world = unreal.EditorLevelLibrary.get_editor_world()
    actors = unreal.GameplayStatics.get_all_actors_of_class(world, unreal.Actor)
    return f"Total actors: {len(actors)}"


@register_tool
def list_actor_names(limit: int = 5):
    """List actor names in the current Unreal level."""
    world = unreal.EditorLevelLibrary.get_editor_world()
    actors = unreal.GameplayStatics.get_all_actors_of_class(world, unreal.Actor)
    names = [actor.get_name() for actor in actors[:limit]]
    return "\n".join(f"- {n}" for n in names)

