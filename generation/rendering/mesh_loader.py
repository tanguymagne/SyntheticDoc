import math
import os

import bpy
import config
from blender_utils import (
    ensureInObjectMode,
    selectObject,
    setObjectTransform,
    setShading,
    suppressOutput,
)


def cleanObjFile(path):
    """Rewrite the OBJ in place without the non-standard lines Blender's importer warns about."""
    with open(path, "r") as f:
        lines = f.readlines()

    # NOTE: these prefixes cover our own meshes; other OBJ files may need a different filter.
    cleaned_lines = [
        line
        for line in lines
        if not line.strip().startswith(("ts ", "td ", "ed", "s", "ea", "e", "ny", "nv"))
    ]
    with open(path, "w") as f:
        f.writelines(cleaned_lines)


def importOBJ(filepath, object_name=None):
    """Import an OBJ file into the scene and return the new object."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"OBJ file not found: {filepath}")

    if config.VERBOSE:
        print(f"Importing OBJ: {filepath}")

    ensureInObjectMode()

    # The import operator returns no handle on what it created, so diff the scene around it.
    objects_before = set(bpy.data.objects)

    with suppressOutput():
        bpy.ops.wm.obj_import(filepath=filepath)

    objects_after = set(bpy.data.objects)
    new_objects = objects_after - objects_before

    if not new_objects:
        raise RuntimeError(f"No objects were imported from {filepath}")

    # An OBJ file may hold several objects; only the first one is kept.
    imported_obj = list(new_objects)[0]

    if config.VERBOSE:
        print(f"  - Imported object: {imported_obj.name}")
        print(f"  - Vertices: {len(imported_obj.data.vertices)}")
        print(f"  - Faces: {len(imported_obj.data.polygons)}")

    if object_name:
        imported_obj.name = object_name
        if config.VERBOSE:
            print(f"  - Renamed to: {object_name}")

    return imported_obj


def loadPaperMesh(obj_filepath, name=config.PAPER_OBJECT_NAME):
    """Import a paper mesh, then center, position and shade it, and return the object."""
    if config.VERBOSE:
        print("\n" + "=" * 50)
        print(f"Loading paper mesh: {os.path.basename(obj_filepath)}")
        print("=" * 50)

    cleanObjFile(obj_filepath)

    paper_obj = importOBJ(obj_filepath, object_name=name)

    selectObject(paper_obj.name)
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")

    setObjectTransform(
        paper_obj.name, config.PAPER_LOCATION, config.PAPER_ROTATION, config.PAPER_SCALE
    )

    setShading(
        paper_obj.name,
        smooth=config.SMOOTH_SHADING,
        auto_smooth=config.AUTO_SMOOTH,
        angle=config.SHADING_ANGLE,
    )

    # Left selected and active: the UV unwrapping steps that follow operate on the active object.
    selectObject(paper_obj.name)

    if config.VERBOSE:
        print(f"Centered origin and positioned '{paper_obj.name}':")
        print(f"  - Location: {config.PAPER_LOCATION}")
        print(f"  - Rotation: {config.PAPER_ROTATION}")
        print(f"  - Scale: {config.PAPER_SCALE}")
        print(f"Paper mesh '{name}' loaded successfully\n")

    return paper_obj


def flipPaperToBackSide(obj):
    """Turn the paper over, so the document is rendered on its back side."""
    current_rotation = list(obj.rotation_euler)
    current_rotation[1] += math.pi
    obj.rotation_euler = current_rotation

    # The rotation alone would mirror the document, so mirror the UVs back.
    if obj.data.uv_layers:
        uv_layer = obj.data.uv_layers.active.data
        for uv in uv_layer:
            uv.uv[0] = 1.0 - uv.uv[0]

    if config.VERBOSE:
        print(f"Flipped paper '{obj.name}' to back side")
        print("  - Rotated 180° around Y axis")
        print("  - Flipped UVs horizontally to preserve texture orientation")
