import os

import bpy
import config
from blender_utils import selectObject
from material_handler import applyMaterialToObject, createPBRMaterial


def createTableSurface(table_texture_path):
    """Create the textured plane the paper rests on, and return it."""
    if config.VERBOSE:
        print("\n" + "=" * 50)
        print("Setting up environment")
        print("=" * 50)

    if not (os.path.exists(table_texture_path) and os.path.isdir(table_texture_path)):
        raise ValueError(f"Invalid texture path for table surface: {table_texture_path}")

    bpy.ops.mesh.primitive_plane_add(size=1, location=config.TABLE_LOCATION)
    table = bpy.context.active_object
    table.name = config.TABLE_OBJECT_NAME

    # A plane is flat, so Z is left alone; the unit size above makes the scale the size in meters.
    table.scale = (config.TABLE_SIZE[0], config.TABLE_SIZE[1], 1.0)

    if config.VERBOSE:
        print(f"Table dimensions: {config.TABLE_SIZE[0]}m x {config.TABLE_SIZE[1]}m")
        print(f"Table location: {config.TABLE_LOCATION}")

    # Bake the scale into the mesh, so the rigid body simulation sees the real dimensions.
    selectObject(table.name)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    mat = createPBRMaterial(f"{table.name}_Material", texture_dir=table_texture_path)
    applyMaterialToObject(table.name, mat)

    if config.VERBOSE:
        print(f"Applied PBR material from directory: {table_texture_path}")
        print("Environment setup complete\n")

    return table
