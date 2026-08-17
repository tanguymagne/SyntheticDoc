import os

import bpy
import config
from blender_utils import suppressOutput


def renderImage(output_path):
    """Render the scene from its active camera to an image file."""
    if config.VERBOSE:
        print(f"\nRendering to: {output_path}")

    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    scene = bpy.context.scene
    scene.render.filepath = output_path

    with suppressOutput():
        bpy.ops.render.render(write_still=True)

    if config.VERBOSE:
        print(f"Render complete: {output_path}\n")
