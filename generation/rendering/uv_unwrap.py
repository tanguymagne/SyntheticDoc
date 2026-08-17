import bpy
import config
import numpy as np
from blender_utils import deselectAll, ensureInObjectMode, selectObject


def unwrapUVMap(objectName):
    """Angle-based unwrap of the object, normalized so the UVs fill the [0, 1] square."""
    if config.VERBOSE:
        print(f"\nUnwrapping UVs for '{objectName}'...")

    ensureInObjectMode()

    selectObject(objectName)
    obj = bpy.data.objects[objectName]

    bpy.ops.object.mode_set(mode="EDIT")

    # Sync mesh and UV selection, so selecting every face also selects every UV.
    bpy.context.scene.tool_settings.use_uv_select_sync = True
    bpy.ops.mesh.select_all(action="SELECT")

    bpy.ops.uv.unwrap(method="ANGLE_BASED", margin=0.001, correct_aspect=True)

    if config.VERBOSE:
        print("Angle-based unwrap complete")

    # The UV data is only readable outside of edit mode.
    bpy.ops.object.mode_set(mode="OBJECT")
    deselectAll()

    if not obj.data.uv_layers.active:
        raise RuntimeError(f"Unwrapping '{objectName}' produced no UV layer")

    # Normalize the UV coordinates to fit within [0, 1] range, with a small margin.
    max_u, min_u = -np.inf, np.inf
    max_v, min_v = -np.inf, np.inf

    for face in obj.data.polygons:
        for loop_idx in face.loop_indices:
            uv_coords = obj.data.uv_layers.active.data[loop_idx].uv
            if uv_coords.x > max_u:
                max_u = uv_coords.x
            if uv_coords.x < min_u:
                min_u = uv_coords.x
            if uv_coords.y > max_v:
                max_v = uv_coords.y
            if uv_coords.y < min_v:
                min_v = uv_coords.y

    if config.VERBOSE:
        print("UV bounds before normalization:")
        print(f"U: [{min_u:.4f}, {max_u:.4f}]")
        print(f"V: [{min_v:.4f}, {max_v:.4f}]")

    def fitLinear(x0, y0, x1, y1):
        a = (y1 - y0) / (x1 - x0)
        b = y0 - a * x0
        return a, b

    # Keeps the extreme UVs just inside the texture, away from the edge pixels.
    margin = 0.0001

    # Mapping min to 0 and max to 1 stretches the document over the whole mesh
    # without mirroring it, since the scale factors stay positive.
    #
    # U and V are stretched independently, so the document fills the UV bounding box
    # whatever its aspect ratio: a page is expected to cover the sheet edge to edge.
    # This deliberately undoes the correct_aspect=True of the unwrap above, which would
    # otherwise leave the document letterboxed on non-A4-shaped meshes.
    a_u, b_u = fitLinear(min_u, 0 + margin, max_u, 1 - margin)
    a_v, b_v = fitLinear(min_v, 0 + margin, max_v, 1 - margin)

    for face in obj.data.polygons:
        for loop_idx in face.loop_indices:
            uv_coords = obj.data.uv_layers.active.data[loop_idx].uv
            uv_coords.x = a_u * uv_coords.x + b_u
            uv_coords.y = a_v * uv_coords.y + b_v

    if config.VERBOSE:
        print(f"UV coordinates normalized to [0, 1] with margin={margin}")
        print(f"UV unwrapping complete for '{objectName}'\n")


def ensureCorrectUVOrientation(objectName):
    """Rotate the UVs by 180° when the unwrap came out upside down."""
    obj = bpy.data.objects[objectName]
    mesh = obj.data

    if not mesh.uv_layers.active:
        raise RuntimeError(f"'{objectName}' has no UV layer; unwrap it before orienting it")

    # The unwrap can land either way up. The mesh lies flat at this point, so the vertex
    # furthest along +Y is the top of the page and should map to the top of the texture.
    vertices_world = [(obj.matrix_world @ v.co, i) for i, v in enumerate(mesh.vertices)]
    _, top_vertex_idx = max(vertices_world, key=lambda x: x[0].y)

    # A vertex is shared by several faces, so average the V coordinate over all its loops.
    uv_coords_for_top = []
    for face in mesh.polygons:
        for loop_idx in face.loop_indices:
            if mesh.loops[loop_idx].vertex_index == top_vertex_idx:
                uv_coords_for_top.append(mesh.uv_layers.active.data[loop_idx].uv.y)

    if not uv_coords_for_top:
        raise RuntimeError(f"Top vertex of '{objectName}' belongs to no face, so it has no UV")

    avg_top_uv_y = sum(uv_coords_for_top) / len(uv_coords_for_top)

    if avg_top_uv_y < 0.5:
        if config.VERBOSE:
            print("UVs detected as upside down - flipping both U and V coordinates")
        for uv in mesh.uv_layers.active.data:
            uv.uv.x = 1.0 - uv.uv.x
            uv.uv.y = 1.0 - uv.uv.y
