"""Ground truth passes, rendered from the scene left behind by a finished sample render.

Each pass swaps the paper's material for one that paints a quantity instead of a look, blacks
out the table, renders once, and puts the scene's render settings back as it found them.
"""

import os
from contextlib import contextmanager

import bpy
import config
from blender_utils import suppressOutput


@contextmanager
def groundTruthRenderSettings(output_path, file_format, color_depth, samples, use_denoising):
    """Switch the scene over to one ground truth pass, and put every setting back afterwards.

    The passes run one after another on the same scene, so anything left changed would silently
    apply to the passes that follow. Restoring happens even when the render raises.
    """
    scene = bpy.context.scene
    image_settings = scene.render.image_settings

    saved = {
        "engine": scene.render.engine,
        "samples": scene.cycles.samples,
        "use_denoising": scene.cycles.use_denoising,
        "filepath": scene.render.filepath,
        "film_transparent": scene.render.film_transparent,
        "file_format": image_settings.file_format,
        "color_mode": image_settings.color_mode,
        "color_depth": image_settings.color_depth,
    }

    # The world lights the background, which has to come out pure black for the paper to be
    # separable from the table, so the background nodes are blacked out and saved along with it.
    background_nodes = []
    if scene.world and scene.world.use_nodes:
        background_nodes = [n for n in scene.world.node_tree.nodes if n.type == "BACKGROUND"]
    saved_world = [
        (n, tuple(n.inputs["Color"].default_value), n.inputs["Strength"].default_value)
        for n in background_nodes
    ]

    try:
        scene.render.engine = "CYCLES"
        scene.cycles.samples = samples
        scene.cycles.use_denoising = use_denoising
        scene.render.film_transparent = False
        image_settings.color_mode = "RGB"
        image_settings.file_format = file_format
        image_settings.color_depth = color_depth
        if file_format == "OPEN_EXR":
            image_settings.exr_codec = "ZIP"

        for node in background_nodes:
            node.inputs["Color"].default_value = (0, 0, 0, 1)
            node.inputs["Strength"].default_value = 0.0

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        scene.render.filepath = output_path

        if config.VERBOSE:
            print(f"Format: {file_format}, samples: {samples}, denoising: {use_denoising}")
            print(f"Output: {output_path}")
            print(f"Resolution: {scene.render.resolution_x}x{scene.render.resolution_y}")
            print("Rendering...")

        yield scene

    finally:
        scene.render.engine = saved["engine"]
        scene.cycles.samples = saved["samples"]
        scene.cycles.use_denoising = saved["use_denoising"]
        scene.render.filepath = saved["filepath"]
        scene.render.film_transparent = saved["film_transparent"]
        image_settings.file_format = saved["file_format"]
        image_settings.color_mode = saved["color_mode"]
        image_settings.color_depth = saved["color_depth"]

        for node, color, strength in saved_world:
            node.inputs["Color"].default_value = color
            node.inputs["Strength"].default_value = strength


def createUVGradientMaterial(objectName):
    """Paint a mesh with its own UV coordinates: U in red, V in green, a constant 1.0 in blue."""
    if config.VERBOSE:
        print(f"Creating UV gradient material for '{objectName}'...")

    mat_name = "UV_Gradient_Material"

    # Rebuilt from scratch on every call, so a stale version from an earlier sample cannot
    # survive into this one.
    if mat_name in bpy.data.materials:
        bpy.data.materials.remove(bpy.data.materials[mat_name])

    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    nodes.clear()

    node_uv = nodes.new(type="ShaderNodeUVMap")
    node_uv.location = (-400, 0)

    node_separate = nodes.new(type="ShaderNodeSeparateXYZ")
    node_separate.location = (-200, 0)

    node_combine = nodes.new(type="ShaderNodeCombineRGB")
    node_combine.location = (0, 0)
    node_combine.inputs["B"].default_value = 1.0

    # Emission rather than a shaded surface: the pixel value has to be the coordinate itself,
    # unaffected by the lights that are still in the scene.
    node_emission = nodes.new(type="ShaderNodeEmission")
    node_emission.location = (200, 0)
    node_emission.inputs["Strength"].default_value = 1.0

    node_output = nodes.new(type="ShaderNodeOutputMaterial")
    node_output.location = (400, 0)

    links.new(node_uv.outputs["UV"], node_separate.inputs["Vector"])
    links.new(node_separate.outputs["X"], node_combine.inputs["R"])
    links.new(node_separate.outputs["Y"], node_combine.inputs["G"])
    links.new(node_combine.outputs["Image"], node_emission.inputs["Color"])
    links.new(node_emission.outputs["Emission"], node_output.inputs["Surface"])

    obj = bpy.data.objects.get(objectName)
    if obj:
        obj.data.materials.clear()
        obj.data.materials.append(mat)
        if config.VERBOSE:
            print(f"UV gradient material applied to '{objectName}'")

    return mat


def setupBlackMaterial(objectName):
    """Paint an object pure black, so it reads as background and can be masked out."""
    mat_name = "Black_Background_Material"

    if mat_name in bpy.data.materials:
        bpy.data.materials.remove(bpy.data.materials[mat_name])

    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    nodes.clear()

    # Black emission, not a black surface: a surface would still catch a highlight from the
    # lights and leave the background short of a clean zero.
    node_emission = nodes.new(type="ShaderNodeEmission")
    node_emission.inputs["Color"].default_value = (0, 0, 0, 1)
    node_emission.inputs["Strength"].default_value = 1.0

    node_output = nodes.new(type="ShaderNodeOutputMaterial")
    node_output.location = (200, 0)

    links.new(node_emission.outputs["Emission"], node_output.inputs["Surface"])

    obj = bpy.data.objects.get(objectName)
    if obj:
        obj.data.materials.clear()
        obj.data.materials.append(mat)
        if config.VERBOSE:
            print(f"Black material applied to '{objectName}'")

    return mat


def renderUVInverseMap(output_path):
    """Render each visible point of the paper as its UV coordinate, black elsewhere.

    This is the map that says where every pixel of the photograph came from on the flat page,
    which is what makes unwarping possible. EXR keeps the coordinates at full float precision.
    """
    if config.VERBOSE:
        print("\n" + "=" * 50)
        print("Rendering Backward UV Map")
        print("=" * 50)

    createUVGradientMaterial(config.PAPER_OBJECT_NAME)
    setupBlackMaterial(config.TABLE_OBJECT_NAME)

    # A single sample, because there is nothing to converge: the emission shader already outputs
    # the exact value, and more samples would only average neighbours in and blur the edges.
    # Denoising is off for the same reason: it would invent coordinates that lie off the surface.
    with (
        groundTruthRenderSettings(
            output_path, file_format="OPEN_EXR", color_depth="32", samples=1, use_denoising=False
        ),
        suppressOutput(),
    ):
        bpy.ops.render.render(write_still=True)

    if config.VERBOSE:
        print(f"Backward map rendered to: {output_path}\n")

    return output_path


def create3DCoordinateMaterial(objectName):
    """Paint a mesh with its world position: X in red, Y in green, Z in blue."""
    if config.VERBOSE:
        print(f"Creating 3D coordinate material for '{objectName}'...")

    mat_name = "3D_Coordinate_Material"

    if mat_name in bpy.data.materials:
        bpy.data.materials.remove(bpy.data.materials[mat_name])

    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    nodes.clear()

    # The Geometry node's Position output is already in world space, so the shader needs no
    # transform of its own.
    node_geometry = nodes.new(type="ShaderNodeNewGeometry")
    node_geometry.location = (-400, 0)

    node_separate = nodes.new(type="ShaderNodeSeparateXYZ")
    node_separate.location = (-200, 0)

    node_combine = nodes.new(type="ShaderNodeCombineRGB")
    node_combine.location = (0, 0)

    node_emission = nodes.new(type="ShaderNodeEmission")
    node_emission.location = (200, 0)
    node_emission.inputs["Strength"].default_value = 1.0

    node_output = nodes.new(type="ShaderNodeOutputMaterial")
    node_output.location = (400, 0)

    links.new(node_geometry.outputs["Position"], node_separate.inputs["Vector"])
    links.new(node_separate.outputs["X"], node_combine.inputs["R"])
    links.new(node_separate.outputs["Y"], node_combine.inputs["G"])
    links.new(node_separate.outputs["Z"], node_combine.inputs["B"])
    links.new(node_combine.outputs["Image"], node_emission.inputs["Color"])
    links.new(node_emission.outputs["Emission"], node_output.inputs["Surface"])

    obj = bpy.data.objects.get(objectName)
    if obj:
        obj.data.materials.clear()
        obj.data.materials.append(mat)
        if config.VERBOSE:
            print(f"3D coordinate material applied to '{objectName}'")

    return mat


def render3DMap(output_path):
    """Render each visible point of the paper as its world XYZ position, black elsewhere.

    Always EXR: world coordinates are metres in the scene's own range, which 8-bit PNG would
    clip and quantize into uselessness.
    """
    if config.VERBOSE:
        print("\n" + "=" * 50)
        print("Rendering 3D Coordinate Map")
        print("=" * 50)

    create3DCoordinateMaterial(config.PAPER_OBJECT_NAME)
    setupBlackMaterial(config.TABLE_OBJECT_NAME)

    if not output_path.endswith(".exr"):
        output_path = os.path.splitext(output_path)[0] + ".exr"

    # One sample and no denoising, so the coordinates come out exactly as the shader wrote them.
    with (
        groundTruthRenderSettings(
            output_path, file_format="OPEN_EXR", color_depth="32", samples=1, use_denoising=False
        ),
        suppressOutput(),
    ):
        bpy.ops.render.render(write_still=True)

    if config.VERBOSE:
        print(f"3D coordinate map rendered to: {output_path}\n")

    return output_path


def createAlbedoMaterial(objectName, texturePath):
    """Show a texture on a mesh, with no lighting, shadows or shading of any kind."""
    if config.VERBOSE:
        print(f"Creating albedo material for '{objectName}' with image '{texturePath}'...")

    image = bpy.data.images.load(texturePath)
    if not image:
        print(f"ERROR image '{texturePath}' not found in bpy.data.images!")
        return None

    mat_name = "Albedo_Material"

    if mat_name in bpy.data.materials:
        bpy.data.materials.remove(bpy.data.materials[mat_name])

    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    nodes.clear()

    node_texture = nodes.new(type="ShaderNodeTexImage")
    node_texture.image = image
    node_texture.location = (-200, 0)

    node_emission = nodes.new(type="ShaderNodeEmission")
    node_emission.location = (0, 0)
    node_emission.inputs["Strength"].default_value = 1.0

    node_output = nodes.new(type="ShaderNodeOutputMaterial")
    node_output.location = (200, 0)

    links.new(node_texture.outputs["Color"], node_emission.inputs["Color"])
    links.new(node_emission.outputs["Emission"], node_output.inputs["Surface"])

    obj = bpy.data.objects.get(objectName)
    if obj:
        obj.data.materials.clear()
        obj.data.materials.append(mat)
        if config.VERBOSE:
            print(f"Albedo material applied to '{objectName}'")
    else:
        print(f"ERROR object '{objectName}' not found!")

    return mat


def renderAlbedoMap(output_path, texture_path):
    """Render the document texture warped onto the paper, unlit, black elsewhere.

    This is the target a deshadowing model is trained against: the same page geometry as the
    main render, but with every trace of the scene's lighting removed.
    """
    if config.VERBOSE:
        print("\n" + "=" * 50)
        print("Rendering Albedo Map")
        print("=" * 50)

    createAlbedoMaterial(config.PAPER_OBJECT_NAME, texture_path)
    setupBlackMaterial(config.TABLE_OBJECT_NAME)

    # One sample and no denoising, so the texture comes out exactly as it went in.
    with (
        groundTruthRenderSettings(
            output_path, file_format="PNG", color_depth="8", samples=1, use_denoising=False
        ),
        suppressOutput(),
    ):
        bpy.ops.render.render(write_still=True)

    if config.VERBOSE:
        print(f"Albedo map rendered to: {output_path}\n")

    return output_path


def createNormalMaterial(objectName):
    """Paint a mesh with its surface normal: X in red, Y in green, Z in blue."""
    if config.VERBOSE:
        print(f"Creating normals material for '{objectName}'...")

    mat_name = "Normals_Material"

    if mat_name in bpy.data.materials:
        bpy.data.materials.remove(bpy.data.materials[mat_name])

    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    nodes.clear()

    node_geometry = nodes.new(type="ShaderNodeNewGeometry")
    node_geometry.location = (-400, 0)

    node_separate = nodes.new(type="ShaderNodeSeparateXYZ")
    node_separate.location = (-200, 0)

    node_combine = nodes.new(type="ShaderNodeCombineRGB")
    node_combine.location = (0, 0)

    node_emission = nodes.new(type="ShaderNodeEmission")
    node_emission.location = (200, 0)
    node_emission.inputs["Strength"].default_value = 1.0

    node_output = nodes.new(type="ShaderNodeOutputMaterial")
    node_output.location = (400, 0)

    links.new(node_geometry.outputs["Normal"], node_separate.inputs["Vector"])
    links.new(node_separate.outputs["X"], node_combine.inputs["R"])
    links.new(node_separate.outputs["Y"], node_combine.inputs["G"])
    links.new(node_separate.outputs["Z"], node_combine.inputs["B"])
    links.new(node_combine.outputs["Image"], node_emission.inputs["Color"])
    links.new(node_emission.outputs["Emission"], node_output.inputs["Surface"])

    obj = bpy.data.objects.get(objectName)
    if obj:
        obj.data.materials.clear()
        obj.data.materials.append(mat)
        if config.VERBOSE:
            print(f"  ✓ Normals material applied to '{objectName}'")

    return mat


def renderNormalMap(output_path):
    """Render each visible point of the paper as its surface normal, black elsewhere.

    Always EXR: normals run from -1 to 1, and an 8-bit PNG would clip everything negative away
    rather than encode it.
    """
    if config.VERBOSE:
        print("\n" + "=" * 50)
        print("Rendering Normals Map")
        print("=" * 50)

    createNormalMaterial(config.PAPER_OBJECT_NAME)
    setupBlackMaterial(config.TABLE_OBJECT_NAME)

    if not output_path.endswith(".exr"):
        output_path = os.path.splitext(output_path)[0] + ".exr"

    # One sample and no denoising, so the normals come out exactly as the shader wrote them.
    with (
        groundTruthRenderSettings(
            output_path, file_format="OPEN_EXR", color_depth="32", samples=1, use_denoising=False
        ),
        suppressOutput(),
    ):
        bpy.ops.render.render(write_still=True)

    if config.VERBOSE:
        print(f"Normal map rendered to: {output_path}\n")

    return output_path


def makeWhiteDocument():
    """Blank the document texture of the paper already in the scene, keeping its paper shading.

    The material itself is left alone, so the render still carries the paper's real roughness
    and normal detail; only the printed content goes away.
    """
    mat = bpy.data.materials.get(config.BASE_PAPER_MATERIAL_NAME)
    nodes = mat.node_tree.nodes

    group_tree = nodes[2].node_tree
    tex_node = next((n for n in group_tree.nodes if n.type == "TEX_IMAGE"), None)

    if not tex_node:
        raise ValueError("Texture node not found")

    # A generated image rather than an RGB node, so the texture node keeps its place in the
    # group and none of its outgoing links have to be rebuilt.
    width, height = 1024, 1024
    white_image = bpy.data.images.new(
        "White_Document_Texture", width=width, height=height, alpha=True
    )
    white_image.generated_color = (1.0, 1.0, 1.0, 1.0)
    tex_node.image = white_image


def renderShadowMap(output_path):
    """Render the paper blank and lit, black elsewhere, leaving only the scene's shading.

    Dividing the main render by this map recovers the document free of shadows, which is what
    makes the pair usable as deshadowing supervision.
    """
    if config.VERBOSE:
        print("\n" + "=" * 50)
        print("Rendering Shadow Map")
        print("=" * 50)

    makeWhiteDocument()
    setupBlackMaterial(config.TABLE_OBJECT_NAME)

    if not output_path.endswith(".png"):
        output_path = os.path.splitext(output_path)[0] + ".png"

    # The odd one out: this pass actually traces light, so it needs the full sample count to
    # converge, and denoising on to match the main render it will be divided into.
    with (
        groundTruthRenderSettings(
            output_path,
            file_format="PNG",
            color_depth="8",
            samples=config.RENDER_SAMPLES,
            use_denoising=True,
        ),
        suppressOutput(),
    ):
        bpy.ops.render.render(write_still=True)

    if config.VERBOSE:
        print(f"Shadow map rendered to: {output_path}\n")

    return output_path
