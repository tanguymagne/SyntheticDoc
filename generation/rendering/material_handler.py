import os
import random

import bpy
import config
from blender_utils import suppressOutput


def importMaterialFromBlend(blend_file_path, material_name):
    """Append a material from another .blend file and return it."""
    if not os.path.exists(blend_file_path):
        raise FileNotFoundError(f"Blend file not found: {blend_file_path}")

    if config.VERBOSE:
        print(f"Importing material '{material_name}' from {os.path.basename(blend_file_path)}...")

    # Appending twice would give the second copy a ".001" suffix, so drop any earlier import.
    existing_mat = bpy.data.materials.get(material_name)
    if existing_mat:
        if config.VERBOSE:
            print(f"Material '{material_name}' already exists, deleting and reimporting...")
        bpy.data.materials.remove(existing_mat)

    # link=False appends a full copy, so the render does not depend on the source file.
    with bpy.data.libraries.load(blend_file_path, link=False) as (data_from, data_to):
        if material_name not in data_from.materials:
            raise ValueError(f"Material '{material_name}' not found in {blend_file_path}")
        data_to.materials = [material_name]

    imported_mat = bpy.data.materials.get(material_name)

    if not imported_mat:
        raise RuntimeError(f"Material '{material_name}' was appended but is not in bpy.data")

    if config.VERBOSE:
        print(f"Material '{material_name}' imported successfully")

    return imported_mat


def createPaperMaterial(texture_path):
    """Import the base paper material and plug the document image into its texture node."""
    mat = importMaterialFromBlend(config.BASE_PAPER_MATERIAL_BLEND, config.BASE_PAPER_MATERIAL_NAME)

    mat.use_nodes = True

    nodes = mat.node_tree.nodes

    if texture_path and os.path.exists(texture_path):
        if config.VERBOSE:
            print(f"Adding document texture: {os.path.basename(texture_path)}")

        image = bpy.data.images.load(texture_path)

        # The document image lives inside the paper node group, under a frame labelled
        # "Document Texture"; the frame is what identifies it, since the node itself is unnamed.
        group_tree = nodes[2].node_tree
        frame = next(
            (n for n in group_tree.nodes if n.type == "FRAME" and n.label == "Document Texture"),
            None,
        )

        if not frame:
            raise ValueError("Frame not found")

        tex_node = next(
            (n for n in group_tree.nodes if n.parent == frame and n.type == "TEX_IMAGE"), None
        )

        if not tex_node:
            raise ValueError("Texture node not found")

        tex_node.image = image

    else:
        print("Failed to add document texture into material")

    if config.VERBOSE:
        print("Document material created\n")

    return mat


def randomizePaperMaterial(material):
    """Randomize the paper node group inputs, and return the values applied."""

    if not material or not material.use_nodes:
        print("Error: Invalid material or material doesn't use nodes")
        return None

    nodes = material.node_tree.nodes

    group_node = None
    for node in nodes:
        if node.type == "GROUP":
            group_node = node
            break

    if not group_node:
        print("Error: Node group not found in material")
        return None

    randomized_values = {
        "Crumple Strength": random.uniform(0.001, 0.003),
        "Crumple Scale": random.uniform(4.0, 10.0),
        "Roughness Ink": random.uniform(0.4, 0.6),
        "Roughness Paper": random.uniform(0.6, 1.0),
    }

    for param_name, value in randomized_values.items():
        if param_name in group_node.inputs:
            group_node.inputs[param_name].default_value = value
            if config.VERBOSE:
                print(f"  {param_name}: {value:.4f}")
        else:
            print(f"Warning: Parameter '{param_name}' not found in node group")

    if config.VERBOSE:
        print("Paper material randomized.")

    return randomized_values


def applyMaterialToObject(objectName, material):
    """Replace every material slot of the object with the given material."""
    obj = bpy.data.objects.get(objectName)
    if not obj:
        raise ValueError(f"Object '{objectName}' not found")

    obj.data.materials.clear()
    obj.data.materials.append(material)

    if config.VERBOSE:
        print(f"Applied material '{material.name}' to '{objectName}'")


def setupPaperTexture(objectName, texture_path):
    """Build the randomized document material and apply it to the paper object."""
    if config.VERBOSE:
        print("\n" + "=" * 50)
        print(f"Setting up texture for '{objectName}'")
        print("=" * 50)

    mat = createPaperMaterial(texture_path)
    randomizePaperMaterial(mat)
    applyMaterialToObject(objectName, mat)

    if config.VERBOSE:
        print("Texture setup complete\n")

    return mat


def createPBRMaterial(material_name, texture_dir):
    """Create the table material from a directory of PBR maps."""

    # Slight variation between samples, so the surface is never twice the same.
    roughness = random.uniform(0.5, 0.7)
    specular = random.uniform(0.25, 0.35)
    rotation = random.uniform(0, 3.1415)
    scale = random.uniform(0.5, 2.0)
    translation_x = random.uniform(-10, 10)
    translation_y = random.uniform(-10, 10)

    if config.VERBOSE:
        print(f"\nCreating pbr surface material '{material_name}'...")

    mat = bpy.data.materials.new(name=material_name)
    mat.use_nodes = True

    tree = mat.node_tree
    nodes = tree.nodes
    links = tree.links

    # A new material comes with a Principled BSDF and an output already wired; both are
    # rebuilt below to control their placement.
    nodes.clear()

    node_principled = nodes.new(type="ShaderNodeBsdfPrincipled")
    node_principled.location = (0, 0)

    node_principled.inputs["Roughness"].default_value = roughness
    node_principled.inputs["Specular IOR Level"].default_value = specular

    node_output = nodes.new(type="ShaderNodeOutputMaterial")
    node_output.location = (300, 0)

    links.new(node_principled.outputs["BSDF"], node_output.inputs["Surface"])

    # Wiring a full PBR set by hand means guessing which file is which map. Node Wrangler
    # already does that matching, so the whole dance below exists to call its operator.
    if "node_wrangler" not in bpy.context.preferences.addons:
        bpy.ops.preferences.addon_enable(module="node_wrangler")

    # The operator reads the active node, so nothing else may be selected.
    node_output.select = False
    node_principled.select = True
    tree.nodes.active = node_principled

    # It also refuses to run outside a Node Editor, so an existing area is temporarily
    # converted into one and pointed at this material.
    area = bpy.context.window.screen.areas[0]
    old_area_type = area.type
    area.type = "NODE_EDITOR"
    area.ui_type = "ShaderNodeTree"
    area.spaces.active.node_tree = tree

    ctx_override = {
        "window": bpy.context.window,
        "screen": bpy.context.window.screen,
        "area": area,
        "space_data": area.spaces.active,
        "region": area.regions[-1],  # The last region is the main one, not a header
        "material": mat,
        "node": node_principled,
        "edit_tree": tree,
        "id": mat,
    }

    # Node Wrangler builds candidate paths as `directory + filename` (plain string
    # concatenation), so `directory` must carry a trailing separator or every texture
    # is discarded as "No matching images found".
    texture_dir_arg = os.path.join(texture_dir, "")
    with bpy.context.temp_override(**ctx_override):
        with suppressOutput():
            bpy.ops.node.nw_add_textures_for_principled(
                filepath=texture_dir_arg,
                directory=texture_dir_arg,
                files=[{"name": f} for f in os.listdir(texture_dir)],
                relative_path=not os.path.isabs(texture_dir),
            )

    # Node Wrangler adds the Mapping node, so it can only be tweaked once it has run.
    nodes["Mapping"].inputs["Scale"].default_value = (scale, scale, 1.0)
    nodes["Mapping"].inputs["Rotation"].default_value = (0.0, 0.0, rotation)
    nodes["Mapping"].inputs["Location"].default_value = (translation_x, translation_y, 0.0)

    area.type = old_area_type

    if config.VERBOSE:
        print(f"Surface texture loaded: {os.path.basename(texture_dir)}")
        print(
            f"Transform - Scale: {scale:.2f}, Rotation: {rotation:.2f}, "
            f"Translation: ({translation_x:.2f}, {translation_y:.2f})"
        )

    if config.VERBOSE:
        print(f"Surface material created (roughness: {roughness:.2f}, specular: {specular:.2f})\n")

    return mat
