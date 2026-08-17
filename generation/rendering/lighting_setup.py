import math
import random

import bpy
import config
from mathutils import Matrix, Vector

# The lights making up each preset, as (role, object name, config prefix). A prefix stands for
# the <PREFIX>_TYPE, _LOCATION, _ENERGY and _SIZE constants of config.py, which is how a preset
# stays a few lines here instead of a function: adding a light means adding a row and its four
# constants, and nothing else.
LIGHTING_SETUPS = {
    # Key, fill and rim: the classic even-but-modelled studio look.
    "three_point": [
        ("key", "KeyLight", "KEY_LIGHT"),
        ("fill", "FillLight", "FILL_LIGHT"),
        ("rim", "RimLight", "RIM_LIGHT"),
    ],
    # Two strong side lights and a weak front fill, which emphasizes edges and creases.
    "rim": [
        ("rim_left", "RimLeftLight", "RIM_SETUP_LEFT"),
        ("rim_right", "RimRightLight", "RIM_SETUP_RIGHT"),
        ("fill", "RimFillLight", "RIM_SETUP_FILL"),
    ],
    # Large area lights above and to the sides, which gives soft, even shadows.
    "softbox": [
        ("top", "SoftboxTopLight", "SOFTBOX_TOP"),
        ("left", "SoftboxLeftLight", "SOFTBOX_LEFT"),
        ("right", "SoftboxRightLight", "SOFTBOX_RIGHT"),
    ],
    # A strong window light with sky and bounce fill, which simulates daylight.
    "natural": [
        ("sun", "NaturalSunLight", "NATURAL_SUN"),
        ("sky", "NaturalSkyLight", "NATURAL_SKY"),
        ("bounce", "NaturalBounceLight", "NATURAL_BOUNCE"),
    ],
}


def createLight(name, light_type, location, energy, size, color):
    """Create a light of the given type (POINT, SUN, SPOT or AREA) and add it to the scene."""
    light_data = bpy.data.lights.new(name=name, type=light_type)
    light_data.energy = energy
    light_data.color = color

    # Only area lights have a size; the attribute does not exist on the other types.
    if light_type == "AREA":
        light_data.size = size

    light_object = bpy.data.objects.new(name=name, object_data=light_data)
    light_object.location = location

    bpy.context.collection.objects.link(light_object)

    if config.VERBOSE:
        print(f"Created {light_type} light '{name}' at {location}")

    return light_object


def pointLightAtTarget(light_object, target_location):
    """Aim a light at a location."""
    direction = Vector(target_location) - Vector(light_object.location)

    # A light shines along its local -Z, so that axis is the one aimed at the target.
    rot_quat = direction.to_track_quat("-Z", "Y")
    light_object.rotation_euler = rot_quat.to_euler()


def setupLighting(setup_name, color):
    """Create the lights of one LIGHTING_SETUPS preset, aimed at the paper, keyed by role."""
    if config.VERBOSE:
        print(f"\nSetting up '{setup_name}' lighting...")

    lights = {}

    for role, light_name, prefix in LIGHTING_SETUPS[setup_name]:
        lights[role] = createLight(
            name=light_name,
            light_type=getattr(config, f"{prefix}_TYPE"),
            location=getattr(config, f"{prefix}_LOCATION"),
            energy=getattr(config, f"{prefix}_ENERGY"),
            size=getattr(config, f"{prefix}_SIZE"),
            color=color,
        )
        pointLightAtTarget(lights[role], config.PAPER_LOCATION)

    if config.VERBOSE:
        print(f"'{setup_name}' lighting setup complete (color: {color})\n")

    return lights


def pickColorTemperature(temp_type):
    """Pick a random RGB color inside the 'warm', 'cool' or 'neutral' range."""
    if temp_type == "warm":
        min_color, max_color = config.COLOR_TEMP_WARM_RANGE
    elif temp_type == "cool":
        min_color, max_color = config.COLOR_TEMP_COOL_RANGE
    else:  # neutral
        min_color, max_color = config.COLOR_TEMP_NEUTRAL_RANGE

    # Each channel is drawn on its own, so the tint varies as well as the brightness.
    r = random.uniform(min_color[0], max_color[0])
    g = random.uniform(min_color[1], max_color[1])
    b = random.uniform(min_color[2], max_color[2])

    color = (r, g, b)

    if config.VERBOSE:
        print(f"Picked {temp_type} color: {color}")

    return color


def rotateLightingSetup(lights, rotation_angle):
    """Rotate every light of a setup around the paper on the Z axis."""
    rotation_matrix = Matrix.Rotation(rotation_angle, 4, "Z")

    center_vec = Vector(config.PAPER_LOCATION)

    for light_name, light_object in lights.items():
        # Rotating about an arbitrary center means working in coordinates relative to it.
        relative_pos = Vector(light_object.location) - center_vec
        rotated_pos = rotation_matrix @ relative_pos
        light_object.location = center_vec + rotated_pos

        light_object.rotation_euler.rotate(rotation_matrix)

        pointLightAtTarget(light_object, config.PAPER_LOCATION)

        if config.VERBOSE:
            print(f"Rotated {light_name} to {light_object.location}")


def randomRotateLightingSetup(lights):
    """Rotate a lighting setup by a random angle, and return the angle used."""
    rotation_angle = random.uniform(
        config.LIGHT_ROTATION_MIN_ANGLE, config.LIGHT_ROTATION_MAX_ANGLE
    )

    if config.VERBOSE:
        print(
            f"Rotating lighting setup by {rotation_angle:.2f} radians "
            f"({math.degrees(rotation_angle):.1f}°) around Z axis"
        )

    rotateLightingSetup(lights, rotation_angle)

    return rotation_angle


def setupRandomLighting():
    """Build one of the lighting presets at random, in a random color and orientation."""
    setup_name = random.choice(list(LIGHTING_SETUPS.keys()))

    color_temperature = random.choice(["warm", "cool", "neutral"])

    actual_color = pickColorTemperature(color_temperature)

    if config.VERBOSE:
        print(
            f"\nRandomly selected '{setup_name}' lighting "
            f"with '{color_temperature}' color temperature"
        )

    lights = setupLighting(setup_name, actual_color)

    # The presets are fixed arrangements, so rotating them is what keeps the light direction
    # from repeating across the dataset.
    rotation_angle = randomRotateLightingSetup(lights)

    return lights, setup_name, color_temperature, actual_color, rotation_angle


def adjustWorldLighting(strength, color):
    """Set the color and strength of the world background, the scene's ambient light."""
    world = bpy.context.scene.world

    if world.use_nodes:
        for node in world.node_tree.nodes:
            if node.type == "BACKGROUND":
                node.inputs["Color"].default_value = (*color, 1.0)
                node.inputs["Strength"].default_value = strength

                if config.VERBOSE:
                    print(f"Adjusted world lighting: strength={strength}, color={color}")
                break
