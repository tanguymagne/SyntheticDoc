import math

import bpy
import config
import numpy as np
from mathutils import Euler, Vector


def createCamera(name, location):
    """Create a camera with the configured lens and sensor, and add it to the scene."""
    camera_data = bpy.data.cameras.new(name=name)

    camera_data.lens = config.CAMERA_LENS
    camera_data.sensor_width = config.CAMERA_SENSOR_WIDTH

    camera_object = bpy.data.objects.new(name, camera_data)
    camera_object.location = location
    camera_object.rotation_euler = (0, 0, 0)

    bpy.context.collection.objects.link(camera_object)

    if config.VERBOSE:
        print(f"Created camera '{name}' at {location}")

    return camera_object


def setActiveCamera(camera):
    """Make a camera the one the scene renders from."""
    bpy.context.scene.camera = camera

    if config.VERBOSE:
        print(f"Set '{camera.name}' as active camera")


def setupCamera(view_direction, target_location, distance, camera_name, roll_deg):
    """Place a camera along a view direction, aim it at the target, and make it active.

    The view direction points FROM the target TO the camera, matching camera_angle_sampler.
    Roll is in degrees, positive tilting clockwise.
    """
    if config.VERBOSE:
        print(f"\nSetting up camera '{camera_name}'...")

    view_dir = np.array(view_direction, dtype=np.float64)
    view_dir = view_dir / np.linalg.norm(view_dir)

    target = np.array(target_location, dtype=np.float64)
    camera_pos = target + view_dir * distance

    camera = createCamera(name=camera_name, location=tuple(camera_pos))

    direction = Vector(target_location) - Vector(camera_pos)

    # Z is the natural up vector, except when looking straight down, where it is parallel
    # to the view and leaves the camera roll undefined.
    view_z_component = abs(view_dir[2])
    up_vector = "Y" if view_z_component > 0.999 else "Z"

    rot_quat = direction.to_track_quat("-Z", up_vector)
    camera.rotation_euler = rot_quat.to_euler()

    # A handheld photo is never perfectly level, so the frame is tilted slightly.
    if abs(roll_deg) > 0.01:
        roll_rad = math.radians(roll_deg)
        roll_euler = Euler((0, 0, roll_rad), "XYZ")
        camera.rotation_euler.rotate(roll_euler)

    setActiveCamera(camera)

    if config.VERBOSE:
        print(f"Camera positioned at {camera_pos} looking at {target_location}")
        print(f"View direction: {view_dir}, Up vector: {up_vector}, Roll: {roll_deg:.1f}°")
        print("Camera setup complete\n")

    return camera
