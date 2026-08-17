import os
import sys
from contextlib import contextmanager

import bpy


def deselectAll():
    """Deselect every object in the scene."""
    bpy.ops.object.select_all(action="DESELECT")


def selectObject(objectName):
    """Select an object by name and make it the active one, deselecting everything else."""
    deselectAll()
    obj = bpy.data.objects.get(objectName)
    if not obj:
        raise ValueError(f"Object '{objectName}' not found")

    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def setObjectTransform(objectName, location=None, rotation=None, scale=None):
    """Set an object's location, rotation (radians) and scale, leaving out any argument to keep."""
    obj = bpy.data.objects.get(objectName)
    if not obj:
        raise ValueError(f"Object '{objectName}' not found")

    if location is not None:
        obj.location = location
    if rotation is not None:
        obj.rotation_euler = rotation
    if scale is not None:
        obj.scale = scale


def ensureInObjectMode():
    """Leave edit or sculpt mode, since most bpy.ops operators only run in object mode."""
    if bpy.context.object and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")


def setShading(objectName, smooth, auto_smooth, angle):
    """Shade an object smooth or flat, keeping edges sharper than `angle` degrees sharp."""
    obj = bpy.data.objects.get(objectName)
    if not obj:
        raise ValueError(f"Object '{objectName}' not found")

    selectObject(objectName)

    if smooth:
        bpy.ops.object.shade_smooth()

        # Flat shading has no angle to preserve, so auto_smooth only applies to smooth shading.
        if auto_smooth:
            # Blender 4.1+ implements auto smooth as a geometry node modifier, which stacks on
            # every call, so any previous one is removed first.
            for mod in obj.modifiers:
                if mod.type == "NODES" and "Smooth by Angle" in mod.name:
                    obj.modifiers.remove(mod)

            bpy.ops.object.shade_smooth_by_angle(angle=angle * (3.14159 / 180))
    else:
        bpy.ops.object.shade_flat()


@contextmanager
def suppressOutput():
    """Silence the per-tile progress Blender's renderer writes straight to the C-level stdout.

    Redirecting sys.stdout is not enough: the renderer writes to file descriptors 1 and 2 from
    C, bypassing Python entirely, so the descriptors themselves are pointed at /dev/null.
    """
    # Whatever Python has buffered is written out first, since it would otherwise be flushed
    # while the descriptors point at /dev/null and be lost. This happens as soon as stdout is a
    # log file rather than a terminal, where the buffer is not flushed on every line.
    sys.stdout.flush()
    sys.stderr.flush()

    devnull = os.open(os.devnull, os.O_WRONLY)
    saved_fds = {}
    try:
        for fd in (1, 2):
            saved_fds[fd] = os.dup(fd)
            os.dup2(devnull, fd)
        yield
    finally:
        # Recorded as they were taken, so an interrupted setup still restores what it changed.
        for fd, saved_fd in saved_fds.items():
            os.dup2(saved_fd, fd)
            os.close(saved_fd)
        os.close(devnull)
