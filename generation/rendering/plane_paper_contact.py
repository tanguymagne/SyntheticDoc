import bpy
import config
from blender_utils import deselectAll, selectObject, setObjectTransform


def setRigidBody(objectName, rigidBodyType, useMargin, margin):
    """Give an object a rigid body, so the simulation either moves it (ACTIVE) or not (PASSIVE)."""
    selectObject(objectName)

    bpy.ops.rigidbody.objects_add(type=rigidBodyType)

    obj = bpy.data.objects[objectName]
    rb = obj.rigid_body

    rb.friction = config.RIGID_BODY_FRICTION

    rb.use_margin = useMargin
    if useMargin:
        rb.collision_margin = margin

    if config.VERBOSE:
        print(f"  Added {rigidBodyType} rigid body to '{objectName}' (margin: {margin})")


def applyRigidBodySimulation(objectNames, rigidBodyTypes, useMargins, margins):
    """Settle the objects against each other with a physics simulation, and bake the result.

    The four lists describe the objects position by position: names, ACTIVE/PASSIVE, whether
    to keep a collision margin, and how wide that margin is.
    """
    # Checked up front, so a mismatch cannot leave half the objects with a rigid body attached.
    if not len(objectNames) == len(rigidBodyTypes) == len(useMargins) == len(margins):
        raise ValueError(
            "objectNames, rigidBodyTypes, useMargins and margins must have the same length, got "
            f"{len(objectNames)}, {len(rigidBodyTypes)}, {len(useMargins)} and {len(margins)}"
        )

    if config.VERBOSE:
        print(f"\nSimulating contact between {len(objectNames)} objects...")

    for i, objectName in enumerate(objectNames):
        setRigidBody(objectName, rigidBodyTypes[i], useMargins[i], margins[i])

    # Gravity points up so that the table rises to meet the paper, rather than the paper
    # falling onto the table. The paper has to stay exactly where it is: the camera angle was
    # validated against this pose and the camera already aims at it, so any movement would
    # push it out of the frame. Hence the paper is PASSIVE and only the table moves.
    bpy.context.scene.gravity = config.SIMULATION_GRAVITY

    # A rigid body solver advances from the previous frame, so the frames have to be played
    # through in order; jumping straight to the last one would evaluate nothing.
    for i in range(config.SIMULATION_FRAMES):
        bpy.context.scene.frame_set(i)
        bpy.context.view_layer.update()

    if config.VERBOSE:
        print(f"  Simulated {config.SIMULATION_FRAMES} frames, baking the result...")

    # Removing a rigid body snaps its object back to the transform it had before the
    # simulation, so the simulated transform is read out first and reapplied by hand.
    for objectName in objectNames:
        selectObject(objectName)
        matrix_world = bpy.data.objects[objectName].matrix_world.copy()
        bpy.ops.rigidbody.object_remove()
        setObjectTransform(
            objectName,
            scale=matrix_world.to_scale(),
            location=matrix_world.to_translation(),
            rotation=matrix_world.to_euler(),
        )

    bpy.context.scene.frame_set(1)
    bpy.context.view_layer.update()

    deselectAll()

    if config.VERBOSE:
        print("Contact simulation complete\n")
