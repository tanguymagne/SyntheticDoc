import bmesh
import bpy
import config
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector


def sampleViewDirections(
    inclinations_deg=config.CAMERA_INCLINATIONS_DEG,
    azimuth_range_deg=config.CAMERA_AZIMUTH_RANGE_DEG,
    azimuth_counts=config.CAMERA_AZIMUTH_COUNTS,
):
    """Sample view directions on a sphere, spread over the given inclination and azimuth ranges.

    A view direction points FROM the target TO the camera. The paper sits at the origin and
    faces -Y, so in this convention:
    - inclination is measured from the zenith: 0° = top-down, 90° = horizontal
    - azimuth 0° = camera in -Y, i.e. the reader position; +90° = right (+X); ±180° = behind
    """
    assert len(inclinations_deg) == len(azimuth_counts)

    azimuth_min, azimuth_max = azimuth_range_deg

    dirs = []

    for inc, n_az in zip(inclinations_deg, azimuth_counts):
        inc_rad = np.deg2rad(inc)
        z = np.cos(inc_rad)
        r = np.sin(inc_rad)

        if n_az == 1:
            # A single view sits at the center of the azimuth range.
            azimuth_center = (azimuth_min + azimuth_max) / 2.0
            azimuth_rad = np.deg2rad(azimuth_center)

            # Rotate by -90°, since azimuth 0° is -Y while atan2 measures from +X.
            theta = azimuth_rad - np.pi / 2.0

            x = r * np.cos(theta)
            y = r * np.sin(theta)
            dirs.append(np.array([x, y, z]))
            continue

        for i in range(n_az):
            azimuth = azimuth_min + (azimuth_max - azimuth_min) * i / (n_az - 1)
            azimuth_rad = np.deg2rad(azimuth)
            theta = azimuth_rad - np.pi / 2.0
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            dirs.append(np.array([x, y, z]))

    dirs = np.vstack(dirs)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return dirs


def isMeshVisibleFromCamera(
    mesh_obj,
    camera_obj,
    margin=0.02,
    normal_angle_threshold_deg=120,
    min_normal_consistency=0.96,
    debug=False,
):
    """Check that the mesh fits in the camera frame and that it is entirely seen (no foldover).

    Args:
        margin: fraction of the frame kept free on each edge, as a safety border
        normal_angle_threshold_deg: max angle between the view direction and a vertex normal
            for that vertex to count as facing the camera
        min_normal_consistency: fraction of the vertices that must face the same way
    """

    scene = bpy.context.scene

    cam_direction = camera_obj.matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1.0))

    # Bake the object transform into the copy, so the vertices below are in world space.
    bm = bmesh.new()
    bm.from_mesh(mesh_obj.data)
    bm.transform(mesh_obj.matrix_world)

    ct_facing_camera = 0
    ct_facing_away = 0

    min_x, max_x = float("inf"), float("-inf")
    min_y, max_y = float("inf"), float("-inf")

    normal_threshold_rad = np.deg2rad(normal_angle_threshold_deg)

    for v in bm.verts:
        # Normalized device coordinates: (0, 0) is the bottom left of the frame, (1, 1) the top
        # right, so anything outside [0, 1] falls off the rendered image.
        co_ndc = world_to_camera_view(scene, camera_obj, v.co)

        min_x = min(min_x, co_ndc.x)
        max_x = max(max_x, co_ndc.x)
        min_y = min(min_y, co_ndc.y)
        max_y = max(max_y, co_ndc.y)

        if (
            co_ndc.x < margin
            or co_ndc.x > (1.0 - margin)
            or co_ndc.y < margin
            or co_ndc.y > (1.0 - margin)
        ):
            if debug:
                # The remaining vertices are left unscanned, so report the offending one
                # rather than bounds that would only cover part of the mesh.
                print(
                    f"    REJECTED: Vertex out of bounds - "
                    f"NDC ({co_ndc.x:.3f}, {co_ndc.y:.3f}), margin={margin:.3f}"
                )
            bm.free()
            return False  # Mesh extends outside the camera frame

        world_normal = mesh_obj.matrix_world.to_3x3() @ v.normal
        angle_to_camera = cam_direction.angle(world_normal)

        if angle_to_camera < normal_threshold_rad:
            ct_facing_camera += 1
        if angle_to_camera > np.deg2rad(180 - normal_angle_threshold_deg):
            ct_facing_away += 1

    total_verts = len(bm.verts)
    bm.free()

    # Vertices facing both ways mean the sheet folds back onto itself and occludes part of
    # the document, so whichever side is in the minority must stay negligible.
    min_count = min(ct_facing_camera, ct_facing_away)

    if min_count / total_verts > (1.0 - min_normal_consistency):
        if debug:
            print(
                f"    REJECTED: Normal inconsistency - "
                f"facing_camera={ct_facing_camera}/{total_verts} "
                f"({ct_facing_camera / total_verts * 100:.1f}%), "
                f"facing_away={ct_facing_away}/{total_verts} "
                f"({ct_facing_away / total_verts * 100:.1f}%), "
                f"min_consistency={min_normal_consistency * 100:.1f}%"
            )
        return False

    if debug:
        print(
            f"    ACCEPTED - NDC X=[{min_x:.3f}, {max_x:.3f}], Y=[{min_y:.3f}, {max_y:.3f}], "
            f"facing_camera={ct_facing_camera / total_verts * 100:.1f}%"
        )

    return True


def getValidCameraAngles(
    mesh_obj,
    target_location=(0, 0, 0),  # The point the camera looks at, usually the center of the mesh
    distance=2.0,  # The distance from the target to the camera along the view direction
    temp_camera_name="_temp_validation_camera",
    debug=False,
):
    """Return the view directions from which the mesh is entirely visible, unoccluded and in frame.

    Every candidate direction is checked through an actual camera rather than judged on the
    geometry alone, so the frame borders and the projection are taken into account.
    """

    view_dirs = sampleViewDirections()

    if debug:
        print(f"\n  Testing {len(view_dirs)} candidate camera angles...")

    valid_angles = []

    if temp_camera_name in bpy.data.objects:
        temp_camera = bpy.data.objects[temp_camera_name]
    else:
        temp_camera_data = bpy.data.cameras.new(name=temp_camera_name)
        temp_camera = bpy.data.objects.new(temp_camera_name, temp_camera_data)
        bpy.context.collection.objects.link(temp_camera)

    original_camera = bpy.context.scene.camera

    try:
        for i, view_dir in enumerate(view_dirs):
            view_dir_norm = view_dir / np.linalg.norm(view_dir)
            target = np.array(target_location, dtype=np.float64)
            camera_pos = target + view_dir_norm * distance

            temp_camera.location = tuple(camera_pos)

            direction = Vector(target_location) - Vector(camera_pos)

            # Z is the natural up vector, except when looking straight down, where it is
            # parallel to the view and leaves the camera roll undefined.
            view_z_component = abs(view_dir_norm[2])
            up_vector = "Z" if view_z_component <= 0.999 else "Y"

            rot_quat = direction.to_track_quat("-Z", up_vector)
            temp_camera.rotation_euler = rot_quat.to_euler()

            # world_to_camera_view reads the scene camera, and only sees the new pose once
            # the dependency graph has caught up.
            bpy.context.scene.camera = temp_camera
            bpy.context.view_layer.update()

            if debug:
                inc, az = viewDirectionToSpherical(view_dir_norm)
                print(f"  Candidate {i + 1}/{len(view_dirs)}: incl={inc:.1f}°, azim={az:.1f}°")

            if isMeshVisibleFromCamera(mesh_obj, temp_camera, debug=debug):
                valid_angles.append(view_dir_norm.copy())

    finally:
        # Restore the scene as it was found, so a failed check leaves no camera behind.
        if original_camera:
            bpy.context.scene.camera = original_camera

        if temp_camera_name in bpy.data.objects:
            bpy.data.objects.remove(temp_camera, do_unlink=True)

    if debug:
        print(f"\n  Result: {len(valid_angles)}/{len(view_dirs)} angles passed visibility check")

    if len(valid_angles) == 0:
        raise ValueError(
            f"No valid camera angles found for mesh "
            f"(all {len(view_dirs)} candidates failed visibility check)"
        )

    return np.vstack(valid_angles)


def viewDirectionToSpherical(view_dir):
    """Convert a view direction to the (inclination, azimuth) degrees of sampleViewDirections."""
    view_dir = view_dir / np.linalg.norm(view_dir)
    x, y, z = view_dir

    inclination_rad = np.arccos(np.clip(z, -1.0, 1.0))
    inclination_deg = np.rad2deg(inclination_rad)

    # atan2 measures from +X, while azimuth 0° is -Y, hence the +90° offset.
    azimuth_rad_standard = np.arctan2(y, x)
    azimuth_rad = azimuth_rad_standard + np.pi / 2.0

    # Wrap back into [-180°, 180°], which the offset above can overshoot.
    azimuth_rad = np.arctan2(np.sin(azimuth_rad), np.cos(azimuth_rad))
    azimuth_deg = np.rad2deg(azimuth_rad)

    return inclination_deg, azimuth_deg
