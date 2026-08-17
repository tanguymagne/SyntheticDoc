import sys
from pathlib import Path

# Blender runs this file by path rather than importing it as part of a package, so its own
# directory is not on sys.path and the sibling modules below would not resolve without this.
sys.path.append(str(Path(__file__).resolve().parent))
import argparse
import json
import random
import subprocess
from datetime import datetime

import bpy
import camera_angle_sampler
import camera_setup
import config
import environment_setup
import ground_truth_module
import lighting_setup
import material_handler
import mesh_loader
import plane_paper_contact
import render_utils
import scene_setup
import uv_unwrap
from blender_utils import suppressOutput


class NoValidCameraAngleError(Exception):
    """Raised when no valid camera angles exist for a mesh."""


def generateSample(
    sample_id: int,
    mesh_path: str,
    document_path: str,
    background_path: str,
    flip_mesh: bool,
    output_base_dir: str,
    camera_distance: float,
    save_blend_file: bool = False,
    compress_pngs: bool = False,
):
    """Build a scene from one mesh, document and background, render it with its ground truth.

    Everything lands in <output_base_dir>/<7-digit sample id>/, and the returned metadata is the
    same dict written there as metadata.json. sample_id doubles as the random seed, so the same
    id and assets reproduce the sample exactly.
    """
    sample_id_str = str(sample_id).zfill(7)

    sample_output_dir = Path(output_base_dir) / sample_id_str
    sample_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 80}")
    print(f"Generating Sample {sample_id_str}")
    print(f"{'=' * 80}")
    print(f"Mesh       : {mesh_path}")
    print(f"Document   : {document_path}")
    print(f"Background : {background_path}")
    print(f"Output     : {sample_output_dir}")
    print(f"{'=' * 80}\n")

    metadata = {
        "sample_id": sample_id_str,
        "sample_index": sample_id,
        "seed": sample_id,
        "timestamp": datetime.now().isoformat(),
        "files": {"mesh": mesh_path, "document": document_path, "surface_texture": background_path},
        "outputs": {},
    }

    try:
        # Only place a seed is set, so every random operation below derives from it
        random.seed(sample_id)

        print("[1/10] Preparing scene...")
        scene_setup.prepareScene()

        print("[2/10] Loading paper mesh...")
        paper_obj = mesh_loader.loadPaperMesh(mesh_path)

        print("[3/10] UV unwrapping...")
        uv_unwrap.unwrapUVMap(paper_obj.name)
        uv_unwrap.ensureCorrectUVOrientation(paper_obj.name)
        if flip_mesh:
            mesh_loader.flipPaperToBackSide(paper_obj)

        print("[4/10] Validating and selecting camera angle...")

        # Narrowed to a named exception, so a caller generating a dataset can tell a mesh that
        # simply cannot be framed from a genuine failure and skip it.
        try:
            valid_angles = camera_angle_sampler.getValidCameraAngles(
                mesh_obj=paper_obj,
                target_location=config.PAPER_LOCATION,
                distance=camera_distance,
                debug=False,  # set True to have each rejected angle explain itself
            )
        except ValueError as e:
            # Chained, so the visibility check's own message survives in the traceback.
            raise NoValidCameraAngleError(str(e)) from e

        selected_angle = valid_angles[random.randint(0, len(valid_angles) - 1)]
        inclination_deg, azimuth_deg = camera_angle_sampler.viewDirectionToSpherical(selected_angle)

        camera_roll_deg = random.uniform(*config.CAMERA_ROLL_RANGE_DEG)

        if config.VERBOSE:
            print(f"  Found {len(valid_angles)} valid camera angles (after visibility check)")
            print(
                "  Selected angle: "
                f"[{selected_angle[0]:.4f}, {selected_angle[1]:.4f}, {selected_angle[2]:.4f}]"
            )
            print(
                f"  Spherical coords: "
                f"inclination={inclination_deg:.2f}°, azimuth={azimuth_deg:.2f}°"
            )
            print(f"  Camera roll: {camera_roll_deg:.2f}°")

        metadata["camera"] = {
            "view_direction": selected_angle.tolist(),
            "inclination_deg": float(inclination_deg),
            "azimuth_deg": float(azimuth_deg),
            "roll_deg": float(camera_roll_deg),
            "num_valid_angles": len(valid_angles),
            "distance": camera_distance,
        }

        print("[5/10] Applying document texture...")
        material_handler.setupPaperTexture(paper_obj.name, document_path)

        print("[6/10] Setting up environment...")
        table_obj = environment_setup.createTableSurface(table_texture_path=background_path)

        print("[7/10] Setting up lighting...")
        _ = lighting_setup.setupRandomLighting()
        lighting_setup.adjustWorldLighting(
            strength=config.WORLD_LIGHT_STRENGTH, color=config.WORLD_LIGHT_COLOR
        )

        print("[8/10] Setting up camera with validated angle...")
        _ = camera_setup.setupCamera(
            camera_name=config.CAMERA_OBJECT_NAME,
            target_location=config.PAPER_LOCATION,
            view_direction=selected_angle,
            distance=camera_distance,
            roll_deg=camera_roll_deg,
        )

        print("[9/10] Running rigid body simulation...")
        plane_paper_contact.applyRigidBodySimulation(
            objectNames=[paper_obj.name, table_obj.name],
            rigidBodyTypes=["PASSIVE", "ACTIVE"],
            useMargins=[True, True],
            margins=[0.001, 0.001],
        )

        print("[10/10] Rendering outputs...")
        # Saved before the renders, since the ground truth passes overwrite the paper's material
        # with the maps they paint and would leave the scene unopenable as the sample it made.
        if save_blend_file:
            blend_file_path = str(sample_output_dir / "scene.blend")
            with suppressOutput():
                bpy.ops.wm.save_as_mainfile(filepath=blend_file_path, compress=True, copy=True)
            metadata["outputs"]["blend_file"] = blend_file_path

        main_render_path = str(sample_output_dir / "render.png")
        render_utils.renderImage(main_render_path)
        metadata["outputs"]["render"] = main_render_path

        shadow_path = str(sample_output_dir / "shadow.png")
        ground_truth_module.renderShadowMap(output_path=shadow_path)
        metadata["outputs"]["shadow_map"] = shadow_path

        albedo_path = str(sample_output_dir / "albedo.png")
        ground_truth_module.renderAlbedoMap(output_path=albedo_path, texture_path=document_path)
        metadata["outputs"]["albedo_map"] = albedo_path

        uv_path = str(sample_output_dir / "uv_inverse.exr")
        ground_truth_module.renderUVInverseMap(output_path=uv_path)
        metadata["outputs"]["uv_inverse_map"] = uv_path

        map_3d_path = str(sample_output_dir / "3d.exr")
        ground_truth_module.render3DMap(output_path=map_3d_path)
        metadata["outputs"]["3d_map"] = map_3d_path

        normal_path = str(sample_output_dir / "normal.exr")
        ground_truth_module.renderNormalMap(output_path=normal_path)
        metadata["outputs"]["normal_map"] = normal_path

        # Compress png files to reduce size
        if compress_pngs:
            subprocess.run(["oxipng", "-o", "max", "-r", sample_output_dir])

        metadata["status"] = "success"
        print(f"\n✓ Sample {sample_id_str} completed successfully!")

    except Exception as e:
        # Record why the sample failed, then let the caller decide what to do with it:
        # dataset_generator.py skips and moves on, an interactive run gets a traceback.
        metadata["status"] = "failed"
        metadata["error"] = str(e)
        metadata["error_type"] = type(e).__name__
        print(f"\n✗ Sample {sample_id_str} failed ({metadata['error_type']}): {e}")
        raise

    finally:
        # Written on success and failure alike, so a failed sample still leaves a record
        # on disk explaining why.
        metadata_path = sample_output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        metadata["outputs"]["metadata"] = str(metadata_path)

    return metadata


def main():
    # Blender consumes the arguments before "--" itself and passes the rest through untouched,
    # so this script only ever parses what follows that separator.
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Render a single dataset sample")
    parser.add_argument("--mesh-path", type=str, required=True, help="Path to the .obj mesh")
    parser.add_argument(
        "--document-path", type=str, required=True, help="Path to the document .png"
    )
    parser.add_argument(
        "--background-path",
        type=str,
        required=True,
        help="Path to the background material directory",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./renders", help="Directory the sample is written to"
    )
    parser.add_argument(
        "--sample-id", type=int, default=0, help="ID of the sample, also used as the random seed"
    )
    parser.add_argument(
        "--camera-distance", type=float, default=0.6, help="Distance from the page to the camera"
    )
    parser.add_argument(
        "--flip-mesh", action="store_true", help="Render the back side of the page"
    )
    parser.add_argument(
        "--save-blend-file", action="store_true", help="Also save the Blender scene"
    )
    parser.add_argument(
        "--compress-pngs", action="store_true", help="Compress the PNG outputs with oxipng"
    )

    args = parser.parse_args(argv)

    sample_metadata = generateSample(
        sample_id=args.sample_id,
        mesh_path=args.mesh_path,
        document_path=args.document_path,
        background_path=args.background_path,
        flip_mesh=args.flip_mesh,
        output_base_dir=args.output_dir,
        camera_distance=args.camera_distance,
        save_blend_file=args.save_blend_file,
        compress_pngs=args.compress_pngs,
    )

    print("\nSample metadata:")
    print(json.dumps(sample_metadata, indent=2))


# Renders one sample. Failures print their cause and propagate as a traceback.
if __name__ == "__main__":
    main()
