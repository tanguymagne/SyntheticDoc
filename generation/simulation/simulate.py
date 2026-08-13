"""Run arcsim on generated configs and keep only the meshes needed downstream."""

import argparse
import json
import re
import subprocess
from pathlib import Path


def run_simulation(scenario, id):
    """Simulate config `id` of `scenario` with arcsim and return the output directory."""
    parent_dir = Path(__file__).resolve().parent
    config_path = parent_dir / "configs" / scenario / f"{id:05d}.json"
    save_path = parent_dir / "meshes" / scenario / f"{id:05d}"
    save_path.mkdir(parents=True, exist_ok=True)

    # Override the time step of the config, the faster the collisions the finer it has to be.
    # The fold scenario is slow enough to run with the time step of its base config.
    if scenario != "fold_by_pull":
        with Path(config_path).open("r") as f:
            config = json.load(f)
        if scenario == "fall_on_roof" or scenario == "fall_on_ball":
            config["frame_time"] = 0.001
            config["frame_steps"] = 1
        elif scenario == "fall_on_many":
            config["frame_time"] = 0.0005
            config["frame_steps"] = 1
        elif scenario == "fall_on_roller" or scenario == "curve_by_pull":
            config["frame_time"] = 0.005
            config["frame_steps"] = 5

        with config_path.open("w") as f:
            json.dump(config, f, indent=4)

    arcsim_path = parent_dir / "arcsim-0.2.1" / "bin" / "arcsim"
    cmd = f"{arcsim_path} simulateoffline {config_path} {save_path}"
    subprocess.run(cmd, shell=True, check=True)
    return save_path


def clean_output_directory(path):
    """Keep only the last simulated mesh in `path` and flag the simulation as done."""
    all_res_files = list([f.name for f in path.iterdir()])
    all_res_files.sort(reverse=True)

    # Simulated meshes are named <frame>_<step>.obj, keep only the last one
    pattern = r"^\d{4}_\d{2}\.obj$"
    output_meshes = [f for f in all_res_files if re.match(pattern, f)]

    if output_meshes:
        max_obj = max(output_meshes)
        for f in output_meshes:
            if f != max_obj:
                (path / f).unlink()

    # Drop the arcsim logs and the obstacle meshes
    for res_file in all_res_files:
        if res_file.endswith(".txt"):
            (path / res_file).unlink()
        if res_file.startswith(("obs_", "timing")):
            (path / res_file).unlink()

    # Marker file, so an interrupted simulation is not mistaken for a finished one
    with (path / "done.txt").open("w") as f:
        f.write("ok")


if __name__ == "__main__":
    # Run the simulator on specific ids of a scenario, once their configs are generated
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario", type=str, required=True, help="Scenario name, matching the config generator"
    )
    parser.add_argument(
        "--ids", type=int, nargs="+", required=True, help="IDs of the configs to simulate"
    )
    args = parser.parse_args()

    for id in args.ids:
        print(f"Simulating scenario {args.scenario}, config {id:05d}...")
        save_path = run_simulation(args.scenario, id)
        clean_output_directory(save_path)
