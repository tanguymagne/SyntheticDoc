"""Interactive polyscope viewer to inspect the meshes produced by the simulations."""

from pathlib import Path

import polyscope as ps
import polyscope.imgui as psim
import trimesh


class MeshVisualizer:
    """Browse simulated meshes one at a time.

    The meshes are collected recursively, and labelled after the directories
    they sit in, which assumes the <scenario>/<id>/ layout written by simulate.py.
    """

    def __init__(self, root_dir):
        """Collect the meshes to display under `root_dir`."""
        self.root_dir = Path(root_dir)
        self.mesh_files = sorted(self.root_dir.rglob("*.obj"))

        if not self.mesh_files:
            raise ValueError(f"No .obj files found in {root_dir}")

        self.current_idx = 0

        print(f"Found {len(self.mesh_files)} mesh files")
        for f in self.mesh_files[:5]:
            print(" ", f.relative_to(self.root_dir))
        if len(self.mesh_files) > 5:
            print(f"  ... and {len(self.mesh_files) - 5} more")

    def load_and_display_single(self):
        """Display the mesh at the current index, replacing the one on screen."""
        ps.remove_all_structures()

        filepath = self.mesh_files[self.current_idx]
        mesh = trimesh.load(filepath, process=False)
        scenario, number = self.get_mesh_info(filepath)

        mesh_obj = ps.register_surface_mesh(
            f"{scenario}/{number}", mesh.vertices, mesh.faces, smooth_shade=True
        )
        mesh_obj.set_color((1.0, 1.0, 1.0))

    def get_mesh_info(self, filepath):
        """Return the scenario and the config id `filepath` was simulated from, if available."""
        parts = filepath.relative_to(self.root_dir).parts
        scenario = parts[0] if len(parts) > 0 else "unknown"
        number = parts[1] if len(parts) > 1 else "unknown"
        return scenario, number

    def ui_callback(self):
        """Draw the navigation panel, called by polyscope at every frame."""
        psim.TextUnformatted(f"Total meshes: {len(self.mesh_files)}")
        psim.TextUnformatted(f"Current index: {self.current_idx}")

        if psim.Button("Previous"):
            self.current_idx = (self.current_idx - 1) % len(self.mesh_files)
            self.update_display()

        psim.SameLine()
        if psim.Button("Next"):
            self.current_idx = (self.current_idx + 1) % len(self.mesh_files)
            self.update_display()

        changed, new_idx = psim.SliderInt("Jump to", self.current_idx, 0, len(self.mesh_files) - 1)
        if changed:
            self.current_idx = new_idx
            self.update_display()

        psim.Separator()
        filepath = self.mesh_files[self.current_idx]
        scenario, number = self.get_mesh_info(filepath)
        psim.TextUnformatted(f"Scenario: {scenario}")
        psim.TextUnformatted(f"Number: {number}")
        psim.TextUnformatted(f"File: {filepath.name}")

    def update_display(self):
        """Refresh the scene after the current index changed."""
        self.load_and_display_single()

    def run(self):
        """Open the viewer and block until its window is closed."""
        ps.init()
        ps.set_user_callback(self.ui_callback)
        ps.set_ground_plane_mode("none")

        self.update_display()
        ps.show()


if __name__ == "__main__":
    # Path is relative to the working directory, run this from generation/simulation
    visualizer = MeshVisualizer("./meshes/")
    visualizer.run()
