import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

# Blender runs this file by path rather than importing it as part of a package, so its own
# directory is not on sys.path and the sibling module below would not resolve without this.
sys.path.append(str(Path(__file__).resolve().parent))


from single_sample_renderer import generateSample  # noqa: E402


class AssetManager:
    """Decides which mesh, document and background each sample id gets.

    Assets are paired by arithmetic on the sample id rather than drawn at random, so a run can
    be split across machines by id range and still produce the same dataset. Meshes are used in
    blocks of four consecutive ids: the first two photograph the front of the page, the last two
    the back, and all four get a different document.
    """

    def __init__(self, mesh_dir: str, document_dir: str, surface_background_dir: str):
        self.mesh_dir = mesh_dir
        self.document_dir = document_dir
        self.surface_background_dir = surface_background_dir

        self.meshes: List[str] = []
        self.documents: List[str] = []
        self.surface_backgrounds: List[str] = []

        self.shuffled_documents: List[str] = []
        self.shuffled_backgrounds: List[str] = []

        print("Scanning for assets...")
        self.scanAssets()
        self.shuffleAssets()

    def scanAssets(self):
        """Collect every mesh, document and background directory below the three asset roots."""
        self.meshes = self._findFiles(self.mesh_dir, ".obj")
        print(f"Found {len(self.meshes)} meshes")

        self.documents = self._findFiles(self.document_dir, ".png")
        print(f"Found {len(self.documents)} documents")

        self.surface_backgrounds = self._findBackgroundDirs(self.surface_background_dir)
        print(f"Found {len(self.surface_backgrounds)} surface backgrounds")

        # An empty category would otherwise surface much later as a modulo by zero, once the
        # first sample tries to pick from it.
        if not self.meshes:
            raise ValueError(f"No .obj files found in {self.mesh_dir}")
        if not self.documents:
            raise ValueError(f"No .png files found in {self.document_dir}")
        if not self.surface_backgrounds:
            raise ValueError(f"No backgrounds directories found in {self.surface_background_dir}")

    def shuffleAssets(self):
        """Shuffle documents and backgrounds once, so neighbouring sample ids do not look alike.

        The scan returns both lists sorted, which groups documents from the same source next to
        each other. A fixed seed keeps the shuffle identical between runs and machines.
        """
        shuffle_seed = 505

        self.shuffled_documents = self.documents.copy()
        random.Random(shuffle_seed).shuffle(self.shuffled_documents)
        print(f"Shuffled {len(self.shuffled_documents)} documents")

        self.shuffled_backgrounds = self.surface_backgrounds.copy()
        random.Random(shuffle_seed).shuffle(self.shuffled_backgrounds)
        print(f"Shuffled {len(self.shuffled_backgrounds)} surface backgrounds")

    def _findFiles(self, directory: str, extension: str) -> List[str]:
        """Recursively find every file with the given extension, sorted by absolute path."""
        files = []
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                if filename.lower().endswith(extension):
                    files.append(os.path.abspath(os.path.join(root, filename)))
        return sorted(files)

    def _findBackgroundDirs(self, directory: str) -> List[str]:
        """Find the directories holding a PBR material, recognised by containing any image."""
        background_dirs = []
        for root, _, files in os.walk(directory):
            has_textures = any(
                f.lower().endswith((".png", ".jpg", ".jpeg", ".exr", ".hdr")) for f in files
            )
            if has_textures:
                background_dirs.append(os.path.abspath(root))
        return sorted(background_dirs)

    def getMeshForSample(self, sample_id: int) -> str:
        """Pick the mesh for a sample: one mesh per block of four consecutive ids."""
        mesh_index = (sample_id // 4) % len(self.meshes)
        return self.meshes[mesh_index]

    def getOrientationForSample(self, sample_id: int) -> int:
        """Return 0 for the first two ids of a mesh's block and 180 for the last two."""
        position_in_group = sample_id % 4
        return 180 if position_in_group >= 2 else 0

    def getDocumentForSample(self, sample_id: int) -> str:
        """Pick the document for a sample, walking the shuffled list one per sample."""
        mesh_index = sample_id // 4
        position_in_group = sample_id % 4

        doc_index = (mesh_index * 4 + position_in_group) % len(self.shuffled_documents)

        return self.shuffled_documents[doc_index]

    def getSurfaceBackgroundForSample(self, sample_id: int) -> str:
        """Pick the background for a sample, cycling through the shuffled background directories."""
        background_index = sample_id % len(self.shuffled_backgrounds)

        # Trailing separator: the background directory is read as a directory downstream.
        return self.shuffled_backgrounds[background_index] + "/"

    def getSampleConfig(self, sample_id: int) -> Dict[str, Any]:
        """Gather every asset choice for one sample into a single dict."""
        return {
            "sample_id": sample_id,
            "mesh_path": self.getMeshForSample(sample_id),
            "document_path": self.getDocumentForSample(sample_id),
            "surface_background_path": self.getSurfaceBackgroundForSample(sample_id),
            "orientation": self.getOrientationForSample(sample_id),
        }


class DatasetGenerator:
    """Renders a range of sample ids into an output directory."""

    def __init__(
        self, mesh_dir: str, document_dir: str, surface_background_dir: str, output_dir: str
    ):
        self.asset_manager = AssetManager(mesh_dir, document_dir, surface_background_dir)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generateSamples(self, start: int, end: int):
        """Render every sample id from start (inclusive) to end (exclusive)."""
        total = end - start
        print(f"\nGenerating samples {start} to {end - 1} ({total} total)")
        print("=" * 80)

        successful = 0
        failed = 0

        for sample_id in range(start, end):
            try:
                config = self.asset_manager.getSampleConfig(sample_id)

                generateSample(
                    sample_id=sample_id,
                    mesh_path=config["mesh_path"],
                    document_path=config["document_path"],
                    background_path=config["surface_background_path"],
                    flip_mesh=config["orientation"] == 180,
                    output_base_dir=self.output_dir,
                    camera_distance=0.6,
                )

                # The renderer knows nothing about the four-sample blocks, so the orientation is
                # added to the metadata it just wrote rather than passed down to it.
                sample_dir = os.path.join(self.output_dir, str(sample_id).zfill(7))
                metadata_path = os.path.join(sample_dir, "metadata.json")
                with open(metadata_path, "r") as f:
                    meta = json.load(f)
                meta["orientation"] = config["orientation"]
                with open(metadata_path, "w") as f:
                    json.dump(meta, f, indent=2)

                successful += 1

            except Exception as e:
                # One unusable asset must not end a run of thousands. The sample's own
                # metadata.json already records what went wrong.
                failed += 1
                print(f"Failed sample {str(sample_id).zfill(7)} ({type(e).__name__}): {e}")
                continue

        print("\n" + "=" * 80)
        print("Generation complete!")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Total: {successful + failed}")
        print("=" * 80)


def main():
    # Blender consumes the arguments before "--" itself and passes the rest through untouched,
    # so this script only ever parses what follows that separator.
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Generate crumpled paper dataset")
    parser.add_argument(
        "--mesh-dir",
        default="../meshes",
        type=str,
        required=False,
        help="Directory containing mesh files",
    )
    parser.add_argument(
        "--document-dir",
        default="../documents",
        type=str,
        required=False,
        help="Directory containing document textures",
    )
    parser.add_argument(
        "--background-dir",
        default="./backgrounds",
        type=str,
        required=False,
        help="Directory containing backgrounds textures",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./renders", help="Output directory for dataset"
    )
    parser.add_argument(
        "--start", type=int, required=True, help="First sample ID to generate (inclusive)"
    )
    parser.add_argument(
        "--end", type=int, required=True, help="Last sample ID to generate (exclusive)"
    )

    args = parser.parse_args(argv)

    generator = DatasetGenerator(
        mesh_dir=args.mesh_dir,
        document_dir=args.document_dir,
        surface_background_dir=args.background_dir,
        output_dir=args.output_dir,
    )

    generator.generateSamples(args.start, args.end)


if __name__ == "__main__":
    main()
