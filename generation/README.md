# 🏭 Dataset Generation

This folder contains everything needed to generate the SyntheticDoc dataset, from the deformed paper meshes to the final samples and their ground-truth annotations.

Generation runs in three stages, each in its own folder and with its own README:

| Stage                                  | What it does                                                                                   |
| -------------------------------------- | ---------------------------------------------------------------------------------------------- |
| [`simulation`](simulation)             | Physically simulates sheets of paper and saves the deformed meshes.                            |
| [`rendering`](rendering)               | Textures a document onto each mesh, lights it, and path-traces the render and its annotations. |
| [`backward_mapping`](backward_mapping) | Inverts the rendered UV maps into the coarse backward maps used as unwarping supervision.      |

The stages are meant to be run in that order, each consuming the output of the previous one, but they are independent programs: any of them can be run on its own inputs. In particular, the rendering can be pointed at any collection of sheet meshes, and the backward mapping at any UV map the renderer produced.

They are kept apart because they need entirely different environments: the simulation needs ARCSim compiled from source, the rendering runs inside Blender's own Python interpreter, and the backward mapping is plain NumPy and SciPy.