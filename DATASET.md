# SyntheticDoc

A synthetic dataset for **document unwarping**. Each sample is a photorealistic render of a document page that has been mapped onto a deformed 3D mesh, together with the ground-truth geometry needed to flatten it back to a clean scan.

## Dataset structure

The dataset is split into three standard subsets:

```
synthetic_doc/
├── train/   # 1,000,000 samples
├── val/     #   100,000 samples
└── test/    #    38,602 samples
```

Within each split, samples are packed into uncompressed zip shards of 1,000 samples each.

## Sample structure

Each shard unpacks to one folder per sample, named by a unique `sample_id` (e.g.
`0237435`) and contains the following files:

| File               | Format              | Resolution    | Description                                                                                                                                                                   |
|--------------------|---------------------|---------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `render.png`       | RGB PNG             | 1024 × 1440   | The final photorealistic render: the document warped on the 3D surface, lit and shadowed. This is the **network input**.                                                      |
| `albedo.png`       | RGB PNG             | 1024 × 1440   | The albedo / diffuse-color pass of the same view — the document content without lighting or shadows.                                                                          |
| `shadow.png`       | RGB PNG             | 1024 × 1440   | The shadow / shading pass, separating illumination from the albedo.                                                                                                           |
| `uv_inverse.exr`   | OpenEXR             | 1024 × 1440   | Per-pixel UV coordinates that tell, for each rendered pixel, where it came from on the flat document. Stored in float EXR to preserve precision.                              |
| `backward_map.npy` | float32 NumPy array | (177, 121, 2) | The **ground-truth backward map**: a coarse 2-channel flow field that maps points from the warped render back to the flat document, used as the unwarping supervision target. |
| `metadata.json`    | JSON                | —             | Generation metadata for the sample (see below).                                                                                                                               |

Sample code used to read a sample is provided in the file `read_sample.py`.

### `metadata.json`

Records how the sample was generated and is useful for reproducibility and
filtering:

- `sample_id` / `sample_index` / `seed` — sample identity and the RNG seed used.
- `timestamp` — when the sample was rendered.
- `files` — the source assets that were combined:
  - `mesh` — the deformed 3D mesh (`.obj`) the page was draped on.
  - `document` — the source document page image (here, an arXiv paper page).
  - `surface_texture` — the background surface material (e.g. a fabric texture).
- `outputs` — relative paths of the rendered output maps.
- `camera` — camera setup: `view_direction`, `inclination_deg`, `azimuth_deg`,
  `roll_deg`, `distance`, and `num_valid_angles`.
- `orientation` — in-plane document orientation in degrees.
- `status` — `success` if the sample rendered correctly.
