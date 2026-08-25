# 🔄 Backward Mapping

This folder contains the last step of the generation pipeline: turning the UV maps produced by the [rendering](../rendering) into the **backward maps** that document unwarping networks are trained to predict.

The two are inverses of each other. The rendered `uv_inverse.exr` answers *"which point of the flat page does this pixel of the photograph show?"*, for every pixel of the render. A backward map answers the opposite question, *"where in the photograph is this point of the flat page?"*, on a coarse regular grid laid over the flat document. Sampling the photograph through that grid is what unwarps it, so this is the supervision signal used at training time.

This step is kept apart from the rendering because it does not involve Blender: it is plain NumPy and SciPy, and runs on the rendered EXR files after the fact.

---

## 📋 Table of Contents
- [🛠️ Installation](#️-installation)
- [🚀 Usage](#-usage)
- [🖼️ Output](#️-output)

---

## 🛠️ Installation

```bash
conda create -n syntheticdoc-bm python=3.12 -y
conda activate syntheticdoc-bm
pip install -r requirements.txt
```

---

## 🚀 Usage

The script takes either a single EXR file:

```bash
python uv_to_backward_map.py --input ../rendering/renders/0000000/uv_inverse.exr
```

or a directory, which is searched recursively for every UV map inside it:

```bash
python uv_to_backward_map.py --input ../rendering/renders
```

| Option      | Description                                                              |
| ----------- | ------------------------------------------------------------------------ |
| `--input`   | UV map to convert, or directory to search. **Required.**                 |
| `--pattern` | Name of the UV maps looked for in a directory. Default `uv_inverse.exr`. |
| `--size`    | Height and width of the backward map grid. Default `177 121`.            |

The default `--pattern` matches the name the renderer gives its UV maps, so that the other EXR annotations sitting next to them (`3d.exr`, `normal.exr`) are left out.

---

## 🖼️ Output

Each backward map is written as `backward_map.npy` next to the UV map it comes from:

```
renders/0000000/
├── uv_inverse.exr
└── backward_map.npy
```

It is a `float32` array of shape `(height, width, 2)`, holding the `(x, y)` position of each grid point in the rendered image, normalized to `[-1, 1]`. That is the convention of [`grid_sample`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html), so the render can be unwarped by sampling it through the map directly, as done in [`training/unwarp.py`](../../training/unwarp.py).
