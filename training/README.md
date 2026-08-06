# 🏋️ Training & Inference

This folder contains everything needed to train a simple document unwarping and illumination correction model on the SyntheticDoc dataset and to run the pretrained model on your own images.

The architecture of the model is based on [UVDoc](https://github.com/tanguymagne/UVDoc), extended with a second decoder for the shadow map. From a single photo of a warped document, the model predicts:
- a **backward map**, a `2 × H × W` grid in `[-1, 1]`, where each cell says where in the input image the corresponding point of the flat document is.
- a **shadow map**, a `3 × H × W` image at the input resolution, representing the illumination of the page. Dividing the input by it removes the shading.

The folder is organised as follows:
```
training/
├── configs/            # Training configurations
│   ├── uvdoc_31x45.yaml
│   └── uvdoc_121x177.yaml
├── data/               # Where the dataset shards go
├── model/              # Where the pretrained checkpoints go
├── dataset.py          # WebDataset pipeline reading the released zip shards
├── model.py            # UVDocNet: shared encoder, backward map head, shadow head
├── losses.py           # Backward map, shadow and reconstruction losses
├── unwarp.py           # Bilinear unwarping of an image given a backward map
├── train.py            # Training entry point
├── inference.py        # Inference entry point
└── requirements.txt    # Python dependencies
```

---

## 📋 Table of Contents
- [🛠️ Installation](#%EF%B8%8F-installation)
- [📁 Data](#-data)
- [🚀 Training](#-training)
- [▶️ Inference](#%EF%B8%8F-inference)

---

## 🛠️ Installation

We advise creating a virtual environment. Install PyTorch first, picking the build that matches your CUDA version from the [official instructions](https://pytorch.org/get-started/locally/), then install the remaining dependencies:

```bash
conda create -n syntheticdoc-train python=3.12 -y
conda activate syntheticdoc-train
pip install torch torchvision            # see pytorch.org for the right CUDA build
pip install -r requirements.txt
```

**PyTorch 2.10 or newer is required**. The code was tested with Python 3.12, PyTorch 2.10 and Lightning 2.6.
<!-- PyTorch 2.10 or newer is required because the training script relies on `torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch`, which earlier versions do not provide -->

---

## 📁 Data

Download the SyntheticDoc dataset [here](https://doi.org/10.3929/ethz-c-000801994) and place the zip shards under `data/`, so that the folder looks as follows:

```
training/data/
├── train/    train_000.zip … train_999.zip
├── val/      val_000.zip   … val_099.zip
└── test/     test_000.zip  … test_038.zip
```

**⚠️ The data shards are read directly, no unzipping is needed.**

Each shard holds 1,000 samples. The full dataset is not required to get started: pointing the config at a handful of shards is the quickest way to check that everything is set up correctly.

---

## 🚀 Training

To train a model, run:

```bash
python train.py --config configs/uvdoc_31x45.yaml
```

Two configurations are provided, differing in the resolution of the input image and of the output backward map. The first one is a good starting point: it is much lighter to train, and its coarse grid is already enough to unwarp most documents.

| Config                 | Input resolution | Backward map |
|------------------------|------------------|--------------|
| `uvdoc_31x45.yaml`     | 720 × 512        | 31 × 45      |
| `uvdoc_121x177.yaml`   | 1440 × 1024      | 121 × 177    |

### Configuration

Everything is set in the YAML config. The keys most worth adjusting:

| Key                                          | Description                                                                              |
|----------------------------------------------|------------------------------------------------------------------------------------------|
| `data.train_shard_pattern`, `val_shard_pattern` | Shards to train and validate on. Accept brace ranges (`data/train/train_{000..999}.zip`). |
| `data.image_size`                            | `[H, W]` the input is resized to.                                                         |
| `data.bm_out_size`                           | Resolution of the predicted backward map.                            |
| `data.crop_input`                            | Crop the input around the document before resizing, with a small random margin.           |
| `data.augmentation`                          | Photometric augmentations (brightness, contrast, gamma, blur, noise, …).                   |
| `data.num_workers`, `data.shuffle_buffer`    | Dataloader workers and the shard shuffling buffer. A larger buffer means better shuffling and more memory. |
| `training.batch_size`                        | Batch size **per GPU**.                                                                   |
| `training.loss.*`                            | Weights of the backward map, shadow and reconstruction losses, and `l1` / `l2` choice.    |
| `logging.experiment_name`                    | Name of the run, used for the checkpoint and log directories.                             |

An epoch is one pass over all the samples of the given shards, estimated as 1,000 samples per shard.

### Multiple GPUs

Training uses DDP, launched through `torchrun`:

```bash
torchrun --nproc_per_node=4 train.py --config configs/uvdoc_31x45.yaml --gpus 4
```

Shards are distributed over the ranks, then over the workers of each rank. Since `training.batch_size` is per GPU, the effective batch size scales with the number of GPUs.

### Resuming

To resume from the latest checkpoint of a run, use `--resume auto`. This picks up `checkpoints/<experiment_name>/last.ckpt`, falling back to the highest epoch checkpoint. A specific checkpoint can be given instead of `auto`.

```bash
python train.py --config configs/uvdoc_31x45.yaml --resume auto
```

### Outputs

```
training/
├── checkpoints/<experiment_name>/
│   ├── epoch_<n>/<n>.ckpt      # One checkpoint per epoch
│   └── last.ckpt               # Latest checkpoint, used by --resume auto
└── logs/<experiment_name>/
    ├── metrics.csv             # Training and validation losses
    ├── hparams.yaml            # Config the run was started with
    └── epoch_<n>/<sample_id>/  # Validation visualisations
```

After each validation epoch, one batch is written out to inspect how the model is doing: the input (`input.png`), the predicted backward map as an array (`bm_pred.npy`) and drawn as a grid over the input (`bm_grid.png`), the predicted shadow map (`shadow.png`), and the input unwarped (`unwarped.png`) and then relit (`unwarped_shadow_removed.png`).

---

## ▶️ Inference

The two pretrained models should be downloaded and placed in the `model` folder:
- [Original 31 × 45 model](https://igl.ethz.ch/projects/SyntheticDoc/original_31x45.ckpt)
- [High-resolution 121 × 177 model](https://igl.ethz.ch/projects/SyntheticDoc/highres_121x177.ckpt)

To unwarp a folder of images with a trained model:

```bash
python inference.py \
    --checkpoint model/original_31x45.ckpt \
    --input-dir  path/to/images \
    --output-dir path/to/results
```

| Option                | Description                                                                                |
|-----------------------|--------------------------------------------------------------------------------------------|
| `--no-shadow-removal` | Only unwarp, without dividing the shadow map out of the result.                            |
| `--device`            | `cuda` (default) or `cpu`.                                                                 |

All the PNG and JPG images in `--input-dir` are unwarped and written to `--output-dir` under the same name.
<!-- Unwarping is done at the **original resolution** of each image: the backward map is predicted at the model resolution, then upsampled before being applied, so no detail is lost. The model hyperparameters and the resolution it was trained at are read back from the checkpoint, so nothing else needs to be specified. -->

Before being divided out, the predicted shadow map is attenuated on inputs that are already bright, so that correcting them does not wash the page out. This is an inference-time heuristic.

The released `original_31x45.ckpt` model was trained on inputs cropped to the page, with only a small margin of background around it (`crop_input: true`). On a photo where the document covers only a small part of the frame, the predicted backward map is considerably worse. Therefore, you should crop your images around the document before running inference when using this model. The crop does not need to be tight, a rough one is enough.
