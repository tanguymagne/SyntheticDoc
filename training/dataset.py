import ctypes
import glob
import io
import json
import random
import zipfile
from pathlib import Path, PurePosixPath
from typing import List, Optional, Tuple

import braceexpand
import numpy as np
import torch
import torch.nn.functional as F
import webdataset as wds
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms
from torchvision.transforms.v2 import functional as TF

# Files read out of a sample folder. The other released files (albedo.png, uv_inverse.exr)
# are not used for training and are never read from the archive, which saves most of the I/O.
RENDER_FILE = "render.png"
SHADOW_FILE = "shadow.png"
BM_FILE = "backward_map.npy"
META_FILE = "metadata.json"

SAMPLE_FILES = frozenset({RENDER_FILE, SHADOW_FILE, BM_FILE, META_FILE})

BM_HEIGHT = 121
BM_WIDTH = 177


def _malloc_trim():
    """Ask glibc to return freed memory to the OS, to keep the dataloader workers small."""
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _make_img_transforms(image_size, normalize):
    """Build the resize / to-tensor / normalise pipeline applied to the input image."""
    transforms_list = []

    if image_size is not None:
        transforms_list.append(transforms.Resize(image_size, antialias=True))

    transforms_list.append(transforms.ToImage())
    transforms_list.append(transforms.ToDtype(torch.float32, scale=True))

    if normalize:
        transforms_list.append(transforms.Normalize([0.5] * 3, [0.5] * 3))

    return transforms.Compose(transforms_list)


def _make_input_only_aug():
    """Augmentations applied to the input alone, i.e. that must not touch the shadow map."""
    return transforms.Compose(
        [
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=0.1),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.01, 2.0))], p=0.25
            ),
            transforms.RandomApply([transforms.GaussianNoise()], p=0.5),
        ]
    )


def _joint_color_aug(inp_t: torch.Tensor, shd_t: torch.Tensor):
    """Random photometric augmentations, applied to the shadow map too when they affect it."""
    if random.random() < 0.6:
        brightness_factor = random.uniform(0.7, 1.4)
        inp_t = TF.adjust_brightness(inp_t, brightness_factor)
        shd_t = TF.adjust_brightness(shd_t, brightness_factor)

    if random.random() < 0.4:
        contrast_factor = random.uniform(0.7, 1.4)
        inp_t = TF.adjust_contrast(inp_t, contrast_factor)
        shd_t = TF.adjust_contrast(shd_t, contrast_factor)

    if random.random() < 0.4:
        saturation_factor = random.uniform(0.8, 1.2)
        inp_t = TF.adjust_saturation(inp_t, saturation_factor)

    if random.random() < 0.2:
        hue_factor = random.uniform(-0.1, 0.1)
        inp_t = TF.adjust_hue(inp_t, hue_factor)

    if random.random() < 0.6:
        gamma = random.uniform(0.45, 1.3)
        inp_t = TF.adjust_gamma(inp_t, gamma)
        shd_t = TF.adjust_gamma(shd_t, gamma)

    if random.random() < 0.1:
        inp_t = TF.autocontrast(inp_t)
        shd_t = TF.autocontrast(shd_t)

    if random.random() < 0.1:
        inp_u8 = (inp_t * 255).to(torch.uint8)
        inp_t = TF.equalize(inp_u8).to(torch.float32) / 255.0

    return inp_t, shd_t


def split_by_rank(src, group=None):
    """Webdataset node splitter: give each distributed rank every world_size-th shard."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        yield from src
        return
    rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
    for i, item in enumerate(src):
        if i % world_size == rank:
            yield item


def expand_shards(pattern: str) -> List[str]:
    """Expand a shard spec into a list of zip shard paths.

    Accepts brace ranges ('data/train/train_{000..999}.zip') or a plain path. Shell globs
    should not be used: `get_number_of_samples` in train.py only understands brace ranges,
    so a glob would silently give the wrong epoch size.
    """
    paths: List[str] = []
    for path in braceexpand.braceexpand(pattern):
        if any(c in path for c in "*?["):
            paths.extend(sorted(glob.glob(path)))
        else:
            paths.append(path)
    if not paths:
        raise ValueError(f"No shard matched {pattern!r}")
    not_zip = [path for path in paths if not path.endswith(".zip")]
    if not_zip:
        raise ValueError(f"Expected .zip shards, got {not_zip}")
    return paths


def _zip_sample_groups(zf: zipfile.ZipFile):
    """Group the members of a zip shard by sample folder, keeping the archive order."""
    groups = {}
    for info in zf.infolist():
        if info.is_dir():
            continue
        path = PurePosixPath(info.filename)
        if path.name not in SAMPLE_FILES:
            continue
        groups.setdefault(str(path.parent), []).append((path.name, info))
    return list(groups.items())


def zipfile_to_samples(shuffle_samples: bool = False, seed: int = 0, handler=wds.warn_and_continue):
    """Webdataset stage turning a stream of dict(url=<zip shard>) into webdataset samples.

    This replaces webdataset's own tar expander. The released shards are uncompressed zips
    holding one folder per sample, so members can be read individually and in any order:
    `shuffle_samples` shuffles the samples inside a shard for free, which a tar-based
    pipeline could only do with a large in-memory buffer.
    """

    def stage(src):
        for shard in src:
            url = shard["url"]
            shard_key = Path(url).name.split(".")[0]
            try:
                with zipfile.ZipFile(url) as zf:
                    groups = _zip_sample_groups(zf)
                    if shuffle_samples:
                        random.Random(f"{seed}:{url}").shuffle(groups)
                    for folder, members in groups:
                        sample = {"__key__": f"{shard_key}/{folder}", "__url__": url}
                        for name, info in members:
                            sample[name] = zf.read(info)
                        yield sample
            except Exception as exn:
                exn.args = exn.args + (url,)
                if handler(exn):
                    continue
                break

    return stage


def decode_bm(bm_bytes: bytes) -> np.ndarray:
    """Decode a backward map from its .npy bytes and return it as HxWx2."""
    bio = io.BytesIO(bm_bytes)
    bm = np.load(bio)
    bio.close()
    if bm.ndim != 3 or (bm.shape[-1] != 2 and bm.shape[0] != 2):
        raise ValueError(f"Unexpected bm shape: {bm.shape}")
    if bm.shape[0] == 2:
        bm = bm.transpose(1, 2, 0)
    return bm.astype(np.float32)


def make_decoder(
    image_size: Optional[Tuple[int, int]] = None,
    normalize: bool = True,
    crop_input: bool = False,
    augmentation: bool = False,
    bm_out_size: Optional[Tuple[int, int]] = None,
):
    """Return the function turning a raw webdataset sample into an (input, bm, shadow) triplet."""
    img_transforms = _make_img_transforms(image_size, normalize=False)
    to_tensor = transforms.Compose(
        [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
    )
    input_only_aug = _make_input_only_aug() if augmentation else None
    _sample_count = [0]

    def decode(sample):
        """Decode one sample, returning None if anything goes wrong so it can be skipped."""
        try:
            inp_bio = io.BytesIO(sample[RENDER_FILE])
            inp_pil = Image.open(inp_bio).convert("RGB")
            orig_w, orig_h = inp_pil.size
            inp_bio.close()

            shd_bio = io.BytesIO(sample[SHADOW_FILE])
            shd_pil = Image.open(shd_bio).convert("RGB")
            shd_bio.close()

            bm_raw = decode_bm(sample[BM_FILE])
            del sample[BM_FILE]

            # NOTE: this branch is taken for every released sample, it is not a rare fallback.
            # The backward maps ship as 177x121, so they are always resampled here to 121x177,
            # that is into a grid transposed with respect to the (portrait) document. This is
            # not correct, but it is how the released models were trained, and the later
            # upsampling in `bilinear_unwarping` resamples back to the aspect ratio of the
            # image, so the round trip mostly cancels out.
            if bm_raw.shape[:2] != (BM_HEIGHT, BM_WIDTH):
                ch0 = np.array(
                    Image.fromarray(bm_raw[:, :, 0], mode="F").resize(
                        (BM_WIDTH, BM_HEIGHT), Image.BILINEAR
                    )
                )
                ch1 = np.array(
                    Image.fromarray(bm_raw[:, :, 1], mode="F").resize(
                        (BM_WIDTH, BM_HEIGHT), Image.BILINEAR
                    )
                )
                bm_raw = np.stack([ch0, ch1], axis=-1)

            key = sample["__key__"]
            sample_id = json.loads(sample[META_FILE]).get("sample_id", key)
            sample.clear()

            _sample_count[0] += 1
            if _sample_count[0] % 100 == 0:
                _malloc_trim()

            if crop_input:
                px = (bm_raw[:, :, 0] + 1) / 2 * orig_w
                py = (bm_raw[:, :, 1] + 1) / 2 * orig_h

                pad_ref = max(orig_h, orig_w)
                if not augmentation:
                    pl = pr = pt = pb = int(0.02 * pad_ref)
                else:
                    pl = int(random.uniform(0.01, 0.03) * pad_ref)
                    pr = int(random.uniform(0.01, 0.03) * pad_ref)
                    pt = int(random.uniform(0.01, 0.03) * pad_ref)
                    pb = int(random.uniform(0.01, 0.03) * pad_ref)

                x_min = max(int(px.min()) - pl, 0)
                x_max = min(int(px.max()) + pr, orig_w)
                y_min = max(int(py.min()) - pt, 0)
                y_max = min(int(py.max()) + pb, orig_h)

                inp_pil = inp_pil.crop((x_min, y_min, x_max, y_max))
                shd_pil = shd_pil.crop((x_min, y_min, x_max, y_max))

                crop_w = x_max - x_min
                crop_h = y_max - y_min

                bm_cropped = bm_raw.copy()
                bm_cropped[:, :, 0] = (px - x_min) / crop_w * 2 - 1
                bm_cropped[:, :, 1] = (py - y_min) / crop_h * 2 - 1
                bm_tensor = torch.from_numpy(bm_cropped.transpose(2, 0, 1).copy())
            else:
                bm_tensor = torch.from_numpy(bm_raw.transpose(2, 0, 1).copy())

            if bm_out_size is not None:
                # NOTE: this is not correct, but is how we trained the model
                th, tw = bm_out_size
                if (th, tw) == (31, 45):
                    bm_tensor = bm_tensor[:, ::4, ::4]
                else:
                    bm_tensor = F.interpolate(
                        bm_tensor.unsqueeze(0), size=(th, tw), mode="bilinear", align_corners=True
                    )[0]

            inp = img_transforms(inp_pil)
            out_size = (
                (image_size[1], image_size[0]) if image_size is not None else (orig_w, orig_h)
            )
            shadow = to_tensor(shd_pil.resize(out_size, Image.BILINEAR))

            if augmentation:
                inp, shadow = _joint_color_aug(inp, shadow)
            if input_only_aug is not None:
                inp = input_only_aug(inp)

            if normalize:
                inp = transforms.Normalize([0.5] * 3, [0.5] * 3)(inp)
                shadow = shadow * 2 - 1

            return {
                "input": inp,
                "bm_gt": bm_tensor,
                "shadow_gt": shadow,
                "sample_id": sample_id,
                "__key__": key,
            }

        except Exception as e:
            print(f"Warning: failed to decode sample {sample.get('__key__', '?')}: {e}")
            return None

    return decode


def _worker_init_fn(worker_id):
    """Limit the allocator arenas of each dataloader worker to keep memory usage bounded."""
    import os

    os.environ["MALLOC_ARENA_MAX"] = "2"
    _malloc_trim()


def make_sample_pipeline(
    shard_pattern: str, shuffle_buffer: int = 0, seed: int = 42, shuffle_within_shard: bool = True
) -> wds.DataPipeline:
    """Build the webdataset pipeline yielding raw samples from the released zip shards.

    Shards are distributed over ranks then over dataloader workers, optionally shuffled, and
    finally expanded into samples by `zipfile_to_samples` instead of webdataset's tar expander.
    """
    urls = expand_shards(shard_pattern)
    stages = [wds.SimpleShardList(urls, seed=seed), split_by_rank, wds.split_by_worker]
    if shuffle_buffer > 0:
        stages.append(wds.shuffle(shuffle_buffer, seed=seed))
    stages.append(
        zipfile_to_samples(shuffle_samples=shuffle_buffer > 0 and shuffle_within_shard, seed=seed)
    )
    return wds.DataPipeline(*stages)


def create_webdataset_loader(
    shard_pattern: str,
    image_size: Optional[Tuple[int, int]] = None,
    normalize: bool = True,
    crop_input: bool = False,
    augmentation: bool = False,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle_buffer: int = 0,
    seed: int = 42,
    epoch_size: Optional[int] = None,
    prefetch_factor: int = 2,
    bm_out_size: Optional[Tuple[int, int]] = None,
    shuffle_within_shard: bool = True,
) -> DataLoader:
    """Build the DataLoader streaming and decoding the zip shards.

    `epoch_size` is the number of samples one rank sees per epoch, split over its workers.
    """
    samples = make_sample_pipeline(shard_pattern, shuffle_buffer, seed, shuffle_within_shard)
    decoder = make_decoder(image_size, normalize, crop_input, augmentation, bm_out_size)
    samples.append(wds.map(decoder))
    samples.append(wds.select(lambda x: x is not None))
    if epoch_size is not None:
        # Set on the sample stream, so epoch_size counts samples and not batches. Every worker
        # runs its own copy of the pipeline, hence the split over the workers of this rank.
        active_workers = max(1, min(num_workers, len(expand_shards(shard_pattern))))
        samples = samples.with_epoch(epoch_size // active_workers)
    dataset = wds.DataPipeline(samples, wds.batched(batch_size, partial=True))
    return DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
    )
