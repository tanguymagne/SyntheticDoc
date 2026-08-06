import argparse
import os
import sys
import warnings
from pathlib import Path

import yaml

warnings.filterwarnings("ignore", message=".*LeafSpec.*is deprecated.*")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

import lightning as L
import torch
import torch.autograd

torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)
torch.set_float32_matmul_precision("high")
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.plugins.environments import TorchElasticEnvironment
from lightning.pytorch.strategies import DDPStrategy

sys.path.insert(0, str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parents[1]))

import random

import braceexpand
import numpy as np
import torch.nn.functional as F
from dataset import create_webdataset_loader
from losses import MultiObjectiveLoss
from model import UVDocNet
from PIL import Image, ImageDraw
from torchvision.utils import save_image
from unwarp import bilinear_unwarping

TRAIN_SAMPLES = 1_000_000
VAL_SAMPLES = 100_000


def build_model(config):
    """Instantiate the UVDoc network from the model and data sections of the config."""
    mc = config["model"]

    bm_out_size = tuple(config["data"].get("bm_out_size", [121, 177]))
    return UVDocNet(
        num_filter=mc.get("num_filter", 32),
        kernel_size=mc.get("kernel_size", 5),
        bm_out_size=bm_out_size,
        output_size=tuple(config["data"]["image_size"]) if config["data"]["image_size"] else None,
    )


class DocUnwarpModule(L.LightningModule):
    """Lightning module training the network on the backward map and shadow tasks jointly."""

    def __init__(self, config):
        """Build the model and the multi-objective loss described by the config."""
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)
        self.model = build_model(config)

        lc = config["training"]["loss"]
        self.criterion = MultiObjectiveLoss(
            bm_weight=lc.get("bm_weight", 1.0),
            shadow_weight=lc.get("shadow_weight", 0.5),
            recon_weight=lc.get("recon_weight", 0.0),
            loss_type=lc.get("loss_type", "l1"),
        )

        self._vis_batch = None

    def forward(self, x):
        """Run the network on a batch of input images."""
        return self.model(x)

    def _shared_step(self, batch):
        """Run the model on a batch and return the loss dict and the predictions."""
        predictions = self(batch["input"])
        loss_dict = self.criterion(
            predictions,
            {"bm_gt": batch["bm_gt"], "shadow_gt": batch["shadow_gt"], "input": batch["input"]},
        )
        return loss_dict, predictions

    def on_train_epoch_start(self):
        """Log the epoch and step, so a resume can be verified."""
        print(
            f"[Epoch {self.current_epoch} | global_step {self.global_step} | "
            f"rank {self.global_rank}] on_train_epoch_start. device: {self.device}"
        )
        self.log("resume/epoch", float(self.current_epoch), rank_zero_only=True)
        self.log("resume/global_step", float(self.global_step), rank_zero_only=True)

    def training_step(self, batch, batch_idx):
        """Run one training step and log every loss component."""
        loss_dict, _ = self._shared_step(batch)
        bs = batch["input"].size(0)
        self.log_dict(
            {
                "train/loss": loss_dict["total_loss"],
                "train/bm_loss": loss_dict["bm_loss"],
                "train/shadow_loss": loss_dict["shadow_loss"],
                "train/recon_loss": loss_dict["recon_loss"],
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=bs,
        )
        return loss_dict["total_loss"]

    def validation_step(self, batch, batch_idx):
        """Run one validation step, keeping the first batch for the end-of-epoch visualisations."""
        loss_dict, predictions = self._shared_step(batch)
        bs = batch["input"].size(0)
        self.log_dict(
            {
                "val/loss": loss_dict["total_loss"],
                "val/bm_loss": loss_dict["bm_loss"],
                "val/shadow_loss": loss_dict["shadow_loss"],
                "val/recon_loss": loss_dict["recon_loss"],
            },
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=bs,
        )
        if batch_idx == 0:
            self._vis_batch = (
                {
                    k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                },
                {
                    k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in predictions.items()
                },
            )

    def on_validation_epoch_end(self):
        """Save the predicted backward map and the reconstructions of the kept validation batch."""
        if self._vis_batch is None or self.trainer.sanity_checking or self.global_rank != 0:
            self._vis_batch = None
            return

        batch, predictions = self._vis_batch
        out_dir = (
            Path(self.config["paths"]["log_dir"])
            / self.config["logging"]["experiment_name"]
            / f"epoch_{self.current_epoch}"
        )

        out_size = (
            tuple(self.config["data"]["image_size"]) if self.config["data"]["image_size"] else None
        )
        for i in range(len(batch["input"])):
            sample_dir = out_dir / batch["sample_id"][i]
            sample_dir.mkdir(parents=True, exist_ok=True)
            np.save(sample_dir / "bm_pred.npy", predictions["bm_map"][i].numpy())
            self._reconstruct_and_save(
                batch["input"][i],
                predictions["bm_map"][i],
                predictions["shadow_map"][i],
                sample_dir,
                out_size,
            )
        self._vis_batch = None

    @staticmethod
    def _reconstruct_and_save(inp, bm, shadow, out_dir, out_size=None):
        """Unwarp and relight one sample, and save the input, shadow, grid and unwarped images."""
        H, W = (out_size[0], out_size[1]) if out_size else (inp.shape[-2], inp.shape[-1])

        inp_norm = inp.unsqueeze(0).float()  # normalised input, in [-1, 1]
        bm_b = bm.unsqueeze(0).float()

        inp_vis = inp_norm * 0.5 + 0.5  # same image, in [0, 1]
        inp_resized = F.interpolate(inp_vis, size=(H, W), mode="bilinear", align_corners=False)[0]

        shadow_01 = shadow * 0.5 + 0.5
        shadow_resized = F.interpolate(
            shadow_01.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
        )

        shadow_removed_input = (inp_vis / shadow_resized.clamp(min=0.1)).clamp(0, 1)
        shadow_removed_input_orig_size = F.interpolate(
            shadow_removed_input,
            size=(inp.shape[-2], inp.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        unwarped_shadow_removed = bilinear_unwarping(
            shadow_removed_input_orig_size, bm_b, img_size=(W, H)
        )[0]

        unwarped = bilinear_unwarping(inp_norm, bm_b, img_size=(W, H))[0]
        unwarped_01 = unwarped * 0.5 + 0.5

        BM_H, BM_W = bm.shape[-2], bm.shape[-1]
        bm_raw = bm.numpy()
        px = (bm_raw[0] + 1) / 2 * W
        py = (bm_raw[1] + 1) / 2 * H

        inp_np = (inp_resized.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        overlay = Image.fromarray(inp_np).convert("RGBA")
        draw = ImageDraw.Draw(overlay)
        color = (0, 255, 0, 180)

        for r in range(BM_H):
            pts = [(float(px[r, c]), float(py[r, c])) for c in range(BM_W)]
            draw.line(pts, fill=color, width=1)
        for c in range(BM_W):
            pts = [(float(px[r, c]), float(py[r, c])) for r in range(BM_H)]
            draw.line(pts, fill=color, width=1)

        bm_grid_tensor = (
            torch.from_numpy(np.array(overlay.convert("RGB"))).permute(2, 0, 1).float() / 255.0
        )

        save_image(inp_resized, out_dir / "input.png")
        save_image(shadow_resized[0], out_dir / "shadow.png")
        save_image(unwarped_01, out_dir / "unwarped.png")
        save_image(unwarped_shadow_removed, out_dir / "unwarped_shadow_removed.png")
        save_image(bm_grid_tensor, out_dir / "bm_grid.png")

    def on_fit_start(self):
        """Put the model in training mode before the fit starts."""
        self.model.train()

    def configure_optimizers(self):
        """Build the AdamW optimizer, with a tenth of the LR on the encoder, and its scheduler."""
        tc = self.config["training"]

        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=tc["learning_rate"],
            weight_decay=tc["weight_decay"],
        )

        sc = tc["scheduler"]
        if sc["type"] == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=tc["num_epochs"], eta_min=1e-6
            )
        elif sc["type"] == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        else:
            return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


class DocUnwarpDataModule(L.LightningDataModule):
    """Lightning data module serving the train and val loaders over the dataset zip shards."""

    def __init__(self, config):
        """Keep the config, the loaders themselves are built on demand by Lightning."""
        super().__init__()
        self.config = config

    def _loader(
        self,
        shard_pattern,
        shuffle_buffer,
        epoch_size=None,
        num_workers=None,
        augmentation=False,
        seed=None,
    ):
        """Build one dataloader, mapping the config onto create_webdataset_loader."""
        dc, tc = self.config["data"], self.config["training"]
        workers = num_workers if num_workers is not None else dc["num_workers"]
        bm_out_size = tuple(dc["bm_out_size"]) if dc.get("bm_out_size") else None
        return create_webdataset_loader(
            shard_pattern=shard_pattern,
            image_size=tuple(dc["image_size"]) if dc["image_size"] else None,
            normalize=dc["normalize"],
            crop_input=dc["crop_input"],
            augmentation=augmentation,
            batch_size=tc["batch_size"],
            num_workers=workers,
            shuffle_buffer=shuffle_buffer,
            seed=self.config["seed"] if seed is None else seed,
            epoch_size=epoch_size,
            prefetch_factor=1 if workers > 0 else None,
            bm_out_size=bm_out_size,
        )

    def train_dataloader(self):
        """Build the training loader, with shuffling and augmentations enabled."""
        return self._loader(
            self.config["data"]["train_shard_pattern"],
            self.config["data"].get("shuffle_buffer", 1000),
            epoch_size=TRAIN_SAMPLES // self.trainer.world_size,
            augmentation=self.config["data"]["augmentation"],
            # Offset by the epoch, otherwise the shard order and the order of the samples
            # within each shard would be identical at every epoch. This relies on the loader
            # being rebuilt every epoch (`reload_dataloaders_every_n_epochs=1`).
            seed=self.config["seed"] + self.trainer.current_epoch,
        )

    def val_dataloader(self):
        """Build the validation loader, without shuffling nor augmentations."""
        return self._loader(
            self.config["data"]["val_shard_pattern"],
            shuffle_buffer=0,
            epoch_size=VAL_SAMPLES // self.trainer.world_size,
            augmentation=False,
        )


def get_number_of_samples(shard_pattern, t="train"):
    """Estimate the number of samples from the two brace ranges of a shard pattern."""
    n_shards = len(list(braceexpand.braceexpand(shard_pattern)))
    total = n_shards * 1000
    print(f"Total {t} shards: {n_shards:,}  |  Estimated samples: {total:,}")
    return total


def set_seed(seed: int):
    """Seed python, numpy and torch, and make cudnn deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Make cudnn deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_latest_checkpoint(checkpoint_dir, experiment_name):
    """Return the run's last.ckpt, or its highest epoch checkpoint, or None."""
    ckpt_root = Path(checkpoint_dir) / experiment_name
    last_ckpt = ckpt_root / "last.ckpt"
    if last_ckpt.exists():
        print(f"[Resume] Found last.ckpt: {last_ckpt}")
        return str(last_ckpt)
    # Fall back to highest epoch folder
    epoch_dirs = sorted(ckpt_root.glob("epoch_*/"), key=lambda p: int(p.name.split("_")[1]))
    if epoch_dirs:
        candidates = list(epoch_dirs[-1].glob("*.ckpt"))
        if candidates:
            print(f"[Resume] Found checkpoint: {candidates[0]}")
            return str(candidates[0])
    print("[Resume] No checkpoint found, starting from scratch.")
    return None


def main():
    """Parse the arguments, assemble the model, data, logger and trainer, and run the fit."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/uvdoc_31x45.yaml")
    parser.add_argument(
        "--resume",
        default=None,
        help='Path to checkpoint. Use "auto" to find latest automatically.',
    )
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    global TRAIN_SAMPLES, VAL_SAMPLES
    TRAIN_SAMPLES = get_number_of_samples(config["data"]["train_shard_pattern"], "train")
    VAL_SAMPLES = get_number_of_samples(config["data"]["val_shard_pattern"], "val")

    set_seed(config["seed"])
    Path(config["paths"]["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["paths"]["log_dir"]).mkdir(parents=True, exist_ok=True)

    # Get the checkpoint to resume from, if any.
    resume_path = None
    if args.resume == "auto":
        resume_path = find_latest_checkpoint(
            config["paths"]["checkpoint_dir"], config["logging"]["experiment_name"]
        )
    elif args.resume:
        resume_path = args.resume
        print(f"[Resume] Using provided checkpoint: {resume_path}")

    if resume_path:
        # Quick sanity check: read epoch/step from checkpoint without loading weights
        try:
            ckpt_meta = torch.load(resume_path, map_location="cpu", weights_only=False)
            saved_epoch = ckpt_meta.get("epoch", "?")
            saved_step = ckpt_meta.get("global_step", "?")
            print(f"[Resume] Checkpoint epoch={saved_epoch}, global_step={saved_step}")
        except Exception as e:
            print(f"[Resume] Warning: could not read checkpoint metadata: {e}")
    else:
        print("[Resume] Starting fresh run.")

    model = DocUnwarpModule(config)
    datamodule = DocUnwarpDataModule(config)

    checkpoint_cb = ModelCheckpoint(
        dirpath=Path(config["paths"]["checkpoint_dir"]) / config["logging"]["experiment_name"],
        filename="epoch_{epoch}/{epoch}",
        auto_insert_metric_name=False,
        monitor="val/loss",
        mode="min",
        save_top_k=-1,
        save_last=True,
        every_n_epochs=config["training"]["save_every"],
    )

    lc = config["logging"]
    logger = CSVLogger(save_dir=config["paths"]["log_dir"], name=lc["experiment_name"], version="")

    bs, ws = config["training"]["batch_size"], args.gpus
    train_batches = TRAIN_SAMPLES // (bs * ws)
    val_batches = VAL_SAMPLES // (bs * ws)

    tc = config["training"]
    trainer = L.Trainer(
        max_epochs=tc["num_epochs"],
        accelerator="gpu",
        devices=args.gpus,
        strategy=DDPStrategy(
            cluster_environment=TorchElasticEnvironment(), find_unused_parameters=True
        )
        if args.gpus > 1
        else "auto",
        precision=tc.get("precision", "32"),
        gradient_clip_val=tc.get("gradient_clip_val", None),
        check_val_every_n_epoch=config["validation"]["val_every"],
        callbacks=[checkpoint_cb, TQDMProgressBar(refresh_rate=10), LearningRateMonitor()],
        logger=logger,
        log_every_n_steps=lc.get("log_every", 50),
        limit_train_batches=train_batches,
        limit_val_batches=val_batches,
        reload_dataloaders_every_n_epochs=1,
    )
    trainer.fit(model, datamodule=datamodule, ckpt_path=resume_path)


if __name__ == "__main__":
    main()
