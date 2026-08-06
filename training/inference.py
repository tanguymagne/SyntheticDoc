import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from model import UVDocNet
from PIL import Image
from torchvision import transforms
from unwarp import bilinear_unwarping


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Rebuild the model stored in a Lightning checkpoint, load its weights and return it.

    The input resolution the checkpoint was trained at is returned alongside the model, so
    inference can run at the same resolution without having to be told about it.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("hyper_parameters", {})
    mc = config.get("model", {})
    dc = config.get("data", {})
    bm_out_size = tuple(dc.get("bm_out_size", [121, 177]))
    output_size = tuple(dc.get("image_size", [720, 512])) if dc.get("image_size") else None

    state_dict = checkpoint.get("state_dict", checkpoint)
    if any(k.startswith("model.") for k in state_dict.keys()):
        state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}

    model = UVDocNet(
        num_filter=mc.get("num_filter", 32),
        kernel_size=mc.get("kernel_size", 5),
        bm_out_size=bm_out_size,
        output_size=output_size,
    )
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    print(f"Model loaded from {checkpoint_path}")
    return model, output_size


def preprocess_image(image_path: str, image_size=(720, 512)):
    """Load an image and resize / normalise it to the model input format."""

    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    image_tensor = transform(image).unsqueeze(0)

    return image_tensor, image


@torch.no_grad()
def predict(model, image_tensor, device="cuda"):
    """Predict the backward map and the shadow map for a single image."""
    image_tensor = image_tensor.to(device)
    predictions = model(image_tensor)

    bm_map = predictions["bm_map"][0].cpu()
    shadow_map = predictions["shadow_map"][0].cpu()

    return bm_map, shadow_map


def run_inference(
    model,
    input_image_path: str,
    output_path: str,
    device: str = "cuda",
    image_size: tuple = (720, 512),
    shadow_removal: bool = True,
):
    """Unwarp one image at its original resolution, optionally remove shadows, and save it."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load original image at full resolution
    original_image = Image.open(input_image_path).convert("RGB")
    orig_W, orig_H = original_image.size  # PIL gives (W, H)

    # Preprocess at model resolution for inference
    image_tensor, _ = preprocess_image(input_image_path, image_size)

    # Predict bm_map and shadow_map at model resolution
    bm_map, shadow_map = predict(model, image_tensor, device)

    # Scale bm_map to original resolution (values stay in [-1,1], just upsample spatially)
    bm_b = F.interpolate(
        bm_map.unsqueeze(0).float(), size=(orig_H, orig_W), mode="bilinear", align_corners=True
    )

    # Prepare original resolution image tensor
    to_tensor = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    )
    orig_tensor = to_tensor(original_image).unsqueeze(0).float()  # [1, 3, orig_H, orig_W]

    # Unwarp at full resolution
    unwarped = bilinear_unwarping(orig_tensor, bm_b, img_size=(orig_W, orig_H))[0]
    unwarped_01 = unwarped * 0.5 + 0.5

    if shadow_removal:
        shadow_01 = shadow_map * 0.5 + 0.5
        shadow_resized = F.interpolate(
            shadow_01.unsqueeze(0), size=(orig_H, orig_W), mode="bilinear", align_corners=False
        )
        inp_vis = orig_tensor * 0.5 + 0.5

        # Adapt shadow correction strength based on input brightness
        # Bright images get less correction to avoid overexposure
        input_brightness = inp_vis.mean().item()
        brightness_scale = min(1.0, 0.4 / max(input_brightness, 1e-6))
        shadow_blended = shadow_resized * brightness_scale + (1 - brightness_scale)

        shadow_removed = (inp_vis / shadow_blended.clamp(min=0.1)).clamp(0, 1)
        unwarped_shadow_removed = bilinear_unwarping(
            shadow_removed * 2 - 1, bm_b, img_size=(orig_W, orig_H)
        )[0]
        result_tensor = unwarped_shadow_removed * 0.5 + 0.5
    else:
        result_tensor = unwarped_01

    result = (result_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(result).save(output_path)

    print(f"Saved: {output_path} (resolution: {orig_W}x{orig_H})")


def main():
    """Run the model over every image of a directory and write the unwarped results."""
    parser = argparse.ArgumentParser(description="Unwarp a folder of document images")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to Lightning .ckpt file"
    )
    parser.add_argument("--input-dir", type=str, required=True, help="Directory of input images")
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory to save unwarped predictions"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--no-shadow-removal", action="store_true", help="Disable shadow removal in output"
    )

    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    if device != args.device:
        print(f"Warning: {args.device} not available, using {device}")

    model, ckpt_image_size = load_model(args.checkpoint, device)

    if ckpt_image_size is not None:
        image_size = ckpt_image_size
    else:
        image_size = (720, 512)
        print("Warning: the checkpoint stores no input resolution, falling back to the default")
    print(f"Input resolution: {image_size[0]}x{image_size[1]}")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Grab all images in the input directory (PNG or JPG)
    image_paths = sorted(list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg")))
    print(f"Found {len(image_paths)} images in {input_dir}")

    for img_path in image_paths:
        out_path = output_dir / img_path.name
        run_inference(
            model,
            str(img_path),
            str(out_path),
            device=device,
            image_size=image_size,
            shadow_removal=not args.no_shadow_removal,
        )

    print(f"\nDone! Predictions saved to {output_dir}")


if __name__ == "__main__":
    main()
