"""LDGpt_main.py

# Author:
Richard Bruce Baxter - Copyright (c) 2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:

Python Environment Installation:
conda create -n pytorchsenv
source activate pytorchsenv
conda install python
pip install torch
pip install torchvision
pip install transformers

LDraw Library Installation (ldraw parts):
download latest ldraw library (complete.zip) - https://library.ldraw.org/library/updates/complete.zip
unzip complete.zip (ldraw folder)

LDView Installation (ldraw renderer):
sudo apt-get install libtinyxml2.6.2v5
download latest ldview (eg ldview-qt5-4.6-ubuntu-24.04.amd64.deb) - https://tcobbs.github.io/ldview/Downloads.html?utm_source=chatgpt.com
sudo dpkg -i ldview-qt5-4.6-ubuntu-24.04.amd64.deb 
LDView
when prompted select ldraw folder
which LDView (identify path to LDView executable)
update ldview_path in excecution command accordingly (eg --ldview_path /usr/bin/LDView)

# Usage (example):
source activate pytorchsenv
python LDGpt_main.py --encoder_model google/vit-base-patch16-224 --decoder_model tinyllama/TinyLlama-1.1B-Chat-v1.0 --output_dir ./ldgpy-checkpoints --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --max_target_length 256 --gradient_checkpointing --optim adafactor --fp16 --ldview_path /usr/bin/LDView --debug_render_path rendertmp/latest.png --debug_ldraw_path rendertmp/latest.ldr

# Description:
LDGpt - LDraw Generator for PyTorch

LDGpt synthetic Vision-Language model training entrypoint.

This script procedurally generates simple LDraw scenes, renders them to images
(using LDView when available), and trains a Hugging Face
VisionEncoderDecoderModel to recover the LDraw source from the rendered image.

The pipeline is intentionally modular so you can swap out the renderer,
augmentations, or the underlying encoder/decoder models as the project grows.

"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import logging
import os
import random
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence, Union

import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    VisionEncoderDecoderModel,
)

LOGGER = logging.getLogger("ldgpy")


# ---------------------------------------------------------------------------
# LDraw domain objects
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class LDrawBrick:
    """Simple representation of a single LDraw brick."""

    part_id: str
    color_id: int
    position: tuple[float, float, float]
    rotation_matrix: tuple[float, ...]

    def to_ldraw_line(self) -> str:
        px, py, pz = self.position
        ax, bx, cx, dx, ex, fx, gx, hx, ix = self.rotation_matrix
        return f"1 {self.color_id} {px:.2f} {py:.2f} {pz:.2f} {ax:.3f} {bx:.3f} {cx:.3f} {dx:.3f} {ex:.3f} {fx:.3f} {gx:.3f} {hx:.3f} {ix:.3f} {self.part_id}"


class LDrawSceneGenerator:
    """Produces random LDraw scenes comprised of simple bricks."""

    DEFAULT_PARTS = (
        "3001.dat",  # 2 x 4 brick
        "3002.dat",  # 2 x 3 brick
        "3003.dat",  # 2 x 2 brick
        "3004.dat",  # 1 x 2 brick
        "3005.dat",  # 1 x 1 brick
        "3024.dat",  # 1 x 1 plate
        "3039.dat",  # roof tile 2 x 2 x 45
        "3666.dat",  # slope 45 2 x 2 inverted
    )

    DEFAULT_COLORS = (4, 5, 14, 15, 21, 23, 24, 26, 27, 28, 34, 36, 38, 40, 41)

    ORIENTATION_MATRICES = (
        (1, 0, 0, 0, 1, 0, 0, 0, 1),
        (0, 1, 0, -1, 0, 0, 0, 0, 1),
        (-1, 0, 0, 0, -1, 0, 0, 0, 1),
        (0, -1, 0, 1, 0, 0, 0, 0, 1),
    )

    def __init__(
        self,
        parts: Optional[Sequence[str]] = None,
        color_ids: Optional[Sequence[int]] = None,
        min_bricks: int = 4,
        max_bricks: int = 18,
        grid_extent: float = 80.0,
        grid_step: float = 10.0,
    ) -> None:
        if min_bricks <= 0 or max_bricks < min_bricks:
            raise ValueError("Invalid brick count configuration")
        self.parts = tuple(parts or self.DEFAULT_PARTS)
        self.color_ids = tuple(color_ids or self.DEFAULT_COLORS)
        self.min_bricks = min_bricks
        self.max_bricks = max_bricks
        self.grid_extent = grid_extent
        self.grid_step = grid_step

    def _random_position(self, rng: random.Random) -> tuple[float, float, float]:
        choices = [x for x in self._frange(-self.grid_extent, self.grid_extent, self.grid_step)]
        return (
            rng.choice(choices),
            rng.choice(choices),
            rng.choice(choices),
        )

    @staticmethod
    def _frange(start: float, stop: float, step: float) -> Iterable[float]:
        current = start
        while current <= stop + 1e-6:
            yield round(current, 3)
            current += step

    def generate(self, rng: Optional[random.Random] = None) -> str:
        rng = rng or random
        brick_count = rng.randint(self.min_bricks, self.max_bricks)
        bricks = []
        for _ in range(brick_count):
            part_id = rng.choice(self.parts)
            color_id = rng.choice(self.color_ids)
            position = self._random_position(rng)
            rotation = rng.choice(self.ORIENTATION_MATRICES)
            bricks.append(LDrawBrick(part_id, color_id, position, rotation))

        header = (
            "0 \n"
            "0 Name: LDGptGenerated.ldr\n"
            "0 Author: LDGpt Synthetic Generator\n"
            "0 !LDRAW_ORG Unofficial_Model\n"
            "0 ROTSTEP 1 0 0 0 Absolute\n"
        )
        body = "\n".join(brick.to_ldraw_line() for brick in bricks)
        return f"{header}{body}\n"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

class LDViewRenderer:
    """Interface to LDView CLI for snapshot rendering.

    Falls back to synthetic noise images when LDView is not available, which is
    useful for smoke tests or headless environments.
    """

    def __init__(
        self,
        executable_path: Optional[str] = None,
        image_size: int = 256,
        background: str = "255,255,255",
        fallback_noise: bool = True,
        debug_render_path: Optional[Union[str, Path]] = None,
        debug_ldraw_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.executable_path = executable_path or "LDView"
        self.image_size = image_size
        self.background = background
        self.fallback_noise = fallback_noise
        self.debug_render_path = Path(debug_render_path) if debug_render_path else None
        self.debug_ldraw_path = Path(debug_ldraw_path) if debug_ldraw_path else None

    def render(self, ldraw_source: str, seed: Optional[int] = None) -> Image.Image:
        rng = random.Random(seed)
        with tempfile.TemporaryDirectory(prefix="ldgpy-render-") as tmpdir:
            tmp_path = Path(tmpdir)
            ldr_path = tmp_path / "scene.ldr"
            img_path = tmp_path / "scene.png"
            ldr_path.write_text(ldraw_source, encoding="utf-8")

            command = [
                self.executable_path,
                str(ldr_path),
                f"-SaveSnapShot={img_path}",
                f"-Width={self.image_size}",
                f"-Height={self.image_size}",
                f"-Background={self.background}",
            ]

            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                if completed.returncode != 0:
                    LOGGER.warning(
                        "LDView exited with code %s, falling back to noise image. stderr=%s",
                        completed.returncode,
                        completed.stderr.strip(),
                    )
                    raise RuntimeError("LDView render failed")

                image = Image.open(img_path).convert("RGB")
                self._write_debug_outputs(image, ldraw_source)
                return image
            except (FileNotFoundError, RuntimeError):
                if not self.fallback_noise:
                    raise
                LOGGER.debug("Using synthetic noise render instead of LDView output")
                noise = self._noise_image(rng)
                self._write_debug_outputs(noise, ldraw_source)
                return noise

    def _noise_image(self, rng: random.Random) -> Image.Image:
        base = Image.new("RGB", (self.image_size, self.image_size), "white")
        pixels = base.load()
        for x in range(self.image_size):
            for y in range(self.image_size):
                if rng.random() < 0.15:
                    r = int(rng.uniform(30, 200))
                    g = int(rng.uniform(30, 200))
                    b = int(rng.uniform(30, 200))
                    pixels[x, y] = (r, g, b)
        return base

    def _write_debug_outputs(self, image: Image.Image, ldraw_source: str) -> None:
        if self.debug_render_path:
            self.debug_render_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(self.debug_render_path)
        if self.debug_ldraw_path:
            self.debug_ldraw_path.parent.mkdir(parents=True, exist_ok=True)
            self.debug_ldraw_path.write_text(ldraw_source, encoding="utf-8")


class ImageAugmentor:
    """Lightweight style augmentation inspired by LD renders."""

    def __init__(self, max_rotation: float = 8.0, crop_jitter: float = 0.1) -> None:
        self.max_rotation = max_rotation
        self.crop_jitter = crop_jitter

    def __call__(self, image: Image.Image, rng: Optional[random.Random] = None) -> Image.Image:
        rng = rng or random
        angle = rng.uniform(-self.max_rotation, self.max_rotation)
        image = image.rotate(angle, resample=Image.BICUBIC, expand=False)

        if self.crop_jitter > 0:
            width, height = image.size
            crop_margin_w = int(width * self.crop_jitter * rng.random())
            crop_margin_h = int(height * self.crop_jitter * rng.random())
            left = crop_margin_w
            top = crop_margin_h
            right = width - crop_margin_w
            bottom = height - crop_margin_h
            if right - left > 10 and bottom - top > 10:
                image = image.crop((left, top, right, bottom))
                image = image.resize((width, height), resample=Image.BICUBIC)

        # Mild jitter in color and sharpness to mimic render variance.
        image = ImageEnhance.Color(image).enhance(rng.uniform(0.8, 1.2))
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.9, 1.1))
        image = ImageEnhance.Sharpness(image).enhance(rng.uniform(0.8, 1.5))

        if rng.random() < 0.2:
            image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 1.0)))

        return image


# ---------------------------------------------------------------------------
# Dataset and collator
# ---------------------------------------------------------------------------

class SyntheticLDrawDataset(Dataset):
    """On-the-fly dataset of synthetic LDraw render pairs."""

    def __init__(
        self,
        size: int,
        generator: LDrawSceneGenerator,
        renderer: LDViewRenderer,
        tokenizer: AutoTokenizer,
        max_target_length: int,
        augmentor: Optional[Callable[[Image.Image, random.Random], Image.Image]] = None,
        seed: Optional[int] = None,
    ) -> None:
        if size <= 0:
            raise ValueError("Dataset size must be positive")
        self.size = size
        self.generator = generator
        self.renderer = renderer
        self.tokenizer = tokenizer
        self.max_target_length = max_target_length
        self.augmentor = augmentor
        self.seed = seed or 0

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict:
        rng = random.Random(self.seed + index)
        ldraw_text = self.generator.generate(rng)
        image = self.renderer.render(ldraw_text, seed=self.seed + index)
        if self.augmentor:
            image = self.augmentor(image, rng)

        tokenized = self.tokenizer(
            ldraw_text,
            max_length=self.max_target_length,
            truncation=True,
            return_attention_mask=True,
        )

        return {
            "image": image,
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }


class LDrawDataCollator:
    """Pads image tensors and decoder tokens for Trainer consumption."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        image_processor: AutoImageProcessor,
        max_target_length: int,
        label_pad_token_id: int = -100,
    ) -> None:
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_target_length = max_target_length
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, batch: Sequence[dict]) -> dict:
        images = [item["image"] for item in batch]
        tokens = {
            "input_ids": [item["input_ids"] for item in batch],
            "attention_mask": [item["attention_mask"] for item in batch],
        }

        processed_images = self.image_processor(images=images, return_tensors="pt")

        token_pad_kwargs = {
            "padding": "max_length",
            "max_length": self.max_target_length,
            "return_tensors": "pt",
        }
        signature_params = inspect.signature(self.tokenizer.pad).parameters
        if "truncation" in signature_params:
            token_pad_kwargs["truncation"] = True

        padded = self.tokenizer.pad(tokens, **token_pad_kwargs)

        labels = padded["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = self.label_pad_token_id

        batch_dict = {
            "pixel_values": processed_images["pixel_values"],
            "labels": labels,
            "decoder_attention_mask": padded["attention_mask"],
        }
        return batch_dict


# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------

LDRAW_KEYWORDS = [
    "0", "1", "2", "3", "4", "5", "ROTSTEP", "!LDRAW_ORG", "Unofficial_Model",
    "Name:", "Author:", "LDGpt", "Synthetic", "Generator",
]


def extend_tokenizer_for_ldraw(tokenizer: AutoTokenizer, parts: Sequence[str], colors: Sequence[int]) -> None:
    """Registers LDraw specific tokens with the decoder tokenizer."""

    new_tokens = set(LDRAW_KEYWORDS)
    new_tokens.update(parts)
    new_tokens.update(str(color) for color in colors)

    # Avoid duplicating tokens already known by the tokenizer.
    tokens_to_add = sorted(token for token in new_tokens if token not in tokenizer.get_vocab())
    if tokens_to_add:
        LOGGER.info("Adding %d LDraw tokens to tokenizer", len(tokens_to_add))
        tokenizer.add_tokens(tokens_to_add)

    if tokenizer.pad_token is None:
        LOGGER.warning("Tokenizer has no pad_token. Using eos_token as pad.")
        if tokenizer.eos_token is None:
            tokenizer.add_special_tokens({"pad_token": "<pad>", "eos_token": "</s>", "bos_token": "<s>"})
        else:
            tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.cls_token or tokenizer.pad_token
    if tokenizer.eos_token is None:
        tokenizer.eos_token = tokenizer.sep_token or tokenizer.pad_token


# ---------------------------------------------------------------------------
# Training harness
# ---------------------------------------------------------------------------


def build_model(encoder_model: str, decoder_model: str, tokenizer: AutoTokenizer) -> VisionEncoderDecoderModel:
    LOGGER.info("Loading VisionEncoderDecoderModel with encoder=%s decoder=%s", encoder_model, decoder_model)
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(encoder_model, decoder_model)
    model.decoder.resize_token_embeddings(len(tokenizer))

    if tokenizer.bos_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define BOS/EOS token ids")

    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.tie_word_embeddings = True
    return model


class LDGSeq2SeqTrainer(Seq2SeqTrainer):
    """Trainer that drops decoder_inputs_embeds when decoder_input_ids are present."""

    def _prepare_inputs(self, inputs):  # type: ignore[override]
        prepared = super()._prepare_inputs(inputs)
        if isinstance(prepared, dict):
            if "input_ids" in prepared and "inputs_embeds" in prepared:
                LOGGER.debug("Batch contains both input_ids and inputs_embeds; dropping embeds")
                prepared = dict(prepared)
                prepared.pop("inputs_embeds", None)
            if "decoder_inputs_embeds" in prepared and prepared.get("decoder_input_ids") is not None:
                LOGGER.debug("Removing decoder_inputs_embeds from batch before forward pass")
                prepared = dict(prepared)
                prepared.pop("decoder_inputs_embeds", None)
        return prepared

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):  # type: ignore[override]
        pixel_values = inputs.get("pixel_values")
        labels = inputs.get("labels")
        decoder_attention_mask = inputs.get("decoder_attention_mask")

        if pixel_values is None or labels is None:
            raise ValueError("Batch must contain pixel_values and labels for loss computation")

        outputs = model(
            pixel_values=pixel_values,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
            num_items_in_batch=num_items_in_batch,
            return_dict=False,
        )

        loss = outputs[0] if isinstance(outputs, tuple) else outputs.loss
        if return_outputs:
            return loss, outputs
        return loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a VLM to recover LDraw source from renders.")
    parser.add_argument("--encoder_model", required=True, help="Vision encoder model name (e.g. google/vit-base-patch16-224)")
    parser.add_argument("--decoder_model", required=True, help="Decoder LM model name (e.g. tinyllama/TinyLlama-1.1B-Chat-v1.0)")
    parser.add_argument("--output_dir", required=True, help="Directory to save checkpoints and tokenizer")
    parser.add_argument("--dataset_size", type=int, default=10_000, help="Number of synthetic samples to stream")
    parser.add_argument("--eval_size", type=int, default=512, help="Number of eval samples")
    parser.add_argument("--max_target_length", type=int, default=512, help="Max tokenizer target length")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--bf16", action="store_true", help="Use bf16 mixed precision")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 mixed precision")
    parser.add_argument("--optim", default="adamw_torch", help="Optimizer to use (e.g. adamw_torch, adafactor)")
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce activation memory",
    )
    parser.add_argument(
        "--debug_render_path",
        type=Path,
        default=None,
        help="If set, saves the latest rendered image to this path for inspection",
    )
    parser.add_argument(
        "--debug_ldraw_path",
        type=Path,
        default=None,
        help="If set, saves the latest generated LDraw text to this path",
    )
    parser.add_argument("--ldview_path", default=None, help="Path to LDView executable")
    parser.add_argument("--image_size", type=int, default=256, help="Rendered image size")
    parser.add_argument("--no_augment", action="store_true", help="Disable image augmentations")
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", default=None)
    return parser.parse_args()


def prepare_trainer(args: argparse.Namespace) -> Seq2SeqTrainer:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    generator = LDrawSceneGenerator()
    renderer = LDViewRenderer(
        executable_path=args.ldview_path,
        image_size=args.image_size,
        debug_render_path=args.debug_render_path,
        debug_ldraw_path=args.debug_ldraw_path,
    )
    augmentor = None if args.no_augment else ImageAugmentor()

    image_processor = AutoImageProcessor.from_pretrained(args.encoder_model)
    tokenizer = AutoTokenizer.from_pretrained(args.decoder_model, use_fast=True)

    extend_tokenizer_for_ldraw(tokenizer, generator.parts, generator.color_ids)

    model = build_model(args.encoder_model, args.decoder_model, tokenizer)

    if args.gradient_checkpointing:
        LOGGER.info("Enabling gradient checkpointing on the model")
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    train_dataset = SyntheticLDrawDataset(
        size=args.dataset_size,
        generator=generator,
        renderer=renderer,
        tokenizer=tokenizer,
        max_target_length=args.max_target_length,
        augmentor=augmentor,
        seed=args.seed,
    )
    eval_dataset = SyntheticLDrawDataset(
        size=args.eval_size,
        generator=generator,
        renderer=renderer,
        tokenizer=tokenizer,
        max_target_length=args.max_target_length,
        augmentor=augmentor,
        seed=args.seed + 10_000,
    )

    data_collator = LDrawDataCollator(
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_target_length=args.max_target_length,
    )

    training_kwargs = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "fp16": args.fp16,
        "bf16": args.bf16,
        "label_smoothing_factor": args.label_smoothing,
        "predict_with_generate": True,
        "generation_max_length": args.max_target_length,
        "dataloader_num_workers": args.dataloader_num_workers,
        "max_steps": args.max_train_steps,
        "seed": args.seed,
        "push_to_hub": args.push_to_hub,
        "hub_model_id": args.hub_model_id,
        "remove_unused_columns": False,
        "optim": args.optim,
    }

    if training_kwargs.get("max_steps") is None:
        training_kwargs["max_steps"] = -1

    # Align keyword arguments with the installed transformers version.
    signature_params = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    if "evaluation_strategy" in signature_params:
        training_kwargs["evaluation_strategy"] = "steps"
    elif "eval_strategy" in signature_params:
        training_kwargs["eval_strategy"] = "steps"
    else:
        LOGGER.warning("Transformers version does not support evaluation strategy; eval is disabled.")
        training_kwargs.pop("eval_steps", None)

    if "gradient_checkpointing" in signature_params:
        training_kwargs["gradient_checkpointing"] = args.gradient_checkpointing

    # Drop unsupported kwargs gracefully so older transformer releases still work.
    unsupported_keys = [key for key in training_kwargs if key not in signature_params]
    for key in unsupported_keys:
        LOGGER.debug("Dropping unsupported training arg '%s' for installed transformers", key)
        training_kwargs.pop(key)

    training_args = Seq2SeqTrainingArguments(**training_kwargs)

    trainer = LDGSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    return trainer


def main() -> None:
    logging.basicConfig(level=os.environ.get("LDGPY_LOGLEVEL", "INFO"))
    args = parse_args()
    trainer = prepare_trainer(args)
    trainer.train()
    trainer.save_model()
    trainer.tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
