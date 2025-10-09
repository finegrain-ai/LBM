import logging
import math
from typing import Any, Dict, List, Tuple
from neptune.types import File as NeptuneFile
import numpy as np
import torch
import wandb
from PIL import Image, ImageDraw, ImageFont
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only
from torchvision.utils import make_grid
from lbm.data.datasets.collation_fn import custom_collation_fn
from ..trainer import TrainingPipeline

logging.basicConfig(level=logging.INFO)


def create_grid_texts(
    texts: List[str],
    n_cols: int = 4,
    image_size: Tuple[int] = (512, 512),
    font_size: int = 40,
    margin: int = 5,
    offset: int = 5,
) -> Image.Image:
    """
    Create a grid of white images containing the given texts.

    Args:
        texts (List[str]): List of strings to be drawn on images.
        n_cols (int): Number of columns in the grid.
        image_size (tuple): Size of the generated images (width, height).
        font_size (int): Font size of the text.
        margin (int): Margin around the text.
        offset (int): Offset between lines.

    Returns:
        PIL.Image: List of generated images as a grid
    """

    images = []
    font = ImageFont.load_default(size=font_size)

    for text in texts:
        img = Image.new("RGB", image_size, color="white")
        draw = ImageDraw.Draw(img)
        margin_ = margin
        offset_ = offset
        for line in wrap_text(
            text=text, draw=draw, max_width=image_size[0] - 2 * margin_, font=font
        ):
            draw.text((margin_, offset_), line, font=font, fill="black")
            offset_ += font_size
        images.append(img)

    # create a pil grid
    n_rows = math.ceil(len(images) / n_cols)
    grid = Image.new(
        "RGB", (n_cols * image_size[0], n_rows * image_size[1]), color="white"
    )
    for i, img in enumerate(images):
        grid.paste(img, (i % n_cols * image_size[0], i // n_cols * image_size[1]))

    return grid


def wrap_text(
    text: str, draw: ImageDraw.Draw, max_width: int, font: ImageFont
) -> List[str]:
    """
    Wrap text to fit within a specified width when drawn.
    It will return to the new line when the text is larger than the max_width.

    Args:
        text (str): The text to be wrapped.
        draw (ImageDraw.Draw): The draw object to calculate text size.
        max_width (int): The maximum width for the wrapped text.
        font (ImageFont): The font used for the text.

    Returns:
        List[str]: List of wrapped lines.
    """
    lines = []
    current_line = ""
    for letter in text:
        if draw.textbbox((0, 0), current_line + letter, font=font)[2] <= max_width:
            current_line += letter
        else:
            lines.append(current_line)
            current_line = letter
    lines.append(current_line)
    return lines

class NeptuneLogger(Callback):
    """
    Logger for logging samples to Neptune. This logger is used to log images, text, and metrics to Neptune.

    Args:
        log_batch_freq (int): The frequency of logging samples to Neptune. Default is 100.
    """

    def __init__(self, log_batch_freq: int = 100):
        super().__init__()
        self.log_batch_freq = log_batch_freq
        self.val_metrics: list[Dict[str, Any]] = []
    
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: TrainingPipeline,
        outputs: Dict[str, Any],
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._process_logs(trainer, outputs, split="train")
    
    def on_validation_start(self, trainer: Trainer, pl_module: TrainingPipeline) -> None:
        self.val_metrics = []

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: TrainingPipeline,
        outputs: Dict[str, Any],
        batch: Any,
        batch_idx: int,
    ) -> None:
        assert outputs["visuals"] is not None
        assert outputs["metrics"] is not None

        metrics = outputs.pop("metrics")
        visuals = outputs.pop("visuals")
        self._process_logs(trainer, visuals, split="val", rescale=True)

        # the metrics are aggregated at each batch
        # and stored in self.val_metrics
        # they are logged at the end of the validation epoch
        first_key = pl_module.log_keys[0]
        self.val_metrics.append({
            # batch_size is not always the same 
            # especially with wds.batched(..., partial=True) (which is the default)
            "batch_size": batch[first_key].shape[0],
            **metrics,
        })
    
    def on_validation_end(
        self,
        trainer: Trainer,
        pl_module: TrainingPipeline
    ) -> None:
        
        collated_metrics = custom_collation_fn(self.val_metrics)

        n_items = sum([self.val_metrics[i]["batch_size"] for i in range(len(self.val_metrics))])
        if n_items == 0:
            logging.warning("No items in validation metrics. Skipping metric aggregation.")
            return
        
        # reduce metrics with a mean weighted by batch size
        # TODO: handle/test multi-gpu aggregation
        aggregated_metrics = {
            key: np.sum(
                collated_metrics[key][index] * self.val_metrics[index]["batch_size"]
                for index in range(len(self.val_metrics))
             ) / n_items
            for key in collated_metrics.keys() if key != "batch_size" 
        }
           
        self._process_logs(
            trainer,
            aggregated_metrics,
            split="val"
        )
        self.val_metrics = []

    @rank_zero_only
    def _process_logs(
        self,
        trainer,
        logs: Dict[str, Any],
        rescale : bool =True,
        split: str ="train",
    ) -> Dict[str, Any]:
        for key, value in logs.items():
            full_key = f"{split}/{key}"
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu()
                if value.dim() == 4:
                    images = value
                    if rescale:
                        images = (images + 1.0) / 2.0
                    grid = make_grid(images, nrow=4)
                    grid = grid.permute(1, 2, 0)
                    grid = grid.mul(255).clamp(0, 255).to(torch.uint8)
                    logs[key] = grid.numpy()
                    trainer.logger.experiment[full_key].append(
                        NeptuneFile.as_image(Image.fromarray(logs[key])),
                        step=trainer.global_step,
                    )

                # Scalar tensor
                if value.dim() == 1 or value.dim() == 0:
                    value = value.float().numpy()
                    trainer.logger.experiment[full_key].append(
                        value, step=trainer.global_step
                    )

            # list of string (e.g. text)
            if isinstance(value, list):
                if isinstance(value[0], str):
                    pil_image_texts = create_grid_texts(value)
                    neptune_image = NeptuneFile.as_image(Image.fromarray(pil_image_texts))
                    trainer.logger.experiment[full_key].append(
                        neptune_image, step=trainer.global_step
                    )

            # dict of tensors (e.g. metrics)
            if isinstance(value, dict) and value:
                for k, v in value.items():
                    if isinstance(v, torch.Tensor):
                        value[k] = v.detach().cpu().numpy()

                trainer.logger.experiment[full_key].append(
                    value, step=trainer.global_step
                )

            if isinstance(value, int) or isinstance(value, float):
                trainer.logger.experiment[full_key].append(
                    value, step=trainer.global_step
                )

        return logs
