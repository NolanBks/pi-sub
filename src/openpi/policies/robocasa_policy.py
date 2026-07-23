import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_robocasa_example() -> dict:
    """Create a random RoboCasa365 policy input."""
    return {
        "observation/state": np.random.rand(16).astype(np.float32),
        "observation/agentview_left": np.random.randint(256, size=(256, 256, 3), dtype=np.uint8),
        "observation/eye_in_hand": np.random.randint(256, size=(256, 256, 3), dtype=np.uint8),
        "observation/agentview_right": np.random.randint(256, size=(256, 256, 3), dtype=np.uint8),
        "prompt": "close the fridge",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    if image.ndim != 3:
        raise ValueError(f"Expected a single image with 3 dimensions, got {image.shape}.")
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    if image.shape[-1] != 3:
        raise ValueError(f"Expected an RGB image, got {image.shape}.")
    return image


@dataclasses.dataclass(frozen=True)
class RoboCasaInputs(transforms.DataTransformFn):
    """Map RoboCasa365 observations to the three image slots used by openpi."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/agentview_left"])
        wrist_image = _parse_image(data["observation/eye_in_hand"])
        second_base_image = _parse_image(data["observation/agentview_right"])

        inputs = {
            "state": np.asarray(data["observation/state"]),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # pi0 currently has three fixed visual slots. RoboCasa's second
                # agent view is retained in the remaining slot instead of dropped.
                "right_wrist_0_rgb": second_base_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        for key in ("actions", "prompt", "subtask", "action_loss_mask"):
            if key in data:
                inputs[key] = data[key]
        return inputs


@dataclasses.dataclass(frozen=True)
class RoboCasaOutputs(transforms.DataTransformFn):
    """Trim padded actions while retaining staged subtask inference outputs."""

    action_dim: int = 12

    def __call__(self, data: dict) -> dict:
        outputs = {"actions": np.asarray(data["actions"])[..., : self.action_dim]}
        for key in ("subtask", "subtask_tokens", "subtask_token_mask"):
            if key in data:
                outputs[key] = data[key]
        return outputs
