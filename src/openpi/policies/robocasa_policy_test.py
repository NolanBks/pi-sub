import numpy as np

from openpi.models import model as _model
from openpi.policies import robocasa_policy


def test_robocasa_inputs_preserve_hierarchical_targets():
    transform = robocasa_policy.RoboCasaInputs(model_type=_model.ModelType.PI05)
    data = transform(
        {
            "observation/state": np.zeros(16, dtype=np.float32),
            "observation/agentview_left": np.zeros((3, 256, 256), dtype=np.float32),
            "observation/eye_in_hand": np.zeros((256, 256, 3), dtype=np.uint8),
            "observation/agentview_right": np.zeros((256, 256, 3), dtype=np.uint8),
            "actions": np.zeros((50, 12), dtype=np.float32),
            "prompt": "close the fridge",
            "subtask": "reach for the handle",
            "action_loss_mask": np.ones(50, dtype=np.bool_),
        }
    )

    assert data["image"]["base_0_rgb"].shape == (256, 256, 3)
    assert data["image"]["left_wrist_0_rgb"].shape == (256, 256, 3)
    assert data["image"]["right_wrist_0_rgb"].shape == (256, 256, 3)
    assert data["state"].shape == (16,)
    assert data["actions"].shape == (50, 12)
    assert data["subtask"] == "reach for the handle"
    assert np.all(data["action_loss_mask"])


def test_robocasa_outputs_keep_generated_subtask():
    transform = robocasa_policy.RoboCasaOutputs()
    data = transform(
        {
            "actions": np.zeros((50, 32), dtype=np.float32),
            "subtask": "pull the door",
            "subtask_tokens": np.array([1, 2, 3]),
            "subtask_token_mask": np.array([True, True, True]),
        }
    )

    assert data["actions"].shape == (50, 12)
    assert data["subtask"] == "pull the door"
    assert np.array_equal(data["subtask_tokens"], np.array([1, 2, 3]))
