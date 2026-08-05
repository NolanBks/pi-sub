import pathlib

from openpi import transforms
from openpi.models import pi0_config
from openpi.training import config


def test_robocasa_v21_action_only_config_does_not_require_subtask_annotations():
    train_config = config.get_config("pi05_robocasa365_v21_action_only")

    assert isinstance(train_config.model, pi0_config.Pi0Config)
    assert train_config.model.pi05
    assert not train_config.model.train_subtask_prediction
    assert not train_config.model.sample_subtask_prediction

    data_config = train_config.data.create(pathlib.Path("/tmp/openpi-config-test"), train_config.model)
    assert data_config.action_sequence_keys == ("action",)
    assert data_config.prompt_from_task
    assert len(data_config.repack_transforms.inputs) == 1
    assert isinstance(data_config.repack_transforms.inputs[0], transforms.RepackTransform)
    assert all(
        not isinstance(transform, transforms.ExtractActiveSubtask) for transform in data_config.repack_transforms.inputs
    )
