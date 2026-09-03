from pathlib import Path

from mmengine import Config


CONFIG_DIR = Path(__file__).resolve().parents[2] / 'configs' / 'ascformer'


def test_1gpu_experiment_configs_share_the_split_and_crop_size():
    baseline = Config.fromfile(CONFIG_DIR / 'ascformer_rtm_1gpu.py')
    dwt = Config.fromfile(CONFIG_DIR / 'ascformer_rtm_dwt_1gpu.py')

    for config in (baseline, dwt):
        assert config.train_dataloader.batch_size == 1
        assert config.train_dataloader.dataset.ann_file == (
            'experiment_splits/seed_20260903/train.txt')
        assert config.val_dataloader.dataset.ann_file == (
            'experiment_splits/seed_20260903/val.txt')
        assert config.test_dataloader.dataset.ann_file == 'test.txt'
        assert config.model.data_preprocessor.size == (384, 384)
        assert config.train_dataloader.dataset.pipeline[-2].crop_size == (
            384, 384)
