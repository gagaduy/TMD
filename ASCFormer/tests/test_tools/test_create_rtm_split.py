from pathlib import Path


def test_create_stratified_split_is_deterministic_and_preserves_each_group(
        tmp_path):
    from tools.analysis_tools.create_rtm_split import create_split

    source = tmp_path / 'train.txt'
    source.write_text(
        '\n'.join([
            'good_0001', 'good_0002', 'good_0003', 'good_0004',
            'cpmv_0001', 'cpmv_0002', 'cpmv_0003', 'cpmv_0004',
            'cover_0001', 'cover_0002', 'cover_0003', 'cover_0004',
        ]) + '\n')

    first = create_split(source, tmp_path / 'first', val_ratio=0.25, seed=7)
    second = create_split(source, tmp_path / 'second', val_ratio=0.25, seed=7)

    assert first == second
    assert first['train_count'] == 9
    assert first['val_count'] == 3
    assert first['group_counts']['good'] == {'train': 3, 'val': 1}
    assert first['group_counts']['cpmv'] == {'train': 3, 'val': 1}
    assert first['group_counts']['cover'] == {'train': 3, 'val': 1}
    assert (tmp_path / 'first' / 'train.txt').read_text() == (
        tmp_path / 'second' / 'train.txt').read_text()
    assert (tmp_path / 'first' / 'val.txt').read_text() == (
        tmp_path / 'second' / 'val.txt').read_text()
