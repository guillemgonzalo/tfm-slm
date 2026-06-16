import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.dataset.downloader import DatasetConfig, DatasetDownloader


def test_dataset_config_model():
    """Test the Pydantic model for dataset configuration validation."""
    config = DatasetConfig(path="tatsu-lab/alpaca")
    assert config.path == "tatsu-lab/alpaca"
    assert config.name is None
    assert config.data_dir is None

    config_with_opts = DatasetConfig(
        path="bigcode/the-stack", name="yaml", data_dir="data/yaml"
    )
    assert config_with_opts.path == "bigcode/the-stack"
    assert config_with_opts.name == "yaml"
    assert config_with_opts.data_dir == "data/yaml"


def test_ensure_directory_exists(tmp_path: Path):
    """Test that the directory is created if it does not exist."""
    target_dir = tmp_path / "new_datasets_dir"
    DatasetDownloader(output_dir=str(target_dir))

    assert target_dir.exists()
    assert target_dir.is_dir()


@patch("app.dataset.downloader.load_dataset")
def test_download_single_dataset_success(mock_load_dataset: MagicMock, tmp_path: Path):
    """Test the successful downloading of a single dataset using mocks."""
    downloader = DatasetDownloader(output_dir=str(tmp_path))
    config = DatasetConfig(path="OpenAssistant/oasst1", name=None, data_dir=None)

    downloader.download_single("open_assistant", config)

    # Assert that the Hugging Face load_dataset was called with correct parameters
    mock_load_dataset.assert_called_once_with(
        "OpenAssistant/oasst1",
        name=None,
        cache_dir=str(tmp_path),
        data_dir=None,
        trust_remote_code=True,
    )


@patch("app.dataset.downloader.load_dataset")
def test_download_single_dataset_failure(
    mock_load_dataset: MagicMock, tmp_path: Path, caplog
):
    """Test that a failure in load_dataset is caught and logged."""
    mock_load_dataset.side_effect = Exception("Hugging Face API Error")
    downloader = DatasetDownloader(output_dir=str(tmp_path))
    config = DatasetConfig(path="fake/dataset")

    with caplog.at_level(logging.ERROR):
        downloader.download_single("fake_key", config)

    assert "Failed to download fake_key: Hugging Face API Error" in caplog.text


@patch("app.dataset.downloader.DatasetDownloader.download_single")
def test_download_all(mock_download_single: MagicMock, tmp_path: Path):
    """Test the download_all method to ensure it iterates over all configs."""
    from app.dataset.constants import DATASETS_CONFIG

    downloader = DatasetDownloader(output_dir=str(tmp_path))
    downloader.download_all()

    expected_calls = len(DATASETS_CONFIG)
    assert mock_download_single.call_count == expected_calls
