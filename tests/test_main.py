import logging
from unittest.mock import MagicMock, patch

from app.main import main


@patch("app.main.TrainingService")
@patch("app.main.DatasetCacheService")
def test_main_skips_pipeline_when_dataset_cached_locally(
    mock_cache_cls: MagicMock,
    mock_trainer_cls: MagicMock,
    caplog,
):
    """
    Test that main() skips download/processing when the dataset is already
    cached locally, and always runs training.
    """
    # Given: dataset already cached locally
    mock_cache = mock_cache_cls.return_value
    mock_cache.exists_locally.return_value = True
    mock_trainer = mock_trainer_cls.return_value

    # When: pipeline runs
    with caplog.at_level(logging.INFO):
        main()

    # Then: download/processing/upload are skipped
    mock_cache.exists_in_s3.assert_not_called()
    mock_cache.download.assert_not_called()
    mock_cache.upload.assert_not_called()

    # Then: training still runs with the expected hyperparameters
    mock_trainer_cls.assert_called_once()
    mock_trainer.train.assert_called_once_with(epochs=15, batch_size=64, grad_accum_steps=6)

    # Then: expected log messages are emitted
    assert "Starting SLM Dataset Pipeline" in caplog.text
    assert "Processed dataset found locally. Skipping download/processing." in caplog.text


@patch("app.main.TrainingService")
@patch("app.main.TokenizerService")
@patch("app.main.DatasetProcessor")
@patch("app.main.DatasetDownloader")
@patch("app.main.DatasetCacheService")
def test_main_downloads_and_processes_when_no_cache(
    mock_cache_cls: MagicMock,
    mock_downloader_cls: MagicMock,
    mock_processor_cls: MagicMock,
    mock_tokenizer_cls: MagicMock,
    mock_trainer_cls: MagicMock,
    caplog,
):
    """
    Test that main() downloads, processes, tokenizes and uploads the dataset
    when no cache is available locally or in S3, then runs training.
    """
    # Given: no cached dataset, locally or in S3
    mock_cache = mock_cache_cls.return_value
    mock_cache.exists_locally.return_value = False
    mock_cache.exists_in_s3.return_value = False

    mock_downloader = mock_downloader_cls.return_value
    mock_processor = mock_processor_cls.return_value
    mock_tokenizer = mock_tokenizer_cls.return_value
    mock_trainer = mock_trainer_cls.return_value

    # When: pipeline runs
    with caplog.at_level(logging.INFO):
        main()

    # Then: raw data is downloaded
    mock_downloader_cls.assert_called_once()
    mock_downloader.download_all.assert_called_once()

    # Then: dataset is processed with expected sample sizes
    mock_processor_cls.assert_called_once()
    mock_processor.process.assert_called_once_with(
        total_samples=494_876, benchmark_samples=148_437
    )

    # Then: tokenizer is saved and processed dataset is uploaded for reuse
    mock_tokenizer_cls.assert_called_once()
    mock_tokenizer.save.assert_called_once()
    mock_cache.upload.assert_called_once()

    # Then: training runs with the expected hyperparameters
    mock_trainer_cls.assert_called_once()
    mock_trainer.train.assert_called_once_with(epochs=15, batch_size=64, grad_accum_steps=6)

    # Then: expected log messages are emitted
    assert "Starting SLM Dataset Pipeline" in caplog.text
    assert "No cached dataset. Downloading raw data..." in caplog.text
