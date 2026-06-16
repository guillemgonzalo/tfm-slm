import logging
from unittest.mock import MagicMock, patch

from app.main import main


@patch("app.chat.ChatService")
@patch("app.training.trainer.TrainingService")
@patch("app.dataset.processor.DatasetProcessor")
@patch("app.dataset.downloader.DatasetDownloader")
def test_main_calls_services(
    mock_downloader_cls: MagicMock,
    mock_processor_cls: MagicMock,
    mock_trainer_cls: MagicMock,
    mock_chat_cls: MagicMock,
    caplog,
):
    """
    Test that the main function correctly orchestrates the services.
    """
    # Create instances returned by the mocked classes
    mock_downloader = mock_downloader_cls.return_value
    mock_processor = mock_processor_cls.return_value
    mock_trainer = mock_trainer_cls.return_value
    mock_chat = mock_chat_cls.return_value

    with caplog.at_level(logging.INFO):
        main()

    # Verify that the services were instantiated and called
    mock_downloader_cls.assert_called_once()
    mock_downloader.download_all.assert_called_once()

    mock_processor_cls.assert_called_once()
    mock_processor.process.assert_called_once_with(total_samples=200000)

    mock_trainer_cls.assert_called_once()
    mock_trainer.train.assert_called_once_with(
        epochs=5, batch_size=32, grad_accum_steps=2
    )

    mock_chat_cls.assert_called_once()
    mock_chat.run.assert_called_once()

    # Verify that logs are generated correctly
    assert "Starting SLM Dataset Pipeline" in caplog.text
    assert "Pipeline execution finished successfully." in caplog.text
    assert "Starting interactive chat session..." in caplog.text
