import logging
from pathlib import Path

from app.config import settings
from datasets import load_dataset
from pydantic import BaseModel

from .constants import DATASETS_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class DatasetConfig(BaseModel):
    """
    Schema for individual dataset configuration.
    """

    path: str
    name: str | None = None
    data_dir: str | None = None


class DatasetDownloader:
    """
    Service responsible for downloading datasets from Hugging Face.
    """

    def __init__(self, output_dir: str = settings.datasets_dir):
        self.output_dir = Path(output_dir)
        self._ensure_directory_exists()

    def download_all(self) -> None:
        """Downloads all datasets defined in configuration."""
        for key, raw_config in DATASETS_CONFIG.items():
            config = DatasetConfig(**raw_config)
            self.download_single(key, config)

    def download_single(self, key: str, config: DatasetConfig) -> None:
        """
        Downloads a single dataset if it doesn't already exist in the output directory.
        """
        if self._is_already_downloaded(key, config):
            logger.info(f"Dataset '{key}' already exists. Skipping download.")
            return

        logger.info(f"Starting download for: {key} ({config.path})")
        try:
            load_dataset(
                config.path,
                name=config.name,
                cache_dir=str(self.output_dir),
                data_dir=config.data_dir,
                trust_remote_code=settings.trust_remote_code,
            )
            logger.info(f"Successfully downloaded {key}")
        except Exception as error:
            logger.error(f"Failed to download {key}: {error}")

    def _is_already_downloaded(self, key: str, config: DatasetConfig) -> bool:
        """
        Checks if the dataset directory exists in the cache.
        Hugging Face datasets are stored in cache_dir/path_to_dataset
        """
        # Simplistic check: if the folder exists and is not empty
        # A more robust check would involve checking the actual HF lock files/metadata
        safe_path = config.path.replace("/", "___")
        dataset_path = self.output_dir / safe_path
        return dataset_path.exists() and any(dataset_path.iterdir())

    def _ensure_directory_exists(self) -> None:
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {self.output_dir}")
