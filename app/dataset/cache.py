import logging
import tarfile
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from app.config import settings

logger = logging.getLogger(__name__)

DATASET_NAME = "mixed_dataset"
TOKENIZER_NAME = "gpt2_tokenizer"
DATASET_S3_KEY = f"{DATASET_NAME}.tar.gz"


class DatasetCacheService:
    """
    Service responsible for persisting and restoring processed datasets via S3.
    Includes tokenizer cache to avoid repeated HuggingFace downloads.
    Avoids reprocessing on every training run.
    """

    def __init__(
        self,
        bucket: str = settings.dataset_bucket,
        datasets_dir: str = settings.datasets_dir,
    ):
        self.bucket = bucket
        self.datasets_dir = Path(datasets_dir)
        self.local_path = self.datasets_dir / DATASET_NAME
        self.tokenizer_path = self.datasets_dir / TOKENIZER_NAME
        self.s3 = boto3.client("s3")

    def exists_locally(self) -> bool:
        return self.local_path.exists()

    def tokenizer_exists_locally(self) -> bool:
        return self.tokenizer_path.exists()

    def exists_in_s3(self) -> bool:
        try:
            self.s3.head_object(Bucket=self.bucket, Key=DATASET_S3_KEY)
            return True
        except ClientError:
            return False

    def download(self) -> None:
        tar_path = self.datasets_dir / DATASET_S3_KEY
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading dataset from s3://{self.bucket}/{DATASET_S3_KEY}...")
        self.s3.download_file(self.bucket, DATASET_S3_KEY, str(tar_path))
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=self.datasets_dir)
        tar_path.unlink()
        logger.info("Dataset downloaded and extracted.")

    def upload(self) -> None:
        tar_path = self.datasets_dir / DATASET_S3_KEY
        logger.info("Compressing processed dataset...")
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(self.local_path, arcname=DATASET_NAME)
            if self.tokenizer_path.exists():
                tar.add(self.tokenizer_path, arcname=TOKENIZER_NAME)
                logger.info("Tokenizer included in archive.")
        logger.info(f"Uploading dataset to s3://{self.bucket}/{DATASET_S3_KEY}...")
        self.s3.upload_file(str(tar_path), self.bucket, DATASET_S3_KEY)
        tar_path.unlink()
        logger.info("Dataset uploaded to S3.")
