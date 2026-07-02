import logging

from app.dataset.cache import DatasetCacheService
from app.dataset.downloader import DatasetDownloader
from app.dataset.processor import DatasetProcessor
from app.dataset.tokenizer import TokenizerService
from app.training.trainer import TrainingService

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting SLM Dataset Pipeline")

    # 1. Dataset Cache Service — restore processed dataset from S3 if available
    cache = DatasetCacheService()
    if cache.exists_locally():
        logger.info("Processed dataset found locally. Skipping download/processing.")
    elif cache.exists_in_s3():
        logger.info("Processed dataset found in S3. Restoring...")
        cache.download()
    else:
        # 2. Dataset Downloading Service
        logger.info("No cached dataset. Downloading raw data...")
        downloader = DatasetDownloader()
        downloader.download_all()

        # 3. Dataset Processing Service
        logger.info("Processing and mixing datasets...")
        processor = DatasetProcessor()
        processor.process(total_samples=494_876, benchmark_samples=148_437)

        # 4. Tokenizer Service — save locally before upload (included in tar)
        tokenizer_service = TokenizerService()
        tokenizer_service.save()

        # 5. Upload processed dataset + tokenizer to S3 for future runs
        cache.upload()

    # 6. Training Service
    logger.info("Phase: Training Hybrid Transformer-GRU Model...")
    trainer = TrainingService()
    trainer.train(epochs=15, batch_size=64, grad_accum_steps=6)


if __name__ == "__main__":
    main()
