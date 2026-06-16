import logging
from pathlib import Path

from app.config import settings
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import AutoTokenizer

from .constants import DATASETS_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class DatasetProcessor:
    """
    Service responsible for mixing, harmonizing, and tokenizing datasets.
    """

    def __init__(self, output_dir: str = settings.datasets_dir):
        self.output_dir = Path(output_dir)
        tok = AutoTokenizer.from_pretrained("gpt2")
        if tok is None:
            raise RuntimeError("Failed to load tokenizer")
        self.tokenizer = tok
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def process(
        self, output_path: str = "mixed_dataset", total_samples: int = 100000
    ) -> Dataset | None:
        """
        Mixes, tokenizes, and saves datasets if the output doesn't exist.
        """
        save_path = self.output_dir / output_path

        if save_path.exists():
            logger.info(f"Processed dataset already exists at {save_path}. Skipping.")
            return None

        weights = {
            "open_assistant": 0.30,
            "ultrachat": 0.30,
            "alpaca": 0.20,
            "sharegpt": 0.20,
        }

        logger.info(f"Creating mixed dataset with total samples: {total_samples}")

        datasets_to_mix = []
        for key, weight in weights.items():
            num_samples = int(total_samples * weight)
            if num_samples == 0:
                continue

            ds = self._get_dataset_subset(key, num_samples)
            if ds:
                ds = self._harmonize_dataset(ds, key)
                datasets_to_mix.append(ds)

        if not datasets_to_mix:
            raise ValueError("No datasets were loaded successfully.")

        # Concatenate and shuffle
        mixed_ds = concatenate_datasets(datasets_to_mix)
        mixed_ds = mixed_ds.shuffle(seed=42)

        # Tokenization before saving
        logger.info("Tokenizing the entire mixed dataset...")

        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=1024,
            )

        tokenized_ds = mixed_ds.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing",
        )

        # Save to disk
        tokenized_ds.save_to_disk(str(save_path))
        logger.info(f"Tokenized mixed dataset saved to {save_path}")

        return tokenized_ds

    def _harmonize_dataset(self, ds: Dataset, key: str) -> Dataset:
        """
        Ensures the dataset has a 'text' column based on its specific structure.
        """
        if "text" in ds.column_names and key != "ultrachat":
            return ds.select_columns(["text"])

        if key == "ultrachat":

            def format_ultrachat(example):
                return {"text": "\n".join(example["data"])}

            return ds.map(format_ultrachat, remove_columns=ds.column_names)

        if key == "the_stack_yaml":
            if "content" in ds.column_names:
                return ds.rename_column("content", "text").select_columns(["text"])

        logger.warning(
            f"Using fallback harmonization for {key}. Columns: {ds.column_names}"
        )
        return ds

    def _get_dataset_subset(self, key: str, num_samples: int) -> Dataset | None:
        config = DATASETS_CONFIG.get(key)
        if not config:
            return None

        try:
            # Using type ignore because datasets.load_dataset has complex overloads
            # that ty sometimes struggles to match even when correct.
            ds = load_dataset(  # type: ignore
                path=config["path"],
                name=config["name"],
                data_dir=config.get("data_dir"),
                cache_dir=str(self.output_dir),
                split="train",
                trust_remote_code=settings.trust_remote_code,
            )
            if isinstance(ds, Dataset) and len(ds) > num_samples:
                ds = ds.select(range(num_samples))
            return ds
        except Exception as e:
            logger.error(f"Error loading {key}: {e}")
            return None
