import logging
from pathlib import Path

from app.config import settings
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
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
        self,
        output_path: str = "mixed_dataset",
        total_samples: int = 494_876,
        benchmark_samples: int = 148_437,
    ) -> DatasetDict | None:
        """
        Mixes, tokenizes, splits into disjoint train/benchmark sets, and
        saves the dataset if the output doesn't exist. `benchmark_samples`
        is a fixed-size holdout, genuinely excluded from training; everything
        else (total_samples - benchmark_samples) goes to the train split.
        """
        if benchmark_samples >= total_samples:
            raise ValueError(
                f"benchmark_samples ({benchmark_samples}) must be smaller than total_samples ({total_samples})"
            )
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
        # ultrachat is the only source with enough headroom (>1.4M rows) to
        # absorb the shortfall from the other sources, which are naturally
        # smaller than their weighted quota (e.g. alpaca has only 52,002
        # rows total). Load it last and top it up with whatever the fixed
        # sources came up short by, so the total still hits total_samples.
        elastic_key = "ultrachat"

        logger.info(f"Creating mixed dataset with total samples: {total_samples}")

        datasets_to_mix = []
        fixed_total = 0
        for key, weight in weights.items():
            if key == elastic_key:
                continue
            num_samples = int(total_samples * weight)
            if num_samples == 0:
                continue

            ds = self._get_dataset_subset(key, num_samples)
            if ds:
                fixed_total += len(ds)
                ds = self._harmonize_dataset(ds, key)
                datasets_to_mix.append(ds)

        elastic_num_samples = total_samples - fixed_total
        ds = self._get_dataset_subset(elastic_key, elastic_num_samples)
        if ds:
            ds = self._harmonize_dataset(ds, elastic_key)
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

        actual_total = len(tokenized_ds)
        actual_benchmark = benchmark_samples
        if actual_total <= benchmark_samples:
            logger.warning(
                f"Requested {total_samples} total samples but only {actual_total} "
                f"were available across source datasets, not enough to hold out "
                f"{benchmark_samples} for benchmark. Using a 10% holdout instead."
            )
            actual_benchmark = max(1, round(actual_total * 0.1))
        elif actual_total < total_samples:
            logger.warning(
                f"Requested {total_samples} total samples but only {actual_total} "
                f"were available across source datasets. Keeping the fixed "
                f"{benchmark_samples}-sample benchmark holdout; the rest "
                f"({actual_total - benchmark_samples}) goes to training."
            )

        # Split into disjoint train/benchmark sets (data is already shuffled
        # above, so this positional split is equivalent to a random split and
        # avoids any overlap between the samples used for training and evaluation).
        split_ds = tokenized_ds.train_test_split(test_size=actual_benchmark, seed=42)
        split_ds["benchmark"] = split_ds.pop("test")

        # Save to disk
        split_ds.save_to_disk(str(save_path))
        logger.info(
            f"Saved split dataset to {save_path}: "
            f"{len(split_ds['train'])} train / {len(split_ds['benchmark'])} benchmark samples"
        )

        return split_ds

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

        if key == "sharegpt":

            def format_sharegpt(example):
                turns = example["conversations"]
                return {
                    "text": "\n".join(
                        f"{turn['from']}: {turn['value']}" for turn in turns
                    )
                }

            return ds.map(format_sharegpt, remove_columns=ds.column_names)

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
                data_files=config.get("data_files"),
                cache_dir=str(self.output_dir),
                split="train",
            )
            if isinstance(ds, Dataset) and len(ds) > num_samples:
                ds = ds.select(range(num_samples))
            return ds
        except Exception as e:
            logger.error(f"Error loading {key}: {e}")
            return None
