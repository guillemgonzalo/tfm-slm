import logging
from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizer

from app.config import settings

logger = logging.getLogger(__name__)

TOKENIZER_NAME = "gpt2"
TOKENIZER_LOCAL_DIR = "gpt2_tokenizer"


class TokenizerService:
    """
    Service responsible for loading and persisting the GPT-2 tokenizer.
    Saves locally to avoid repeated HuggingFace downloads on each training run.
    """

    def __init__(self, datasets_dir: str = settings.datasets_dir):
        self.local_path = Path(datasets_dir) / TOKENIZER_LOCAL_DIR

    def exists_locally(self) -> bool:
        return self.local_path.exists()

    def save(self) -> None:
        logger.info(f"Downloading tokenizer '{TOKENIZER_NAME}' from HuggingFace...")
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        self.local_path.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(self.local_path))
        logger.info(f"Tokenizer saved to {self.local_path}")

    def load(self) -> PreTrainedTokenizer:
        path = str(self.local_path) if self.exists_locally() else TOKENIZER_NAME
        logger.info(f"Loading tokenizer from {path}...")
        tokenizer = AutoTokenizer.from_pretrained(path)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
