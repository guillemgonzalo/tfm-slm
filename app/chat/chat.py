import logging
import sys
from pathlib import Path

import boto3
import torch
from app.config import settings
from app.model.architecture import HybridConfig, HybridModel
from botocore.exceptions import ClientError
from transformers import AutoTokenizer, PreTrainedTokenizer

logger = logging.getLogger(__name__)


class ChatService:
    """
    Service responsible for interactive chat inference with the hybrid model.
    Handles checkpoint loading from S3, model initialization, and chat loop.
    """

    def __init__(self, checkpoint_path: Path = Path(".output/checkpoint.pt")) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.s3_client = boto3.client("s3")
        self.bucket_name = settings.checkpoint_bucket

        self.model: HybridModel | None = None
        self.tokenizer: PreTrainedTokenizer | None = None
        self.checkpoint: dict[str, object] | None = None

    def _ensure_checkpoint_local(self) -> bool:
        """
        Ensures checkpoint exists locally, downloading from S3 if necessary.
        Returns True if checkpoint available, False otherwise.
        """
        if self.checkpoint_path.exists():
            return True

        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            print("Checkpoint not found locally. Downloading from S3...")
            print(f"  Bucket: {self.bucket_name}")
            self.s3_client.download_file(
                self.bucket_name,
                "checkpoint.pt",
                str(self.checkpoint_path),
            )
            print("✅ Checkpoint downloaded from S3")
            return True
        except ClientError as e:
            print(f"Error: No checkpoint found locally or in S3. Details: {e}")
            return False

    def _load_tokenizer(self) -> None:
        """Loads and configures GPT-2 tokenizer for model inference."""
        print("Cargando tokenizer (GPT-2)...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token  # type: ignore
        self.tokenizer = tokenizer  # type: ignore

    def _load_model(self) -> None:
        """Loads HybridModel and checkpoint weights from disk."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")

        print(f"Cargando pesos desde {self.checkpoint_path}...")

        config = HybridConfig(
            vocab_size=self.tokenizer.vocab_size,
            max_position_embeddings=1024,
            hidden_size=768,
            num_layers=12,
            num_heads=12,
            intermediate_size=3072,
        )

        model = HybridModel(config)
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        self.model = model
        self.checkpoint = checkpoint

    def _generate_response(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> str:
        """Generates model response for given prompt."""
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("Tokenizer or model not loaded")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs["input_ids"].shape[1]

        autocast_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if self.device.type == "cuda" else torch.amp.autocast("cpu")
        with torch.no_grad(), autocast_ctx:
            output_ids = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
            )

        response = self.tokenizer.decode(
            output_ids[0][input_length:], skip_special_tokens=True
        )
        response_str = response if isinstance(response, str) else ""
        return response_str.split("User:")[0].strip()

    def _run_chat_loop(self) -> None:
        """Manages interactive chat session with model."""
        print("--- Sesión de Chat Interactiva (tfm-slm) ---")
        print("Escribe 'salir' para terminar.\n")

        while True:
            user_query = input("Tú: ")
            if user_query.lower() in ["salir", "exit", "quit"]:
                break

            prompt = f"User: {user_query}\nAssistant:"
            response = self._generate_response(prompt)
            print(f"tfm-slm: {response}")

    def run(self) -> None:
        """
        Main orchestration: ensures checkpoint, loads model, starts chat loop.
        """
        if not self._ensure_checkpoint_local():
            return

        self._load_tokenizer()
        self._load_model()

        if self.checkpoint is None:
            raise RuntimeError("Checkpoint not loaded")
        epoch = self.checkpoint.get("epoch", "unknown")
        print(f"\n✅ Modelo cargado (Epoca: {epoch})")

        if sys.stdin.isatty():
            self._run_chat_loop()
        else:
            print("No interactive terminal. Skipping chat loop.")


if __name__ == "__main__":
    chat_service = ChatService()
    chat_service.run()
