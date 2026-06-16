from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Global application settings using Pydantic.
    Values can be overridden by environment variables.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Dataset configurations
    datasets_dir: str = ".datasets"
    trust_remote_code: bool = True

    # Global project info
    project_name: str = "tfm-slm"
    version: str = "0.1.0"

    # S3 configurations
    checkpoint_bucket: str = "tfm-slm-checkpoints"
    dataset_bucket: str = "tfm-slm-datasets"
    benchmark_bucket: str = "tfm-slm-benchmarks"

    # Hardware configurations
    gpu_vram_gb: int = 48


# Global instance
settings = Settings()
