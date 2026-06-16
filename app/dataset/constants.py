DATASETS_CONFIG: dict[str, dict[str, str | None]] = {
    "open_assistant": {"path": "OpenAssistant/oasst1", "name": None},
    "sharegpt": {"path": "anon8231489123/ShareGPT_Vicuna_unfiltered", "name": None},
    "alpaca": {"path": "tatsu-lab/alpaca", "name": None},
    "ultrachat": {"path": "stingning/ultrachat", "name": None},
    # The Stack is gated and requires manual approval on Hugging Face.
    # "the_stack_yaml": {
    #     "path": "bigcode/the-stack",
    #     "name": None,
    #     "data_dir": "data/yaml",
    # },
}

DEFAULT_OUTPUT_DIR: str = ".datasets"
