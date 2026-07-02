DATASETS_CONFIG: dict[str, dict[str, str | list[str] | None]] = {
    "open_assistant": {"path": "OpenAssistant/oasst1", "name": None},
    "sharegpt": {
        "path": "json",
        "name": None,
        "data_files": [
            "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/HTML_cleaned_raw_dataset/sg_90k_part1_html_cleaned.json",
            "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/HTML_cleaned_raw_dataset/sg_90k_part2_html_cleaned.json",
        ],
    },
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
