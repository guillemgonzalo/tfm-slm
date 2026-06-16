#!/usr/bin/env python3
"""
Inference entry point: Load trained model checkpoint and run interactive chat.
Checkpoint is automatically downloaded from S3 if not available locally.

Usage:
  uv run tfm-slm-inference
"""

import logging

from app.chat import ChatService

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting inference session...")
    chat_service = ChatService()
    chat_service.run()


if __name__ == "__main__":
    main()
