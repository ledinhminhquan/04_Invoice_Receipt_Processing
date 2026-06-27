"""Shared API dependencies: the invoice-agent singleton + config loading."""

from __future__ import annotations

import os
from functools import lru_cache

from ..agent.invoice_agent import InvoiceAgent, get_agent
from ..config import AppConfig, load_config
from ..logging_utils import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    path = os.environ.get("INVOICE_AI_INFER_CONFIG")
    if path and os.path.exists(path):
        logger.info("Loading config from %s", path)
        return load_config(path)
    return AppConfig()


def get_invoice_agent() -> InvoiceAgent:
    return get_agent(get_config())


__all__ = ["get_config", "get_invoice_agent"]
