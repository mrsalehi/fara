"""Distributed-training helpers used by scripts/train_fara.py.

Wraps rank / world-size lookups so callers don't have to remember which env
vars torchrun / accelerate set, and prefers the torch.distributed process group
once it's been initialized. Mirrors torch.distributed's own API: every helper
takes an optional `group=None` and queries the default group when omitted.
"""

import logging
import os
from typing import Optional

import torch.distributed as dist


# Module logger. Configure via setup_logging() at startup.
_logger = logging.getLogger("fara.train")


def _pg_available() -> bool:
    """True iff torch.distributed has a default process group initialized."""
    return dist.is_available() and dist.is_initialized()


def get_rank(group: Optional["dist.ProcessGroup"] = None) -> int:
    """Global rank within `group` (or the default group). Falls back to env
    vars when no process group has been initialized yet.
    """
    if _pg_available():
        return dist.get_rank(group=group)
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))


def get_local_rank() -> int:
    """Local (per-node) rank. torch.distributed has no API for this — env only."""
    return int(os.environ.get("LOCAL_RANK", "0"))


def get_world_size(group: Optional["dist.ProcessGroup"] = None) -> int:
    if _pg_available():
        return dist.get_world_size(group=group)
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_main_process(group: Optional["dist.ProcessGroup"] = None) -> bool:
    return get_rank(group=group) == 0


def setup_logging(
    level: int = logging.INFO,
    fmt: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt: str = "%H:%M:%S",
    group: Optional["dist.ProcessGroup"] = None,
) -> None:
    """Configure root logging once. Rank-0 logs at `level`; non-main ranks are
    silenced down to ERROR so worker processes don't spam the slurm log.
    Also raises common library loggers (transformers, accelerate, datasets) to
    ERROR on non-main ranks.
    """
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, force=True)
    if not is_main_process(group=group):
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
        for name in ("fara", "transformers", "accelerate", "datasets"):
            logging.getLogger(name).setLevel(logging.ERROR)


def log(msg: str, *args, level: int = logging.INFO) -> None:
    """Emit through the `fara.train` logger. Non-main ranks are filtered out
    by setup_logging() raising their level to ERROR, so callers don't need
    a manual rank check.
    """
    _logger.log(level, msg, *args)
