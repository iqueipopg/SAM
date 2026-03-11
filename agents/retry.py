"""Reusable retry utility with exponential backoff + jitter.

Usage:
    # Function wrapper
    result = retry_with_backoff(lambda: requests.get(url), max_retries=3)

    # Decorator
    @with_backoff(max_retries=3, base_delay=1.0)
    def my_api_call():
        return requests.get(url)

    # HTTP-aware (skip 404, longer backoff on 429)
    result = retry_with_backoff(
        lambda: requests.get(url),
        retryable_exceptions=(requests.exceptions.RequestException,),
    )
"""

import time
import random
import functools
import logging
import os
import yaml

logger = logging.getLogger(__name__)

# HTTP status codes that should NOT be retried
_NO_RETRY_CODES = {400, 401, 403, 404, 405, 410, 422}

# HTTP status codes that indicate rate limiting (use longer backoff)
_RATE_LIMIT_CODES = {429, 529}


def _load_retry_config() -> dict:
    """Load retry config from config.yaml if available."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("retry", {})
    except Exception:
        return {}


# Load defaults once at import time
_DEFAULTS = _load_retry_config()


def retry_with_backoff(
    func,
    max_retries=None,
    base_delay=None,
    max_delay=None,
    backoff_factor=None,
    retryable_exceptions=(Exception,),
    on_retry=None,
):
    """Retry a function with exponential backoff + jitter.

    Args:
        func: Callable to retry (no arguments — use lambda for args).
        max_retries: Maximum number of retries (default from config or 3).
        base_delay: Initial delay in seconds (default from config or 1.0).
        max_delay: Maximum delay cap in seconds (default from config or 30.0).
        backoff_factor: Multiplier per attempt (default from config or 2.0).
        retryable_exceptions: Tuple of exception types to catch and retry.
        on_retry: Optional callback(attempt, exception, delay) called before sleeping.

    Returns:
        The return value of func().

    Raises:
        The last exception if all retries are exhausted.
    """
    # Apply defaults from config, then hardcoded fallbacks
    if max_retries is None:
        max_retries = _DEFAULTS.get("max_retries", 3)
    if base_delay is None:
        base_delay = _DEFAULTS.get("base_delay", 1.0)
    if max_delay is None:
        max_delay = _DEFAULTS.get("max_delay", 30.0)
    if backoff_factor is None:
        backoff_factor = _DEFAULTS.get("backoff_factor", 2.0)

    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except retryable_exceptions as e:
            last_exception = e

            # Check if this is an HTTP error we should NOT retry
            if _is_non_retryable_http(e):
                raise

            if attempt == max_retries:
                logger.warning(
                    "[retry] All %d retries exhausted: %s", max_retries, e
                )
                raise

            # Calculate delay with exponential backoff
            delay = min(base_delay * (backoff_factor ** attempt), max_delay)

            # Use longer backoff for rate-limit responses
            if _is_rate_limited(e):
                delay = min(delay * 3, max_delay)
                logger.info(
                    "[retry] Rate limited (attempt %d/%d), waiting %.1fs: %s",
                    attempt + 1, max_retries, delay, e,
                )
            else:
                logger.info(
                    "[retry] Attempt %d/%d failed, retrying in %.1fs: %s",
                    attempt + 1, max_retries, delay, e,
                )

            # Add jitter (0-10% of delay)
            jitter = random.uniform(0, delay * 0.1)
            total_delay = delay + jitter

            if on_retry:
                try:
                    on_retry(attempt, e, total_delay)
                except Exception:
                    pass

            time.sleep(total_delay)

    # Should not reach here, but just in case
    raise last_exception


def with_backoff(
    max_retries=None,
    base_delay=None,
    max_delay=None,
    backoff_factor=None,
    retryable_exceptions=(Exception,),
):
    """Decorator version of retry_with_backoff.

    Usage:
        @with_backoff(max_retries=3, base_delay=1.0)
        def fetch_data(url):
            return requests.get(url)
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return retry_with_backoff(
                lambda: func(*args, **kwargs),
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
                backoff_factor=backoff_factor,
                retryable_exceptions=retryable_exceptions,
            )
        return wrapper
    return decorator


def _is_non_retryable_http(exc):
    """Check if an exception wraps an HTTP status that should not be retried."""
    # requests.exceptions.HTTPError
    response = getattr(exc, "response", None)
    if response is not None:
        status = getattr(response, "status_code", None)
        if status and status in _NO_RETRY_CODES:
            return True
    return False


def _is_rate_limited(exc):
    """Check if an exception indicates rate limiting."""
    response = getattr(exc, "response", None)
    if response is not None:
        status = getattr(response, "status_code", None)
        if status and status in _RATE_LIMIT_CODES:
            return True
    # Also check the string representation for common patterns
    msg = str(exc).lower()
    if "429" in msg or "rate limit" in msg or "too many requests" in msg:
        return True
    return False
