from __future__ import annotations

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter


def retryable(exceptions: tuple[type[BaseException], ...] = (Exception,), attempts: int = 5):
    return retry(
        retry=retry_if_exception_type(exceptions),
        stop=stop_after_attempt(attempts),
        wait=wait_exponential_jitter(initial=1, max=30),
        reraise=True,
    )
