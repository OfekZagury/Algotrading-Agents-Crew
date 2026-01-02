import time
import pytest
from core.rate_limiter import RateLimiter, RateLimitExceededException


def test_rate_limiter_allows_requests_within_limit():
    # Definition: 2 requests per second allowed
    limiter = RateLimiter(max_calls=2, period_seconds=1)

    # These should pass
    assert limiter.allow_request("alpaca_api") == True
    assert limiter.allow_request("alpaca_api") == True


def test_rate_limiter_blocks_requests_exceeding_limit():
    # Definition: 1 request per second allowed
    limiter = RateLimiter(max_calls=1, period_seconds=1)

    # the first one should pass
    limiter.allow_request("alpaca_api")

    # the second one should raise an exception
    with pytest.raises(RateLimitExceededException):
        limiter.allow_request("alpaca_api")


def test_rate_limiter_resets_after_period():
    # Definition: 1 request per second allowed
    limiter = RateLimiter(max_calls=1, period_seconds=1)

    limiter.allow_request("alpaca_api")  # ראשונה

    # wait for the period to expire
    time.sleep(1.1)

    # now it should allow again
    assert limiter.allow_request("alpaca_api") == True