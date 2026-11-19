from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import re
import time
from huggingface_hub.errors import HfHubHTTPError


def hub_request(call, *args, **kwargs):
    while True:
        try:
            return call(*args, **kwargs)
        except HfHubHTTPError as error:
            response = error.response
            if response is None or response.status_code != 429:
                raise
            headers = response.headers
            reset = re.search(
                "(?:^|;)\\s*t\\s*=\\s*(\\d+)", headers.get("RateLimit", "")
            )
            waits = [float(reset[1])] if reset else []
            retry = headers.get("Retry-After")
            if retry:
                try:
                    waits.append(float(retry))
                except ValueError:
                    try:
                        date = parsedate_to_datetime(retry)
                        waits.append(
                            (date - datetime.now(timezone.utc)).total_seconds()
                        )
                    except (ValueError, TypeError, OverflowError):
                        pass
            seconds = max(1, max(waits, default=300)) + 1
            print(
                f"HF rate limit reached, waiting {seconds:g}s before resuming",
                flush=True,
            )
            time.sleep(seconds)
