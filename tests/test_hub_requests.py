import unittest
from unittest.mock import Mock, patch

from huggingface_hub.errors import HfHubHTTPError
from requests import Response

from tabbiomed.data.hub import hub_request


def http_error(status, **headers):
    response = Response()
    response.status_code = status
    response.headers.update(headers)
    return HfHubHTTPError("HTTP error", response=response)


class HubRequests(unittest.TestCase):
    def test_waits_for_quota_reset_then_returns_result(self):
        call = Mock(side_effect=[http_error(429, RateLimit='"api";r=0;t=27'), "done"])
        with patch("tabbiomed.data.hub.time.sleep") as sleep:
            self.assertEqual(hub_request(call, "argument"), "done")
        sleep.assert_called_once_with(28)
        self.assertEqual(call.call_count, 2)
        call.assert_called_with("argument")

    def test_respects_retry_after_and_reset_headers(self):
        error = http_error(429, **{"RateLimit": '"api";r=0;t=12', "Retry-After": "30"})
        with patch("tabbiomed.data.hub.time.sleep") as sleep:
            hub_request(Mock(side_effect=[error, None]))
        sleep.assert_called_once_with(31)

    def test_waits_one_quota_window_without_reset_header(self):
        with patch("tabbiomed.data.hub.time.sleep") as sleep:
            hub_request(Mock(side_effect=[http_error(429), None]))
        sleep.assert_called_once_with(301)

    def test_does_not_retry_authentication_errors(self):
        call = Mock(side_effect=http_error(401))
        with patch("tabbiomed.data.hub.time.sleep") as sleep:
            with self.assertRaises(HfHubHTTPError):
                hub_request(call)
        call.assert_called_once()
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
