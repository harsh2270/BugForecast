import pytest
from predict.server import _parse_github_url


@pytest.mark.parametrize("raw,expected", [
    ("https://github.com/vllm-project/vllm?utm_source=chatgpt.com",
     "https://github.com/vllm-project/vllm.git"),
    ("https://github.com/psf/requests?tab=readme", "https://github.com/psf/requests.git"),
    ("https://github.com/psf/requests/tree/main", "https://github.com/psf/requests.git"),
    ("https://github.com/psf/requests/blob/main/README.md",
     "https://github.com/psf/requests.git"),
    ("https://github.com/psf/requests#readme", "https://github.com/psf/requests.git"),
    ("https://www.github.com/psf/requests", "https://github.com/psf/requests.git"),
    ("github.com/psf/requests", "https://github.com/psf/requests.git"),
    ("https://github.com/psf/requests.git", "https://github.com/psf/requests.git"),
    ("psf/requests", "https://github.com/psf/requests.git"),
    ("psf/requests/", "https://github.com/psf/requests.git"),
])
def test_valid_urls(raw, expected):
    assert _parse_github_url(raw) == expected


@pytest.mark.parametrize("raw", [
    "", "not a url", "https://github.com/psf",           # no repo
    "https://github.com/psf/",                            # no repo
    "https://gitlab.com/psf/requests",                    # wrong host
    "javascript:alert(1)",                                # scheme abuse
    "https://github.com/psf/requests/../etc",             # traversal attempt
])
def test_invalid_urls(raw):
    assert _parse_github_url(raw) is None
