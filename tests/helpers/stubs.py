"""
Stand-ins for the two things a test cannot have: the network and a person.

These are the only fakes in the suite. Storage is real -- real LMDB, real
parquet, real sqlite in a tmp_path -- because three of the package's documented
invariants ARE storage-format invariants, and a mocked store would assert that
the mock was called rather than that the bytes survived. The network and the GUI
have no local truth to check against, so faking them loses nothing.
"""

import cv2


class FakeResponse:
    """One canned HTTP response: bytes, JSON, and a status."""

    def __init__(self, content=b"", json_data=None, status_code=200,
                 headers=None):
        self.content = content
        self._json = json_data
        self.status_code = status_code
        self.headers = headers or {}

    def json(self):
        if self._json is None:
            raise ValueError("no JSON in this response")
        return self._json

    def iter_content(self, chunk_size=8192):
        for start in range(0, len(self.content), chunk_size):
            yield self.content[start:start + chunk_size]

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"{self.status_code} for this url")

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class FakeSession:
    """
    A requests.Session that answers from a routing table and records what it was
    asked for.

    routes -- {url substring: FakeResponse, or a list of them, or a callable
               taking (url, kwargs)}. A list is consumed one call at a time,
               which is how a polling endpoint that reports "pending" and then
               "ready" gets expressed. First matching substring wins, so routes
               should be ordered specific-before-general.

    Every network function in the package takes an injectable `session=`, so
    this reaches all of them without monkeypatching a module attribute -- which
    is worth preserving as the convention it is.
    """

    def __init__(self, routes=None):
        self.routes = dict(routes or {})
        self.calls = []
        self.headers = {}

    def _respond(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        for pattern, response in self.routes.items():
            if pattern in url:
                if isinstance(response, list):
                    if not response:
                        raise AssertionError(
                            f"FakeSession ran out of responses for {pattern!r}")
                    return response.pop(0)
                if callable(response):
                    return response(url, kwargs)
                return response
        raise AssertionError(
            f"FakeSession has no route for {url!r} (routes: {sorted(self.routes)})"
        )

    def get(self, url, **kwargs):
        return self._respond("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self._respond("POST", url, **kwargs)

    def urls(self):
        """Just the urls, in call order -- what most assertions actually want."""
        return [url for _method, url, _kwargs in self.calls]


class FakeCv2:
    """
    cv2 with its five blocking/windowing calls replaced, and everything else
    forwarded to the real thing.

    Forwarding matters: the annotation and manual-segmentation code paths
    legitimately call `cv2.circle`, `cv2.cvtColor`, and so on to build the panel
    a test then asserts on. Replacing the whole module with a mock would make
    those return nothing and the assertions meaningless.

    **Patch this onto the module under test** (`monkeypatch.setattr(manual,
    "cv2", FakeCv2(...))`), never onto the global cv2 -- these functions share
    cv2 with `visualization.panels`, and a global patch would leak into every
    other test in the session.

    keys   -- the key codes waitKey() returns, in order. Running out raises
              rather than returning a default: a stub that doesn't script enough
              keys is a broken test, and the real code's `while True` loop means
              the alternative is hanging forever.
    clicks -- [(event, x, y)] delivered to the mouse callback. Each waitKey()
              call delivers one before returning its key, which is what
              exercises the click-collecting loops.
    """

    def __init__(self, keys=(), clicks=()):
        self.keys = list(keys)
        self.clicks = list(clicks)
        self.shown = []
        self.destroyed = []
        self._callback = None

    def __getattr__(self, name):
        # Only reached for names this class doesn't define, so the five
        # overrides below always win and everything else is real cv2.
        return getattr(cv2, name)

    def imshow(self, window, image):
        self.shown.append((window, image.copy()))

    def setMouseCallback(self, window, callback, param=None):
        self._callback = callback

    def waitKey(self, delay=0):
        if self._callback is not None and self.clicks:
            event, x, y = self.clicks.pop(0)
            self._callback(event, x, y, 0, None)
        if not self.keys:
            raise AssertionError(
                "FakeCv2.waitKey ran dry -- the code under test is still "
                "waiting for input. Script another key, or another click."
            )
        return self.keys.pop(0)

    def destroyWindow(self, window):
        self.destroyed.append(window)

    def destroyAllWindows(self):
        self.destroyed.append("*")
