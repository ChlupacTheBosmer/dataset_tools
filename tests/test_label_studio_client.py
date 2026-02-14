from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

from dataset_tools.config import load_config
from dataset_tools.label_studio import client


class _FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return dict(self._payload)


class _FakeLSClient:
    def __init__(self, url, api_key, connected=True):
        self.url = url
        self.api_key = api_key
        self._connected = connected

    def check_connection(self):
        return self._connected


class LabelStudioClientTests(unittest.TestCase):
    def test_import_label_studio_client_primary_and_legacy(self):
        primary = types.ModuleType("label_studio_sdk")
        primary.Client = object
        with patch.dict(sys.modules, {"label_studio_sdk": primary}):
            self.assertIs(client._import_label_studio_client(), object)

        legacy = types.ModuleType("label_studio_sdk._legacy")
        legacy.Client = int
        # Make primary missing and legacy present
        with patch.dict(sys.modules, {"label_studio_sdk": None, "label_studio_sdk._legacy": legacy}):
            self.assertIs(client._import_label_studio_client(), int)

    def test_import_label_studio_client_raises_when_missing(self):
        with patch.dict(sys.modules, {"label_studio_sdk": None, "label_studio_sdk._legacy": None}):
            with self.assertRaises(RuntimeError):
                client._import_label_studio_client()

    def test_resolve_access_token(self):
        jwt_like = "eyJ" + ("x" * 80)
        self.assertEqual(client._resolve_access_token("https://x", "plain"), "plain")

        with patch("dataset_tools.label_studio.client.requests", None):
            self.assertEqual(client._resolve_access_token("https://x", jwt_like), jwt_like)

        fake_requests = types.SimpleNamespace(
            post=lambda *args, **kwargs: _FakeResponse(200, {"access": "access-token"}),
            RequestException=Exception,
        )
        with patch("dataset_tools.label_studio.client.requests", fake_requests):
            self.assertEqual(client._resolve_access_token("https://x", jwt_like), "access-token")

    def test_connect_to_label_studio_and_check_connection(self):
        with patch("dataset_tools.label_studio.client._import_label_studio_client", return_value=_FakeLSClient), patch(
            "dataset_tools.label_studio.client._resolve_access_token",
            return_value="resolved",
        ):
            ls = client.connect_to_label_studio("https://ls", "token")
        self.assertEqual(ls.api_key, "resolved")

        class BadClient(_FakeLSClient):
            def __init__(self, url, api_key):
                super().__init__(url, api_key, connected=False)

        with patch("dataset_tools.label_studio.client._import_label_studio_client", return_value=BadClient), patch(
            "dataset_tools.label_studio.client._resolve_access_token",
            return_value="x",
        ):
            with self.assertRaises(RuntimeError):
                client.connect_to_label_studio("https://ls", "token")

    def test_ensure_label_studio_client(self):
        cfg = load_config(
            local_config_path="/tmp/does-not-exist.json",
            overrides={"label_studio": {"url": "https://ls", "api_key": "k"}},
        )
        with patch("dataset_tools.label_studio.client.connect_to_label_studio", return_value="ls") as connect:
            out = client.ensure_label_studio_client(cfg)
        connect.assert_called_once_with(url="https://ls", api_key="k")
        self.assertEqual(out, "ls")


if __name__ == "__main__":
    unittest.main()
