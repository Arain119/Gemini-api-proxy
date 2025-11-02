import asyncio
import json
import unittest
from typing import Any, Dict, Optional
from unittest.mock import patch

from google.oauth2 import credentials as google_credentials

from cli_auth import (
    DEFAULT_SCOPES,
    RESTRICTED_SCOPES,
    _sanitize_cli_scope_fields,
    _sanitize_credentials_instance,
    import_cli_credentials,
)


class DummyDB:
    def __init__(self) -> None:
        self._next_account_id = 1
        self._next_key_id = 1
        self.accounts: Dict[int, Dict[str, Any]] = {}
        self.keys: Dict[str, Dict[str, Any]] = {}
        self.created_credentials: Optional[str] = None
        self.updated_credentials: Optional[str] = None

    def create_cli_account(self, credentials_json: str, email: Optional[str], label: Optional[str]) -> int:
        account_id = self._next_account_id
        self._next_account_id += 1
        self.created_credentials = credentials_json
        self.accounts[account_id] = {
            "credentials": credentials_json,
            "account_email": email,
            "status": 1,
            "label": label,
        }
        return account_id

    def add_gemini_key(self, key_value: str, source_type: str, metadata: Dict[str, Any]) -> bool:
        key_id = self._next_key_id
        self._next_key_id += 1
        self.keys[key_value] = {"id": key_id, "metadata": dict(metadata)}
        return True

    def get_gemini_key_by_value(self, key_value: str) -> Optional[Dict[str, Any]]:
        entry = self.keys.get(key_value)
        if not entry:
            return None
        return {"id": entry["id"], "metadata": dict(entry["metadata"])}

    def update_gemini_key(self, key_id: int, metadata: Dict[str, Any]) -> bool:
        for entry in self.keys.values():
            if entry["id"] == key_id:
                entry["metadata"] = dict(metadata)
                return True
        return False

    def update_cli_account_credentials(self, account_id: int, credentials_json: str, email: Optional[str]) -> bool:
        account = self.accounts[account_id]
        account["credentials"] = credentials_json
        account["account_email"] = email
        self.updated_credentials = credentials_json
        return True

    def get_cli_account(self, account_id: int) -> Optional[Dict[str, Any]]:
        return self.accounts.get(account_id)

    def touch_cli_account(self, account_id: int) -> None:
        account = self.accounts.setdefault(account_id, {})
        account["touched"] = True


class CliAuthScopeTests(unittest.TestCase):
    def test_sanitize_scope_fields_filters_restricted_entries(self) -> None:
        restricted_scope = next(iter(RESTRICTED_SCOPES))
        info = {
            "scope": f"{restricted_scope} {DEFAULT_SCOPES[0]}",
            "scopes": [restricted_scope, DEFAULT_SCOPES[1]],
        }

        scopes, changed = _sanitize_cli_scope_fields(info)

        self.assertTrue(changed)
        self.assertNotIn(restricted_scope, scopes)
        for scope in DEFAULT_SCOPES:
            self.assertIn(scope, scopes)
        self.assertEqual(info["scopes"], scopes)
        self.assertEqual(info["scope"].split(), scopes)

    def test_sanitize_credentials_instance_removes_restricted_scopes(self) -> None:
        restricted_scope = next(iter(RESTRICTED_SCOPES))
        info = {
            "client_id": "dummy-client",
            "client_secret": "dummy-secret",
            "refresh_token": "refresh-token",
            "token_uri": "https://oauth2.googleapis.com/token",
            "token": "ya29.test",
            "scope": f"{restricted_scope} {' '.join(DEFAULT_SCOPES)}",
        }
        info["scopes"] = info["scope"].split()

        credentials = google_credentials.Credentials.from_authorized_user_info(info, scopes=info["scopes"])
        credentials.token = "ya29.test"

        sanitized, serialized, changed = _sanitize_credentials_instance(credentials)

        self.assertTrue(changed)
        payload = json.loads(serialized)
        self.assertNotIn(restricted_scope, payload.get("scopes", []))
        self.assertEqual(sanitized.token, "ya29.test")
        for scope in DEFAULT_SCOPES:
            self.assertIn(scope, payload.get("scopes", []))

    def test_import_cli_credentials_persists_sanitized_scopes(self) -> None:
        restricted_scope = next(iter(RESTRICTED_SCOPES))
        payload = {
            "client_id": "dummy-client",
            "client_secret": "dummy-secret",
            "refresh_token": "refresh-token",
            "token_uri": "https://oauth2.googleapis.com/token",
            "scope": f"{restricted_scope} {' '.join(DEFAULT_SCOPES)}",
        }

        async def fake_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        async def fake_fetch_account_email(token: str) -> str:
            return "user@example.com"

        def fake_refresh(self, request) -> None:  # pragma: no cover - simple stub
            self.token = "ya29.refreshed"

        async def run_test() -> None:
            db = DummyDB()
            with patch("cli_auth.asyncio.to_thread", new=fake_to_thread), patch(
                "cli_auth.fetch_account_email", new=fake_fetch_account_email
            ), patch.object(google_credentials.Credentials, "refresh", new=fake_refresh):
                response = await import_cli_credentials(
                    db=db,
                    credentials_json=json.dumps(payload),
                    label="test",
                )

            self.assertEqual(response.account_id, 1)
            self.assertEqual(response.account_email, "user@example.com")

            created_payload = json.loads(db.created_credentials or "{}")
            updated_payload = json.loads(db.updated_credentials or "{}")

            for data in (created_payload, updated_payload):
                self.assertNotIn(restricted_scope, data.get("scopes", []))
                self.assertEqual(
                    sorted(set(DEFAULT_SCOPES) - set(data.get("scopes", []))),
                    [],
                )

        asyncio.run(run_test())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
