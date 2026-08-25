"""Synthetic secret redaction and credential hygiene tests."""

from jobapply.utils.redaction import (
    redact_data,
    redact_exception,
    redact_string,
)


def test_redact_google_api_key_in_query_string():
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemma:generateContent?key=AIzaSyA1234567890abcdef_GhijkLmNoPqrStU"
    redacted = redact_string(url)
    assert "AIzaSy" not in redacted
    assert "key=[REDACTED]" in redacted


def test_redact_telegram_bot_token_in_url():
    url = "https://api.telegram.org/bot123456789:ABCdefGhIJklmnOPqrstUVwxyz_1234567/sendMessage"
    redacted = redact_string(url)
    assert "123456789:ABCdef" not in redacted
    assert "api.telegram.org/bot[REDACTED]/sendMessage" in redacted


def test_redact_bearer_auth():
    header = (
        "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0"
    )
    redacted = redact_string(header)
    assert "eyJhbGci" not in redacted
    assert "Bearer [REDACTED]" in redacted or "Authorization: [REDACTED]" in redacted


def test_redact_mongodb_uri_adversarial_complex_passwords():
    # URI with @ in password: full userinfo redacted
    uri1 = "mongodb+srv://app_user:SecretP@ssw0rd123!@cluster0.mongodb.net/db?retryWrites=true"
    redacted1 = redact_string(uri1)
    assert "SecretP@ssw0rd123!" not in redacted1
    assert "app_user" not in redacted1
    assert "mongodb+srv://[REDACTED]@cluster0.mongodb.net/db?retryWrites=true" == redacted1

    # URI with percent-encoded user and password with colons and ats
    uri2 = "mongodb://my%40user:p%3Ass%40word@host1:27017,host2:27017/my_database?authSource=admin"
    redacted2 = redact_string(uri2)
    assert "p%3Ass%40word" not in redacted2
    assert "my%40user" not in redacted2
    assert "mongodb://[REDACTED]@host1:27017,host2:27017/my_database?authSource=admin" == redacted2

    # URI with token only
    uri3 = "mongodb://token_auth_secret_xyz@mongo.internal:27017"
    redacted3 = redact_string(uri3)
    assert "token_auth_secret_xyz" not in redacted3
    assert "mongodb://[REDACTED]@mongo.internal:27017" == redacted3


def test_redact_standalone_tokens():
    raw_google = "My API key is AIzaSyD_DummyFakeKeyForTestingPurposes123"
    assert "AIzaSy" not in redact_string(raw_google)
    assert "[REDACTED]" in redact_string(raw_google)

    raw_telegram = "My bot token is 987654321:AAFakeSyntheticTelegramTokenXYZ123456"
    assert "987654321:" not in redact_string(raw_telegram)
    assert "[REDACTED]" in redact_string(raw_telegram)


def test_redact_configured_literal_secrets():
    extra = ["SuperSecretCustomTokenXYZ999"]
    text = "Logging with SuperSecretCustomTokenXYZ999 inside message"
    redacted = redact_string(text, extra_secrets=extra)
    assert "SuperSecretCustomTokenXYZ999" not in redacted
    assert "[REDACTED]" in redacted


def test_redact_data_preserves_harmless_keys_and_redacts_sensitive_structures():
    payload = {
        "bypass_csp": False,
        "compass_heading": "north",
        "token_count": 42,
        "password_policy": "strict",
        "api_key": "actual_secret_api_key",
        "password": "actual_user_password",
        "google_api_key": "AIzaSyFakeKey1234567890",
        "telegram_bot_token": "123456789:ABCDEF_token_1234567890",
        "authorization": {
            "credentials": "secret_credentials_leaked",
            "scheme": "bearer",
        },
        "token": {
            "value": "token_nested_value_secret",
            "nested_list": ["secret1", "secret2"],
        },
        "cookies": ["cookie1=secret", "cookie2=secret"],
        "safe_container": {
            "title": "Engineer",
            "url": "https://linkedin.com/jobs/123",
        },
    }
    redacted = redact_data(payload)

    # Harmless keys must be preserved
    assert redacted["bypass_csp"] is False
    assert redacted["compass_heading"] == "north"
    assert redacted["token_count"] == 42
    assert redacted["password_policy"] == "strict"
    assert redacted["safe_container"]["title"] == "Engineer"

    # Entire sensitive key values must be [REDACTED]
    assert redacted["api_key"] == "[REDACTED]"
    assert redacted["password"] == "[REDACTED]"
    assert redacted["google_api_key"] == "[REDACTED]"
    assert redacted["telegram_bot_token"] == "[REDACTED]"
    assert redacted["authorization"] == "[REDACTED]"
    assert redacted["token"] == "[REDACTED]"
    assert redacted["cookies"] == "[REDACTED]"


def test_redact_exception_message():
    exc = RuntimeError(
        "HTTP error for https://api.telegram.org/bot123456789:ABCdefGhIJklmnOPqrstUVwxyz_1234567/getUpdates: 401 Unauthorized"
    )
    msg = redact_exception(exc)
    assert "123456789:ABCdef" not in msg
    assert "api.telegram.org/bot[REDACTED]" in msg


def test_redaction_idempotency():
    text = "Error calling https://api.telegram.org/bot123456789:ABCdefGhIJklmnOPqrstUVwxyz_1234567/sendMessage with key AIzaSyA1234567890abcdef_GhijkLmNoPqrStU"
    first_pass = redact_string(text)
    second_pass = redact_string(first_pass)
    assert first_pass == second_pass


def test_redaction_preserves_harmless_text():
    normal_text = "Experienced Senior Python Engineer with 5 years experience at Acme Corp. Applied at https://www.linkedin.com/jobs/view/987654321"
    assert redact_string(normal_text) == normal_text


def test_redact_data_covers_suffix_and_prefixed_sensitive_keys():
    payload = {
        "db_password": "super_secret_db_pass",
        "custom_service_api_key": "custom_api_key_12345",
        "nested": {
            "internal_client_secret": "my_client_secret",
            "safe_description": "Standard software engineer",
        },
    }
    redacted = redact_data(payload)
    assert redacted["db_password"] == "[REDACTED]"
    assert redacted["custom_service_api_key"] == "[REDACTED]"
    assert redacted["nested"]["internal_client_secret"] == "[REDACTED]"
    assert redacted["nested"]["safe_description"] == "Standard software engineer"


def test_redact_string_dynamic_env_literals(monkeypatch):
    monkeypatch.setenv("JOBAPPLY_CUSTOM_SECRET", "UltraTopSecretToken987")
    monkeypatch.setenv("DATABASE_PASSWORD", "ComplexPass998811!")
    text = "Connecting with UltraTopSecretToken987 and password ComplexPass998811!"
    redacted = redact_string(text)
    assert "UltraTopSecretToken987" not in redacted
    assert "ComplexPass998811!" not in redacted
    assert "[REDACTED]" in redacted
