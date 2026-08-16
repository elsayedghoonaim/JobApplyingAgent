"""Tests for hardened PDF generation and HTML sanitization."""

from unittest.mock import AsyncMock, patch

import pytest

from jobapply.utils.pdf import (
    _handle_route_abort,
    build_hardened_document,
    markdown_to_pdf,
    sanitize_markdown_html,
)


def test_sanitize_markdown_html_strips_hostile_content():
    hostile_md = """# Candidate Resume
<script>alert('pwned')</script>
<img src="http://evil.com/tracker.png" onerror="alert(1)" />
<a href="javascript:stealCookies()">Click me</a>
<a href="data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg==">Data link</a>
<iframe src="http://malicious.com"></iframe>
<object data="exploit.swf"></object>
<style>body { display: none; }</style>
"""
    clean_html = sanitize_markdown_html(hostile_md)

    assert "<script" not in clean_html
    assert "alert(" not in clean_html
    assert "<img" not in clean_html
    assert "javascript:" not in clean_html
    assert "data:text/html" not in clean_html
    assert "<iframe" not in clean_html
    assert "<object" not in clean_html
    assert "<style" not in clean_html
    assert "<h1>Candidate Resume</h1>" in clean_html


def test_sanitize_markdown_html_preserves_safe_elements():
    valid_md = """# Jane Doe
## Experience
* **Software Engineer** at *Acme Corp*
* Built pipelines with Python

| Skill | Level |
| --- | --- |
| Python | Expert |

[Portfolio](https://janedoe.dev)
"""
    clean_html = sanitize_markdown_html(valid_md)

    assert "<h1>Jane Doe</h1>" in clean_html
    assert "<h2>Experience</h2>" in clean_html
    assert "<strong>Software Engineer</strong>" in clean_html
    assert "<em>Acme Corp</em>" in clean_html
    assert "<ul>" in clean_html
    assert "<li>" in clean_html
    assert "<table>" in clean_html
    assert "<th>Skill</th>" in clean_html
    assert 'href="https://janedoe.dev"' in clean_html
    assert "Portfolio</a>" in clean_html


def test_build_hardened_document_contains_strict_csp():
    doc = build_hardened_document("<p>Hello</p>")
    assert "Content-Security-Policy" in doc
    assert "default-src 'none'" in doc
    assert "script-src 'none'" in doc
    assert "object-src 'none'" in doc
    assert "frame-src 'none'" in doc
    assert "connect-src 'none'" in doc
    assert "font-src 'none'" in doc
    assert "img-src 'none'" in doc
    assert "style-src 'unsafe-inline'" in doc
    assert "<style>" in doc
    assert "<p>Hello</p>" in doc


@pytest.mark.asyncio
async def test_route_abort_handler_aborts_all_request_schemes():
    for scheme in (
        "http://evil.com",
        "https://evil.com",
        "file:///etc/passwd",
        "data:text/html,test",
    ):
        mock_route = AsyncMock()
        mock_route.request.url = scheme
        await _handle_route_abort(mock_route)
        mock_route.abort.assert_awaited_once()


@pytest.mark.asyncio
async def test_markdown_to_pdf_playwright_hardening_and_cleanup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir(parents=True, exist_ok=True)

    mock_pw = AsyncMock()
    mock_browser = AsyncMock()
    mock_context = AsyncMock()
    mock_page = AsyncMock()

    mock_pw.chromium.launch.return_value = mock_browser
    mock_browser.new_context.return_value = mock_context
    mock_context.new_page.return_value = mock_page

    output_pdf = tmp_path / "outputs" / "run1" / "test_resume.pdf"

    with patch("jobapply.utils.pdf.async_playwright") as mock_pw_context:
        mock_pw_cm = AsyncMock()
        mock_pw_cm.start.return_value = mock_pw
        mock_pw_context.return_value = mock_pw_cm

        result = await markdown_to_pdf("# Test Resume", str(output_pdf))

        # Check browser was launched headless
        mock_pw.chromium.launch.assert_awaited_once_with(headless=True)

        # Check context created with JavaScript disabled
        mock_browser.new_context.assert_awaited_once_with(
            java_script_enabled=False,
            bypass_csp=False,
        )

        # Check route blocking was registered
        mock_context.route.assert_awaited_once_with("**/*", _handle_route_abort)

        # Check PDF generated with Letter format
        mock_page.pdf.assert_awaited_once_with(
            path=str(output_pdf.resolve()),
            format="Letter",
            print_background=False,
        )

        # Check full cleanup
        mock_page.close.assert_awaited_once()
        mock_context.close.assert_awaited_once()
        mock_browser.close.assert_awaited_once()
        mock_pw.stop.assert_awaited_once()

        assert result == str(output_pdf.resolve())


@pytest.mark.asyncio
async def test_markdown_to_pdf_cleans_up_on_rendering_failure_without_masking(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir(parents=True, exist_ok=True)

    mock_pw = AsyncMock()
    mock_browser = AsyncMock()
    mock_context = AsyncMock()
    mock_page = AsyncMock()

    # Primary rendering exception
    mock_page.pdf.side_effect = RuntimeError("PrimarySentinelRenderError")

    # Cleanup calls also raise secondary errors
    mock_page.close.side_effect = Exception("PageCloseError")
    mock_context.close.side_effect = Exception("ContextCloseError")
    mock_browser.close.side_effect = Exception("BrowserCloseError")
    mock_pw.stop.side_effect = Exception("PlaywrightStopError")

    mock_pw.chromium.launch.return_value = mock_browser
    mock_browser.new_context.return_value = mock_context
    mock_context.new_page.return_value = mock_page

    output_pdf = tmp_path / "outputs" / "run1" / "test_resume.pdf"

    with patch("jobapply.utils.pdf.async_playwright") as mock_pw_context:
        mock_pw_cm = AsyncMock()
        mock_pw_cm.start.return_value = mock_pw
        mock_pw_context.return_value = mock_pw_cm

        # Primary sentinel MUST be the raised exception
        with pytest.raises(RuntimeError, match="PrimarySentinelRenderError"):
            await markdown_to_pdf("# Test Resume", str(output_pdf))

        # Verify all cleanup methods were still awaited despite errors
        mock_page.close.assert_awaited_once()
        mock_context.close.assert_awaited_once()
        mock_browser.close.assert_awaited_once()
        mock_pw.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_markdown_to_pdf_rejects_output_outside_outputs_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir(parents=True, exist_ok=True)

    escaped_path = tmp_path / "outside_dir" / "escaped.pdf"

    with pytest.raises(RuntimeError, match="must resolve beneath"):
        await markdown_to_pdf("# Test Resume", str(escaped_path))
