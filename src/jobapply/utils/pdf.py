"""Hardened PDF generation using sanitized HTML and restricted Playwright page.pdf()."""

from pathlib import Path

import markdown
import nh3
from playwright.async_api import async_playwright

from jobapply.utils.paths import get_outputs_root

# Strict allowlist of HTML tags necessary for ATS resumes
ALLOWED_TAGS = {
    "a",
    "b",
    "blockquote",
    "br",
    "code",
    "div",
    "em",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "hr",
    "i",
    "li",
    "ol",
    "p",
    "pre",
    "span",
    "strong",
    "table",
    "tbody",
    "td",
    "th",
    "thead",
    "tr",
    "ul",
}

ALLOWED_ATTRIBUTES = {
    "a": {"href"},
}

ALLOWED_URL_SCHEMES = {"http", "https", "mailto"}

ATS_CSS = """
    body {
        font-family: Arial, sans-serif;
        font-size: 11pt;
        line-height: 1.4;
        margin: 0.75in;
    }
    h1 {
        font-size: 16pt;
        margin-bottom: 4pt;
    }
    h2 {
        font-size: 13pt;
        border-bottom: 1px solid #333;
        padding-bottom: 2pt;
        margin-top: 12pt;
    }
    h3 {
        font-size: 12pt;
        margin-top: 10pt;
        margin-bottom: 4pt;
    }
    ul, ol {
        margin: 4pt 0;
        padding-left: 20pt;
    }
    p {
        margin: 4pt 0;
    }
    strong, b {
        font-weight: bold;
    }
    em, i {
        font-style: italic;
    }
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 8pt 0;
    }
    th, td {
        border: 1px solid #ddd;
        padding: 4pt 8pt;
        text-align: left;
    }
    th {
        background-color: #f5f5f5;
        font-weight: bold;
    }
    code, pre {
        font-family: monospace;
        font-size: 10pt;
        background-color: #f5f5f5;
        padding: 2pt 4pt;
    }
"""

CSP_POLICY = (
    "default-src 'none'; "
    "script-src 'none'; "
    "object-src 'none'; "
    "frame-src 'none'; "
    "base-uri 'none'; "
    "form-action 'none'; "
    "connect-src 'none'; "
    "font-src 'none'; "
    "img-src 'none'; "
    "style-src 'unsafe-inline';"
)


def sanitize_markdown_html(md_content: str) -> str:
    """Convert Markdown to sanitized HTML with strict tag/attribute allowlisting."""
    raw_html = markdown.markdown(md_content, extensions=["tables", "fenced_code"])
    return nh3.clean(
        raw_html,
        tags=ALLOWED_TAGS,
        attributes=ALLOWED_ATTRIBUTES,
        url_schemes=ALLOWED_URL_SCHEMES,
        strip_comments=True,
    )


def build_hardened_document(clean_html: str) -> str:
    """Wrap sanitized HTML in a hardened HTML envelope with restrictive CSP and safe ATS styling."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta http-equiv="Content-Security-Policy" content="{CSP_POLICY}">
    <title>Resume</title>
    <style>{ATS_CSS}</style>
</head>
<body>{clean_html}</body>
</html>"""


async def _handle_route_abort(route) -> None:
    """Explicit async abort handler for all external / file / data requests."""
    await route.abort()


async def markdown_to_pdf(md_content: str, output_path: str) -> str:
    """Convert Markdown -> Sanitized HTML -> PDF using hardened Playwright.

    Security measures:
    - HTML sanitization with nh3 (strips scripts, styles, objects, embeds, iframes, images, event handlers).
    - Restrictive Content-Security-Policy (CSP) blocking external requests.
    - Playwright context configured with java_script_enabled=False.
    - All network/file route requests are intercepted and aborted via explicit async handler.
    - Output path validated to resolve under the canonical outputs root.
    - Guaranteed cleanup of browser, context, page, and playwright without masking original errors.

    Args:
        md_content: Untrusted markdown content to convert.
        output_path: Path where PDF should be saved.

    Returns:
        Path to the generated PDF file.
    """
    clean_html = sanitize_markdown_html(md_content)
    document_html = build_hardened_document(clean_html)

    out_file = Path(output_path).resolve()
    outputs_root = get_outputs_root()

    # Validate output path is strictly contained within outputs root
    try:
        out_file.relative_to(outputs_root)
    except ValueError:
        raise RuntimeError(f"Output PDF path '{out_file}' must resolve beneath '{outputs_root}'")

    out_file.parent.mkdir(parents=True, exist_ok=True)

    pw = None
    browser = None
    context = None
    page = None

    try:
        pw = await async_playwright().start()
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(
            java_script_enabled=False,
            bypass_csp=False,
        )

        # Abort all network / file / data navigation and subresource requests
        await context.route("**/*", _handle_route_abort)

        page = await context.new_page()
        await page.set_content(document_html, wait_until="load")
        await page.pdf(path=str(out_file), format="Letter", print_background=False)
    finally:
        if page is not None:
            try:
                await page.close()
            except Exception:
                pass
        if context is not None:
            try:
                await context.close()
            except Exception:
                pass
        if browser is not None:
            try:
                await browser.close()
            except Exception:
                pass
        if pw is not None:
            try:
                await pw.stop()
            except Exception:
                pass

    return str(out_file)
