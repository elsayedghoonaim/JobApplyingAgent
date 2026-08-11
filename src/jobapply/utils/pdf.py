"""PDF generation using Playwright page.pdf()."""

import markdown
from playwright.async_api import async_playwright


async def markdown_to_pdf(md_content: str, output_path: str) -> str:
    """Convert Markdown → HTML → PDF using Playwright.

    Uses a headless Chromium page (separate from the Edge CDP connection)
    to render HTML and produce a clean, ATS-friendly PDF.

    Args:
        md_content: Markdown content to convert.
        output_path: Path where PDF should be saved.

    Returns:
        Path to the generated PDF file.
    """
    # Convert markdown to HTML
    html = markdown.markdown(md_content, extensions=["tables", "fenced_code"])
    
    # Add ATS-friendly styling
    styled_html = f"""
    <html><head><style>
        body {{ 
            font-family: Arial, sans-serif; 
            font-size: 11pt;
            line-height: 1.4; 
            margin: 0.75in; 
        }}
        h1 {{ 
            font-size: 16pt; 
            margin-bottom: 4pt; 
        }}
        h2 {{ 
            font-size: 13pt; 
            border-bottom: 1px solid #333;
            padding-bottom: 2pt; 
            margin-top: 12pt; 
        }}
        h3 {{
            font-size: 12pt;
            margin-top: 10pt;
            margin-bottom: 4pt;
        }}
        ul {{ 
            margin: 4pt 0; 
            padding-left: 20pt; 
        }}
        p {{
            margin: 4pt 0;
        }}
        strong {{
            font-weight: bold;
        }}
    </style></head><body>{html}</body></html>
    """
    
    # Generate PDF using Playwright
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.set_content(styled_html)
        await page.pdf(
            path=output_path, 
            format="Letter",
            print_background=False
        )
        await browser.close()
    
    return output_path
