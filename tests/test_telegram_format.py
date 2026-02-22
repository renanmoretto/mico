from __future__ import annotations

from mico.channels import _markdown_to_telegram_html


def test_markdown_to_telegram_html_formats_basic_styles() -> None:
    out = _markdown_to_telegram_html('**bold** and _italic_ and ~~gone~~')
    assert '<b>bold</b>' in out
    assert '<i>italic</i>' in out
    assert '<s>gone</s>' in out


def test_markdown_to_telegram_html_formats_links_and_code() -> None:
    out = _markdown_to_telegram_html('Use `x < y` and [site](https://example.com?a=1&b=2)')
    assert '<code>x &lt; y</code>' in out
    assert '<a href="https://example.com?a=1&amp;b=2">site</a>' in out


def test_markdown_to_telegram_html_formats_code_block() -> None:
    text = '```python\nprint(\"<tag>\")\n```'
    out = _markdown_to_telegram_html(text)
    assert '<pre><code>print("&lt;tag&gt;")\n</code></pre>' == out
