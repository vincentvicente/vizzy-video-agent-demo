"""
Fetch + extract content from a brand URL.

策略: GET + BeautifulSoup, 抽 title / meta description / h1 / main content / product images.
"""
from __future__ import annotations

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin


def fetch_page_content(url: str, timeout: int = 15) -> dict:
    """
    Fetch a brand/product page and extract content useful for Strategist.

    Uses a realistic browser fingerprint to get past basic anti-bot (Cloudflare 等).
    对于更严格的 anti-bot (Akamai, Datadome 等) 仍可能失败 — 那种情况要 headless browser.

    Returns:
        {
            "url": str,
            "title": str,
            "meta_description": str,
            "headings": list[str],
            "body_text": str,             # truncated to ~3000 chars
            "image_urls": list[str],      # up to 10 product images
        }
    """
    # Realistic Chrome on Mac fingerprint — covers ~80% of anti-bot checks
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/avif,image/webp,image/apng,*/*;q=0.8,"
            "application/signed-exchange;v=b3;q=0.7"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Sec-Ch-Ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"macOS"',
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
        "Connection": "keep-alive",
    }

    session = requests.Session()
    session.headers.update(headers)

    try:
        resp = session.get(url, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()
    except requests.HTTPError as e:
        # 给一个比 "403 Forbidden" 更有指导性的错误信息
        if e.response is not None and e.response.status_code in (403, 401, 429):
            raise RuntimeError(
                f"{url} blocked our request ({e.response.status_code}). "
                f"This site has aggressive anti-bot protection. "
                f"Try a different URL (a direct product page works best) or paste the brand text manually."
            ) from e
        raise

    soup = BeautifulSoup(resp.text, "html.parser")

    # title
    title = (soup.title.string or "").strip() if soup.title else ""

    # meta description
    meta = soup.find("meta", attrs={"name": "description"})
    meta_description = meta.get("content", "").strip() if meta else ""
    if not meta_description:
        og = soup.find("meta", attrs={"property": "og:description"})
        meta_description = og.get("content", "").strip() if og else ""

    # headings
    headings = []
    for tag in ["h1", "h2", "h3"]:
        for h in soup.find_all(tag):
            text = h.get_text(" ", strip=True)
            if text and len(text) < 200:
                headings.append(text)
            if len(headings) >= 15:
                break

    # main body text — strip scripts/styles/nav/footer
    for s in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        s.decompose()
    body_text = soup.get_text(" ", strip=True)
    if len(body_text) > 3000:
        body_text = body_text[:3000] + "…"

    # product images
    image_urls = []
    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src")
        if not src:
            continue
        # filter junk
        lower = src.lower()
        if any(x in lower for x in ("icon", "logo", "sprite", ".svg", "pixel", "tracking")):
            continue
        full = urljoin(url, src)
        if full not in image_urls:
            image_urls.append(full)
        if len(image_urls) >= 10:
            break

    return {
        "url": url,
        "title": title,
        "meta_description": meta_description,
        "headings": headings,
        "body_text": body_text,
        "image_urls": image_urls,
    }
