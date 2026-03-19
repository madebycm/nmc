#!/usr/bin/env python3
from __future__ import annotations

import json
import posixpath
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape, unescape
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.request import Request, urlopen

BASE_URL = "https://app.ainm.no"
START_ROUTE = "/docs"
OUTPUT_DIR = Path("docs")
CSS_PATH = OUTPUT_DIR / "_assets" / "docs.css"
TEMPLATE_CSS_PATH = Path(__file__).resolve().parent / "templates" / "docs.css"
USER_AGENT = "nmcx-docs-sync/1.0"
TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class NavItem:
    section: str
    label: str
    route: str


@dataclass
class Page:
    item: NavItem
    title: str
    category: str
    article_html: str
    headings: list[dict[str, str | int]]

    @property
    def source_url(self) -> str:
        return urljoin(BASE_URL, self.item.route)

    @property
    def output_path(self) -> Path:
        if self.item.route == START_ROUTE:
            return OUTPUT_DIR / "index.html"
        return OUTPUT_DIR / self.item.route.removeprefix(f"{START_ROUTE}/") / "index.html"


def fetch(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return response.read()


def fetch_text(route_or_url: str) -> str:
    url = route_or_url if route_or_url.startswith("http") else urljoin(BASE_URL, route_or_url)
    return fetch(url).decode("utf-8", "ignore")


def load_template_css() -> str:
    return TEMPLATE_CSS_PATH.read_text(encoding="utf-8").rstrip() + "\n"


def strip_tags(fragment: str) -> str:
    text = re.sub(r"<[^>]+>", "", fragment)
    return " ".join(unescape(text).split())


def extract_required(pattern: str, text: str, label: str) -> str:
    match = re.search(pattern, text, flags=re.S)
    if not match:
        raise ValueError(f"Could not extract {label}")
    return match.group(1)


def parse_nav(start_html: str) -> list[NavItem]:
    nav_html = extract_required(r'<nav class="space-y-5">(.*?)</nav>', start_html, "sidebar navigation")
    items: list[NavItem] = []
    seen: set[str] = set()
    section_pattern = re.compile(r"<div><h4[^>]*>(.*?)</h4><ul[^>]*>(.*?)</ul></div>", re.S)
    link_pattern = re.compile(r'<a [^>]*href="(/docs(?:/[^"]*)?)"[^>]*>(.*?)</a>', re.S)

    for section_match in section_pattern.finditer(nav_html):
        section = strip_tags(section_match.group(1))
        for link_match in link_pattern.finditer(section_match.group(2)):
            route = link_match.group(1)
            if route in seen:
                continue
            seen.add(route)
            items.append(NavItem(section=section, label=strip_tags(link_match.group(2)), route=route))

    if not items:
        raise ValueError("No docs routes found in sidebar navigation")
    return items


def parse_headings(article_html: str) -> list[dict[str, str | int]]:
    headings: list[dict[str, str | int]] = []
    for level, heading_id, inner_html in re.findall(r"<h([1-3]) id=\"([^\"]+)\">(.*?)</h\1>", article_html, re.S):
        headings.append(
            {
                "id": heading_id,
                "level": int(level),
                "title": strip_tags(inner_html),
            }
        )
    return headings


def local_page_path(route: str) -> Path:
    if route == START_ROUTE:
        return OUTPUT_DIR / "index.html"
    return OUTPUT_DIR / route.removeprefix(f"{START_ROUTE}/") / "index.html"


def local_asset_path(doc_path: str) -> Path:
    return OUTPUT_DIR / doc_path.removeprefix(f"{START_ROUTE}/")


def relative_link(from_path: Path, to_path: Path, fragment: str = "") -> str:
    rel = posixpath.relpath(to_path.as_posix(), start=from_path.parent.as_posix())
    return f"{rel}#{fragment}" if fragment else rel


def rewrite_html(fragment: str, current_route: str, doc_routes: set[str], asset_routes: set[str]) -> str:
    current_path = local_page_path(current_route)

    def replace_attr(match: re.Match[str]) -> str:
        attr = match.group(1)
        raw_url = match.group(2)
        parsed = urlparse(raw_url)

        if not raw_url or raw_url.startswith("#") or parsed.scheme in {"http", "https", "mailto"}:
            return match.group(0)

        if parsed.path in doc_routes:
            target = relative_link(current_path, local_page_path(parsed.path), parsed.fragment)
            return f'{attr}="{escape(target, quote=True)}"'

        if parsed.path.startswith(START_ROUTE + "/") and parsed.path in asset_routes:
            target = relative_link(current_path, local_asset_path(parsed.path), parsed.fragment)
            return f'{attr}="{escape(target, quote=True)}"'

        if raw_url.startswith("/"):
            absolute = urlunparse(parsed._replace(scheme="https", netloc="app.ainm.no"))
            return f'{attr}="{escape(absolute, quote=True)}"'

        return match.group(0)

    return re.sub(r'(href|src)="([^"]+)"', replace_attr, fragment)


def download_asset(doc_path: str) -> None:
    target = local_asset_path(doc_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(fetch(urljoin(BASE_URL, doc_path)))


def render_nav(current_route: str, nav_items: list[NavItem]) -> str:
    chunks: list[str] = []
    current_section = None

    for item in nav_items:
        if item.section != current_section:
            if current_section is not None:
                chunks.append("</ul>")
            current_section = item.section
            chunks.append(f'<div class="section-title">{escape(item.section)}</div><ul class="nav-list">')
        href = relative_link(local_page_path(current_route), local_page_path(item.route))
        active = " active" if item.route == current_route else ""
        chunks.append(f'<li><a class="nav-link{active}" href="{escape(href, quote=True)}">{escape(item.label)}</a></li>')

    if current_section is not None:
        chunks.append("</ul>")

    return "".join(chunks)


def render_toc(headings: Iterable[dict[str, str | int]]) -> str:
    entries = list(headings)
    if not entries:
        return '<p class="meta">No headings found on this page.</p>'

    items = []
    for heading in entries:
        indent = 0 if heading["level"] == 2 else 12 if heading["level"] == 3 else 0
        items.append(
            f'<li><a class="toc-link" style="padding-left: {10 + indent}px" href="#{escape(str(heading["id"]))}">{escape(str(heading["title"]))}</a></li>'
        )
    return f'<ul class="toc-list">{"".join(items)}</ul>'


def render_pager(index: int, pages: list[Page]) -> str:
    prev_link = ""
    next_link = ""

    if index > 0:
        prev = pages[index - 1]
        href = relative_link(pages[index].output_path, prev.output_path)
        prev_link = f'<a href="{escape(href, quote=True)}"><strong>Previous</strong><br>{escape(prev.item.label)}</a>'
    else:
        prev_link = "<span></span>"

    if index + 1 < len(pages):
        nxt = pages[index + 1]
        href = relative_link(pages[index].output_path, nxt.output_path)
        next_link = f'<a class="next" href="{escape(href, quote=True)}"><strong>Next</strong><br>{escape(nxt.item.label)}</a>'
    else:
        next_link = "<span></span>"

    return f'<div class="pager">{prev_link}{next_link}</div>'


def render_page(page: Page, nav_items: list[NavItem], pages: list[Page], generated_at: str) -> str:
    css_href = relative_link(page.output_path, CSS_PATH)
    page_index = next(index for index, candidate in enumerate(pages) if candidate.item.route == page.item.route)
    pager = render_pager(page_index, pages)
    toc = render_toc([heading for heading in page.headings if int(heading["level"]) >= 2])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(page.title)} | NM i AI Docs Mirror</title>
  <link rel="stylesheet" href="{escape(css_href, quote=True)}">
</head>
<body>
  <div class="layout">
    <div class="topbar panel">
      <h1>NM i AI Docs Mirror</h1>
      <p>Local snapshot of <a href="{escape(page.source_url, quote=True)}">{escape(page.source_url)}</a> generated {escape(generated_at)}.</p>
    </div>
    <div class="shell">
      <aside class="sidebar panel">
        {render_nav(page.item.route, nav_items)}
      </aside>
      <main class="content panel">
        <div class="eyebrow"><span>{escape(page.category)}</span><span>/</span><span>{escape(page.item.label)}</span></div>
        <div class="meta">Source route: <a href="{escape(page.source_url, quote=True)}">{escape(page.item.route)}</a></div>
        <article class="prose">
          {page.article_html}
        </article>
        {pager}
      </main>
      <aside class="toc panel">
        <div class="section-title">On This Page</div>
        {toc}
      </aside>
    </div>
  </div>
</body>
</html>
"""


def render_readme(nav_items: list[NavItem], pages: list[Page], generated_at: str) -> str:
    lines = [
        "# Docs Mirror",
        "",
        f"- Source: `{BASE_URL}{START_ROUTE}`",
        f"- Generated: `{generated_at}`",
        f"- Pages: `{len(pages)}`",
        "",
        "Refresh with:",
        "",
        "```bash",
        "python3 scripts/sync_docs.py",
        "```",
        "",
        "## Pages",
        "",
    ]

    current_section = None
    page_map = {page.item.route: page for page in pages}
    for item in nav_items:
        if item.section != current_section:
            current_section = item.section
            lines.extend([f"### {item.section}", ""])
        page = page_map[item.route]
        local_path = page.output_path.relative_to(OUTPUT_DIR)
        lines.append(f"- `{item.label}`: `{local_path.as_posix()}`")
    lines.append("")
    return "\n".join(lines)


def validate_outputs(pages: list[Page], asset_paths: set[Path]) -> None:
    missing = [page.output_path for page in pages if not page.output_path.exists()]
    missing.extend(path for path in asset_paths if not path.exists())
    if missing:
        raise FileNotFoundError(f"Missing generated files: {missing}")

    for page in pages:
        html = page.output_path.read_text(encoding="utf-8")
        for relative in re.findall(r'(?:href|src)="([^"]+)"', html):
            if relative.startswith(("http://", "https://", "mailto:", "#")):
                continue
            target = (page.output_path.parent / relative).resolve()
            if not target.exists():
                raise FileNotFoundError(f"Broken link in {page.output_path}: {relative}")


def main() -> int:
    start_html = fetch_text(START_ROUTE)
    nav_items = parse_nav(start_html)
    doc_routes = {item.route for item in nav_items}

    raw_pages: list[Page] = []
    asset_routes: set[str] = set()

    for item in nav_items:
        html = fetch_text(item.route)
        article_html = extract_required(r'<article class="prose max-w-3xl">(.*?)</article>', html, f"article for {item.route}")
        category = extract_required(
            r'<div class="mb-6 flex items-center justify-between max-w-3xl"><span class="text-sm text-muted-foreground">(.*?)</span>',
            html,
            f"category for {item.route}",
        )
        title_match = re.search(r"<h1 id=\"[^\"]+\">(.*?)</h1>", article_html, re.S)
        title = strip_tags(title_match.group(1)) if title_match else item.label

        for path in re.findall(r'(?:href|src)="(/docs/[^"#?]+(?:\?[^"#]*)?)"', article_html):
            parsed_path = urlparse(path).path
            if parsed_path not in doc_routes:
                asset_routes.add(parsed_path)

        raw_pages.append(
            Page(
                item=item,
                title=title,
                category=strip_tags(category),
                article_html=article_html,
                headings=parse_headings(article_html),
            )
        )

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    CSS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CSS_PATH.write_text(load_template_css(), encoding="utf-8")

    for asset_route in sorted(asset_routes):
        download_asset(asset_route)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    pages: list[Page] = []

    for page in raw_pages:
        rewritten_article = rewrite_html(page.article_html, page.item.route, doc_routes, asset_routes)
        final_page = Page(
            item=page.item,
            title=page.title,
            category=page.category,
            article_html=rewritten_article,
            headings=page.headings,
        )
        final_page.output_path.parent.mkdir(parents=True, exist_ok=True)
        final_page.output_path.write_text(render_page(final_page, nav_items, raw_pages, generated_at), encoding="utf-8")
        pages.append(final_page)

    manifest = {
        "generated_at": generated_at,
        "source": f"{BASE_URL}{START_ROUTE}",
        "pages": [
            {
                "section": page.item.section,
                "label": page.item.label,
                "route": page.item.route,
                "title": page.title,
                "source_url": page.source_url,
                "output_path": page.output_path.relative_to(OUTPUT_DIR).as_posix(),
                "headings": page.headings,
            }
            for page in pages
        ],
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "README.md").write_text(render_readme(nav_items, pages, generated_at), encoding="utf-8")

    validate_outputs(pages, {local_asset_path(route) for route in asset_routes} | {CSS_PATH})

    print(f"Generated {len(pages)} pages and {len(asset_routes)} assets in {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (HTTPError, URLError, ValueError, FileNotFoundError) as exc:
        print(f"docs sync failed: {exc}")
        raise SystemExit(1)
