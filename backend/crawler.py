"""
crawler.py - Dynamic web crawler with Playwright (JS rendering) + link graph traversal
"""
import asyncio
import os
import re
from urllib.parse import urljoin, urlparse, quote_plus
from typing import Callable, Optional
from bs4 import BeautifulSoup

try:
    from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

import httpx

HEADLESS = False
USE_EDGE = True

BLOCKED_DOMAINS = {
    "facebook.com", "twitter.com", "instagram.com", "youtube.com",
    "google.com", "googleapis.com", "gstatic.com", "doubleclick.net",
    "amazon.com", "apple.com", "microsoft.com"
}

# 검색 엔진 URL 패턴
SEARCH_ENGINES = [
    # "https://www.bing.com/search?q={query}",
    # "https://search.yahoo.com/search?p={query}",
    "https://duckduckgo.com/html/?q={query}",
]

# 불법 스트리밍 플레이어/CDN 도메인
STREAMING_PLAYERS = {
    # 주요 스트리밍 플레이어
    "doodstream.com", "dood.la", "dood.sh", "dood.ws",
    "streamtape.com", "streamtape.cc", "streamtape.net",
    "mixdrop.co", "mixdrop.sx", "mixdrop.to", "mixdrop.gl",
    "filemoon.to", "filemoon.sx", "filemoon.io",
    "vidplay.online", "vidplay.org", "vidplay.net",
    "streamwish.com", "streamwish.to", "streamwish.net",
    "streamsu.com", "streamsu.net",
    "vidoza.net", "vidoza.co",
    "upstream.to", "upstream.chat",
    "vtube.to", "vtube.io",
    "highload.to", "highload.sx",
    "voe.sx", "voe.tv", "voe.net",
    "vido.com", "vido.io",
    "trailer.to", "trailer365.com",
    "supervideo.it", "supervideo.cc",
    "vupload.io", "vupload.com",
    "streamhub.to", "streamhub.gg",
    "vidoflash.com", "vidoflash.net",
    "filelist.io", "filelist.ro",
    "rapidvideo.com", "rapidvideo.io",
    "oload.tv", "oload.stream",
    "streamango.com", "streamango.to",
    "streamcherry.com", "cherrystreaming.com",
    "yewtu.be", "yewtu.gg",  # YouTube 인스턴스
    # 한국 불법 사이트常见 도메인
    "tfreeca22.com", "tfreeca21.com", "tfreeca20.com",
    "aniplustv.com", "aniplus.co.kr",
    "dramafree.com", "drama-go.com",
    "moviefree7.com", "movie365.co.kr",
    "koreandrama.org", "kdramafree.com",
}


def is_valid_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        domain = parsed.netloc.lower()
        for blocked in BLOCKED_DOMAINS:
            if blocked in domain:
                return False
        # 파일 확장자 제외
        path = parsed.path.lower()
        skip_exts = ('.jpg', '.jpeg', '.png', '.gif', '.svg', '.ico', '.pdf',
                     '.zip', '.exe', '.mp4', '.mp3', '.css', '.js', '.woff')
        if any(path.endswith(ext) for ext in skip_exts):
            return False
        return True
    except:
        return False


def extract_text_and_links(html: str, base_url: str) -> tuple[str, str, list[str]]:
    """HTML에서 제목, 텍스트, 링크 추출"""
    soup = BeautifulSoup(html, "lxml")

    # 불필요한 태그 제거
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else "제목 없음"

    # 텍스트 추출
    text_parts = []
    for elem in soup.find_all(["p", "article", "section", "div", "span", "li", "td", "pre", "code"]):
        t = elem.get_text(separator=" ", strip=True)
        if len(t) > 30:
            text_parts.append(t)

    full_text = "\n".join(text_parts[:200])  # 최대 200개 블록

    # 링크 추출
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith("#") or href.startswith("javascript:") or href.startswith("mailto:"):
            continue
        full_url = urljoin(base_url, href)
        if is_valid_url(full_url):
            links.append(full_url)

    return title, full_text, list(set(links))


class PlaywrightCrawler:
    def __init__(self):
        self.browser = None
        self.playwright = None

    async def start(self):
        if not PLAYWRIGHT_AVAILABLE:
            return False
        try:
            self.playwright = await async_playwright().start()
            slow = 500 if not HEADLESS else 0
            channel = "msedge" if USE_EDGE else "chromium"
            self.browser = await self.playwright.chromium.launch(
                headless=HEADLESS,
                slow_mo=slow,
                channel=channel,
                args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"]
            )
            mode = "headless" if HEADLESS else f"headed ({channel} 브라우저 표시, slow_mo=500ms)"
            print(f"[Playwright] {mode} 모드로 실행")
            return True
        except Exception as e:
            print(f"[Playwright 시작 실패] {e}")
            return False
            slow = 500 if not HEADLESS else 0
            self.browser = await self.playwright.chromium.launch(
                headless=HEADLESS,
                slow_mo=slow,
                args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"]
            )

    async def stop(self):
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

    async def fetch_page(self, url: str, timeout: int = 15000) -> Optional[str]:
        if not self.browser:
            return None
        try:
            context = await self.browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                viewport={"width": 1280, "height": 720}
            )
            page = await context.new_page()

            # 이미지/폰트/미디어 차단 (속도 향상)
            await page.route("**/*.{png,jpg,jpeg,gif,svg,woff,woff2,mp4,mp3}", lambda r: r.abort())

            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=timeout)
                # JS 렌더링 대기
                await page.wait_for_load_state("networkidle", timeout=5000)
            except PlaywrightTimeout:
                pass  # 타임아웃이어도 현재 내용 사용

            html = await page.content()
            await context.close()
            return html
        except Exception as e:
            print(f"[Playwright 오류] {url}: {e}")
            return None


async def fetch_with_httpx(url: str) -> Optional[str]:
    """Playwright 없을 때 fallback"""
    try:
        async with httpx.AsyncClient(
            timeout=15,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        ) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                return resp.text
    except Exception as e:
        print(f"[httpx 오류] {url}: {e}")
    return None


async def search_keyword(keyword: str) -> list[str]:
    """키워드로 검색 엔진에서 URL 수집"""
    urls = []
    encoded = quote_plus(keyword)

    for engine_template in SEARCH_ENGINES[:2]:  # 2개 검색엔진만
        search_url = engine_template.format(query=encoded)
        try:
            html = await fetch_with_httpx(search_url)
            if not html:
                continue

            soup = BeautifulSoup(html, "lxml")
            for a in soup.find_all("a", href=True):
                href = a["href"]
                # 검색 결과 URL 추출
                if href.startswith("http") and is_valid_url(href):
                    # 검색엔진 자체 링크 제외
                    parsed = urlparse(href)
                    if not any(se in parsed.netloc for se in ["bing.com", "yahoo.com", "duckduckgo.com", "microsoft.com"]):
                        urls.append(href)

            # DuckDuckGo redirect 처리
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if "uddg=" in href:
                    match = re.search(r'uddg=([^&]+)', href)
                    if match:
                        from urllib.parse import unquote
                        real_url = unquote(match.group(1))
                        if is_valid_url(real_url):
                            urls.append(real_url)

        except Exception as e:
            print(f"[검색 오류] {keyword} @ {search_url}: {e}")

        await asyncio.sleep(1)  # 검색엔진 레이트 리밋

    return list(set(urls))[:10]  # 키워드당 최대 10개 URL


class LeakCrawler:
    def __init__(self, session_id: str, on_progress: Callable, max_depth: int = 2, max_pages: int = 5):
        self.session_id = session_id
        self.on_progress = on_progress
        self.playwright_crawler = PlaywrightCrawler()
        self.visited_urls: set = set()
        self.visited_domains: set = set()
        self.max_depth = max_depth
        self.max_pages_per_keyword = max_pages

    async def start(self):
        pw_ok = await self.playwright_crawler.start()
        if pw_ok:
            await self.on_progress("system", "✅ Playwright (JS 렌더링) 초기화 완료")
        else:
            await self.on_progress("system", "⚠️ Playwright 미사용, httpx fallback 모드")

    async def stop(self):
        await self.playwright_crawler.stop()

    async def fetch_page(self, url: str) -> Optional[str]:
        """Playwright 우선, fallback httpx"""
        html = await self.playwright_crawler.fetch_page(url)
        if not html:
            html = await fetch_with_httpx(url)
        return html

    async def crawl_url(self, url: str, keyword: str, depth: int = 0) -> list[dict]:
        """단일 URL 크롤링 + 링크 기반 확산"""
        if url in self.visited_urls or depth > self.max_depth:
            return []
        if not is_valid_url(url):
            return []

        self.visited_urls.add(url)
        results = []

        await self.on_progress("crawl", f"📄 크롤링 중 (depth={depth}): {url[:80]}")

        html = await self.fetch_page(url)
        if not html:
            return []

        title, text, links = extract_text_and_links(html, url)

        results.append({
            "url": url,
            "keyword": keyword,
            "title": title,
            "text": text,
            "links": links,
            "depth": depth
        })

        await self.on_progress("page", f"✅ 페이지 수집: {title[:50]} ({len(text)}자, 링크 {len(links)}개)")

        # 링크 기반 확산 (depth+1)
        if depth < self.max_depth:
            # 관련성 높은 링크 우선 선택
            relevant_links = self._filter_relevant_links(links, keyword)[:3]
            for link in relevant_links:
                if link not in self.visited_urls:
                    await asyncio.sleep(0.5)
                    child_results = await self.crawl_url(link, keyword, depth + 1)
                    results.extend(child_results)

        return results

    def _filter_relevant_links(self, links: list[str], keyword: str) -> list[str]:
        """키워드 관련성 기반 링크 필터링"""
        kw_tokens = keyword.lower().split()
        scored = []
        for link in links:
            score = 0
            url_lower = link.lower()
            for token in kw_tokens:
                if token in url_lower:
                    score += 2
            # 의심 패턴 가중치
            suspicious = ["leak", "dump", "data", "db", "pass", "crack", "hack", "breach", "expose", "secret"]
            for s in suspicious:
                if s in url_lower:
                    score += 1
            if score > 0:
                scored.append((score, link))

        scored.sort(reverse=True)
        # 점수 없어도 일부 포함
        result = [l for _, l in scored]
        non_scored = [l for l in links if l not in result]
        return result + non_scored[:2]

    async def crawl_keywords(self, keywords: list[str]) -> list[dict]:
        """키워드 목록으로 전체 크롤링"""
        all_results = []

        for keyword in keywords:
            await self.on_progress("keyword", f"🔍 키워드 검색 중: {keyword}")

            urls = await search_keyword(keyword)
            await self.on_progress("search", f"🌐 검색 결과 {len(urls)}개 URL 발견")

            page_count = 0
            for url in urls:
                if page_count >= self.max_pages_per_keyword:
                    break
                results = await self.crawl_url(url, keyword, depth=0)
                all_results.extend(results)
                page_count += len(results)
                await asyncio.sleep(0.3)

        return all_results
