"""TBMM Milletvekili crawler — Bronze layer (Path B).

Pipeline:
  1. Discover 28th-term MP list → (mp_guid, name, party, province, donem_guid)
  2. For each MP, GET profile page → parse party + province + name + commissions
  3. Locate "Sahibi Olduğu Yazılı Soru Önergeleri" sub-page from profile
  4. Paginate that sub-page → collect every yazılı-soru detail GUID
  5. Write Bronze: mp_index.parquet + mp_to_guids.parquet + guids.txt

Respect: 1 req/sec rate limit, retry-after, single thread. Mirrors scraper.py.

Known unknown
-------------
The public MP-list page (``/milletvekili-arama/form``) is JS-rendered and
form-based. The legacy endpoint
``https://www5.tbmm.gov.tr/develop/owa/milletvekillerimiz_sd.dagilim`` returns
a static HTML table linking to per-MP detail pages, but its availability and
schema have flipped between TBMM releases. ``discover_mp_list`` performs a
best-effort fetch of that endpoint and parses any ``MilletvekiliDetay?Id=``
links it can find. When the endpoint is unreachable or empty, callers should
provide a manually-curated newline-separated MP GUID file via
``--mp-ids-from`` (see CLI). The downstream pipeline does not depend on
``discover_mp_list`` succeeding — only on having a list of MP GUIDs.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

BASE = "https://www.tbmm.gov.tr"
LEGACY_BASE = "https://www5.tbmm.gov.tr"
MP_PROFILE_TPL = f"{BASE}/milletvekili/MilletvekiliDetay?Id={{guid}}"
MP_PROFILE_WITH_DONEM_TPL = (
    f"{BASE}/milletvekili/MilletvekiliDetay?Id={{guid}}&DonemId={{donem}}"
)
LEGACY_DAGILIM = f"{LEGACY_BASE}/develop/owa/milletvekillerimiz_sd.dagilim"
LEGACY_TUM_UYELER = (
    f"{LEGACY_BASE}/develop/owa/milletvekillerimiz_sd.tum_uyeler_donem_no"
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept-Language": "tr-TR,tr;q=0.9,en;q=0.8",
}

RATE_LIMIT_SEC = 1.0

# GUID regex (uuid v4-ish, TBMM uses lowercase dashed form)
GUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)

# Recognised label fragments on the MP profile sub-menu
YAZILI_SORU_LABEL_RE = re.compile(
    r"Sahibi\s+Olduğu\s+Yazılı\s+Soru\s+Önergeleri", re.IGNORECASE
)
ONERGE_DETAY_PATH = "/Denetim/Yazili-Soru-Onergesi-Detay/"

logger = logging.getLogger("mp_crawler")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s :: %(message)s")
    )
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


@dataclass
class MPProfile:
    mp_guid: str
    name: str
    party: str
    province: str
    donem: str
    commissions: list[str] = field(default_factory=list)
    yazili_soru_listing_url: str | None = None


def _normalise_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _parse_party_province(header_text: str) -> tuple[str, str]:
    """Split fragments like "AK PARTİ İSTANBUL MİLLETVEKİLİ" → (party, province).

    Strategy: locate the literal " MİLLETVEKİLİ" suffix (case-insensitive,
    diacritic-tolerant), strip it, then assume the LAST whitespace-separated
    token of the remainder is the province (Turkish provinces are single
    words; multi-word names like ``KAHRAMANMARAŞ`` are still one token).
    """
    if not header_text:
        return "", ""
    t = _normalise_ws(header_text)
    # Case-insensitive split on the suffix
    m = re.search(r"\s*M[İI]LLETVEK[İI]L[İI]\s*", t, re.IGNORECASE)
    if m:
        t = t[: m.start()].rstrip()
    if not t:
        return "", ""
    if " " not in t:
        # Single token — can't separate
        return t, ""
    party, province = t.rsplit(" ", 1)
    return party.strip(), province.strip()


def _extract_guid_from_href(href: str) -> str | None:
    if not href:
        return None
    # Try query string first (Id= or id=)
    try:
        qs = parse_qs(urlparse(href).query)
        for key in ("Id", "id", "ID", "iD"):
            if key in qs and qs[key]:
                val = qs[key][0]
                if GUID_RE.fullmatch(val):
                    return val.lower()
    except Exception:
        pass
    # Fallback: any GUID anywhere in the href
    m = GUID_RE.search(href)
    return m.group(0).lower() if m else None


class MPCrawler:
    """Mirror of :class:`scraper.TBMMScraper` — same throttle/retry contract."""

    def __init__(self, out_dir: Path, sleep: float = RATE_LIMIT_SEC):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.sleep = sleep
        self._last_req = 0.0

    def _throttle(self) -> None:
        delta = time.monotonic() - self._last_req
        if delta < self.sleep:
            time.sleep(self.sleep - delta)
        self._last_req = time.monotonic()

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=60))
    def _get(self, url: str, **kw) -> requests.Response:
        self._throttle()
        r = self.session.get(url, timeout=30, **kw)
        if r.status_code == 429:
            retry_after = int(r.headers.get("Retry-After", 30))
            time.sleep(retry_after)
            raise requests.HTTPError(f"429 rate-limited; sleeping {retry_after}s")
        r.raise_for_status()
        return r

    # ------------------------------------------------------------------
    # 1) MP list discovery (best-effort; see module docstring)
    # ------------------------------------------------------------------
    def discover_via_alllist(self) -> list[dict]:
        """Primary path: fetch ``/milletvekili/AllList`` which returns a single
        HTML document grouping every current-term MP by province. Each list
        item carries a profile link of the form
        ``/milletvekili/milletvekilidetay?DonemId=<DG>&Id=<MG>`` plus party.
        Returns rows shaped like ``{mp_guid, name, party, province, donem_guid}``.
        """
        url = f"{BASE}/milletvekili/AllList"
        r = self._get(
            url,
            headers={
                "Accept": "text/html, */*; q=0.01",
                "X-Requested-With": "XMLHttpRequest",
                "Referer": f"{BASE}/milletvekili/liste",
            },
        )
        soup = BeautifulSoup(r.text, "lxml")
        rows: list[dict] = []
        current_province = ""
        for li in soup.select("li.list-group-item"):
            cls = " ".join(li.get("class", []))
            txt = _normalise_ws(li.get_text(" ", strip=True))
            if "active" in cls:
                current_province = txt
                continue
            a = li.find("a", href=re.compile(r"milletvekilidetay", re.I))
            if not a:
                continue
            href = a["href"]
            mid = re.search(r"(?<!Donem)[?&]Id=([0-9a-fA-F-]{36})", href, re.IGNORECASE)
            did = re.search(r"DonemId=([0-9a-fA-F-]{36})", href, re.IGNORECASE)
            if not mid:
                continue
            name = _normalise_ws(a.get_text())
            party_el = li.select_one(".col-md-4")
            party = _normalise_ws(party_el.get_text()) if party_el else ""
            rows.append({
                "mp_guid": mid.group(1).lower(),
                "name": name,
                "party": party,
                "province": current_province,
                "donem_guid": did.group(1).lower() if did else "",
            })
        return rows

    def discover_mp_list(self, donem: int = 28) -> list[dict]:
        """Best-effort MP enumeration. Tries /milletvekili/AllList first (works).

        Returns a list of ``{mp_guid, name, party, province, donem_guid}``
        dicts. On failure returns an empty list and logs the cause — the
        caller is expected to fall back to a manually supplied GUID file.
        """
        try:
            rows = self.discover_via_alllist()
            if rows:
                logger.info("MP list discovered via /AllList :: %d rows", len(rows))
                return rows
        except Exception as exc:  # noqa: BLE001
            logger.warning("/milletvekili/AllList failed: %s", exc)

        candidates: list[str] = [
            f"{LEGACY_DAGILIM}?p_donem={donem}",
            LEGACY_DAGILIM,
            f"{LEGACY_TUM_UYELER}?p_donem={donem}",
            LEGACY_TUM_UYELER,
        ]
        for url in candidates:
            try:
                r = self._get(url)
            except Exception as exc:  # noqa: BLE001
                logger.warning("MP-list candidate failed %s :: %s", url, exc)
                continue
            rows = self._parse_mp_listing_html(r.text, donem=donem)
            if rows:
                logger.info("MP-list discovered via %s :: %d rows", url, len(rows))
                return rows
            logger.info("MP-list candidate returned 0 rows: %s", url)
        logger.warning(
            "discover_mp_list exhausted all candidates for dönem=%s; "
            "supply --mp-ids-from <file> instead.",
            donem,
        )
        return []

    def _parse_mp_listing_html(self, html: str, donem: int) -> list[dict]:
        soup = BeautifulSoup(html, "lxml")
        rows: list[dict] = []
        seen: set[str] = set()
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "MilletvekiliDetay" not in href and "milletvekili" not in href.lower():
                continue
            guid = _extract_guid_from_href(href)
            if not guid or guid in seen:
                continue
            seen.add(guid)
            name = _normalise_ws(a.get_text())
            rows.append(
                {
                    "mp_guid": guid,
                    "name": name,
                    "party": "",
                    "province": "",
                    "donem_guid": str(donem),
                }
            )
        return rows

    # ------------------------------------------------------------------
    # 2) MP profile parse
    # ------------------------------------------------------------------
    def fetch_mp_profile(self, mp_guid: str) -> dict:
        url = MP_PROFILE_TPL.format(guid=mp_guid)
        r = self._get(url)
        soup = BeautifulSoup(r.text, "lxml")

        name, donem, party, province = self._parse_header_blob(soup)
        if not name:
            name = self._parse_name(soup)
        if not (party and province):
            party, province = self._parse_party_province_from_profile(soup)
        if not donem:
            donem = self._parse_donem(soup)
        commissions = self._parse_commissions(soup)
        listing_url = self._find_yazili_soru_listing_url(soup, base=url)

        profile = MPProfile(
            mp_guid=mp_guid.lower(),
            name=name,
            party=party,
            province=province,
            donem=donem,
            commissions=commissions,
            yazili_soru_listing_url=listing_url,
        )
        return asdict(profile)

    def _parse_name(self, soup: BeautifulSoup) -> str:
        # Try common containers: h1, h2, .baslik, page title
        for sel in ("h1", "h2", ".milletvekili-ad", ".baslik"):
            el = soup.select_one(sel)
            if el:
                txt = _normalise_ws(el.get_text())
                if txt and "TBMM" not in txt.upper():
                    return txt
        title = soup.find("title")
        if title:
            return _normalise_ws(title.get_text()).split("|")[0].strip()
        return ""

    def _parse_party_province_from_profile(
        self, soup: BeautifulSoup
    ) -> tuple[str, str]:
        # The profile typically prints a header line like
        # "AK PARTİ İSTANBUL MİLLETVEKİLİ" near the top.
        candidates: list[str] = []
        for sel in ("h2", "h3", ".parti", ".secim-cevre", ".unvan", "p", "div"):
            for el in soup.select(sel):
                txt = _normalise_ws(el.get_text())
                if "MİLLETVEKİLİ" in txt.upper() or "MILLETVEKILI" in txt.upper():
                    if len(txt) <= 120:
                        candidates.append(txt)
        for txt in candidates:
            party, province = _parse_party_province(txt)
            if party and province:
                return party, province
        if candidates:
            return _parse_party_province(candidates[0])
        return "", ""

    def _parse_commissions(self, soup: BeautifulSoup) -> list[str]:
        commissions: list[str] = []
        # Look for a heading that mentions "Komisyon" and grab the following list
        for header in soup.find_all(["h2", "h3", "h4", "strong"]):
            htxt = _normalise_ws(header.get_text())
            if "Komisyon" in htxt:
                # Walk forward through siblings until next header
                for sib in header.find_all_next():
                    if sib.name in ("h1", "h2", "h3", "h4"):
                        break
                    if sib.name == "li":
                        ctxt = _normalise_ws(sib.get_text())
                        if ctxt and ctxt not in commissions:
                            commissions.append(ctxt)
                if commissions:
                    break
        return commissions

    def _parse_donem(self, soup: BeautifulSoup) -> str:
        text = soup.get_text(" ", strip=True)
        m = re.search(r"(\d{1,2})\.\s*DÖNEM", text, re.IGNORECASE)
        return m.group(1) if m else ""

    # MP header appears as flat run of text in the form:
    #   "<Ad SOYAD> <N>. DÖNEM <PARTY> <PROVINCE> MİLLETVEKİLİ"
    # Page has no h-tags so we anchor on " MİLLETVEKİLİ" and walk backwards.
    _MILLETVEKILI_RE = re.compile(r"MİLLETVEKİLİ", re.IGNORECASE)
    _DONEM_RE = re.compile(r"(\d{1,2})\.\s*DÖNEM", re.IGNORECASE)

    def _parse_header_blob(self, soup: BeautifulSoup) -> tuple[str, str, str, str]:
        text = soup.get_text(" ", strip=True)
        # Find every " MİLLETVEKİLİ" occurrence and pick the one whose preceding
        # 200-char window contains a "<N>. DÖNEM" pattern with a short prefix —
        # this skips bio prose that mentions parliamentary terms.
        best: tuple[str, str, str, str] | None = None
        best_len = 10_000
        for m in self._MILLETVEKILI_RE.finditer(text):
            end = m.start()
            window = text[max(0, end - 200):end].rstrip()
            d = list(self._DONEM_RE.finditer(window))
            if not d:
                continue
            dm = d[-1]
            donem = dm.group(1)
            after_donem = window[dm.end():].strip()  # party + province
            before_donem = window[:dm.start()].strip()  # name (last 2-3 words)
            # Province = last UPPERCASE token before MİLLETVEKİLİ
            tokens = after_donem.split()
            if not tokens:
                continue
            province = tokens[-1]
            party = " ".join(tokens[:-1])
            # Name = trailing run of name-shaped tokens before DÖNEM. Surname
            # is uppercase (≥3 chars); first/middle names are capitalised but
            # may contain lowercase letters. Walk back from the end and stop at
            # the first menu/noise word (e.g. "Konuşmaları", "Önergeleri").
            name_tokens = before_donem.split()
            picked: list[str] = []
            for tok in reversed(name_tokens):
                if tok.lower().endswith(("ları", "leri", "lar", "ler", "esi", "asi")):
                    break
                if not tok or not tok[0].isalpha():
                    break
                if not tok[0].isupper():
                    break
                picked.append(tok)
                if len(picked) >= 4:
                    break
            name = " ".join(reversed(picked))
            candidate = (name, donem, party, province)
            score = len(party)  # short party preferred (real parties are <=20 chars)
            if score < best_len:
                best_len = score
                best = candidate
        return best or ("", "", "", "")

    def _find_yazili_soru_listing_url(
        self, soup: BeautifulSoup, base: str
    ) -> str | None:
        for a in soup.find_all("a", href=True):
            label = _normalise_ws(a.get_text())
            if YAZILI_SORU_LABEL_RE.search(label):
                return urljoin(base, a["href"])
        # Fallback: any href containing the yazili-soru listing fragment
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "Yazili-Soru" in href and "Detay" not in href:
                return urljoin(base, href)
        return None

    # ------------------------------------------------------------------
    # 3) Yazılı soru listing — collect detail GUIDs, paginate
    # ------------------------------------------------------------------
    def fetch_yazili_soru_guids(self, listing_url: str) -> list[str]:
        if not listing_url:
            return []
        seen: set[str] = set()
        order: list[str] = []
        visited_pages: set[str] = set()
        next_url: str | None = listing_url
        max_pages = 200  # safety cap

        for _ in range(max_pages):
            if next_url is None or next_url in visited_pages:
                break
            visited_pages.add(next_url)
            try:
                r = self._get(next_url)
            except Exception as exc:  # noqa: BLE001
                logger.warning("listing page fetch failed %s :: %s", next_url, exc)
                break
            soup = BeautifulSoup(r.text, "lxml")
            page_guids = self._extract_onerge_guids(soup)
            new_count = 0
            for g in page_guids:
                if g not in seen:
                    seen.add(g)
                    order.append(g)
                    new_count += 1
            next_url = self._find_next_page(soup, current=next_url)
            if new_count == 0 and next_url is None:
                break
        return order

    def _extract_onerge_guids(self, soup: BeautifulSoup) -> list[str]:
        guids: list[str] = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if ONERGE_DETAY_PATH.lower() not in href.lower():
                continue
            m = GUID_RE.search(href)
            if m:
                guids.append(m.group(0).lower())
        # Also catch GUIDs embedded in inline text/scripts
        text_block = soup.get_text(" ", strip=False)
        for m in GUID_RE.finditer(text_block):
            # Only add if it's plausibly linked to a detail URL on this page
            pass
        return guids

    def _find_next_page(self, soup: BeautifulSoup, current: str) -> str | None:
        # Common patterns: rel="next", text "Sonraki" / "İleri" / "»"
        link = soup.find("a", attrs={"rel": "next"}, href=True)
        if link:
            return urljoin(current, link["href"])
        for a in soup.find_all("a", href=True):
            label = _normalise_ws(a.get_text()).lower()
            if label in {"sonraki", "ileri", "»", ">", "next"}:
                return urljoin(current, a["href"])
            if "sonraki" in label and len(label) < 20:
                return urljoin(current, a["href"])
        return None

    # ------------------------------------------------------------------
    # 4) Orchestration
    # ------------------------------------------------------------------
    def run(self, mp_ids: list[str], out_dir: Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        index_rows: list[dict] = []
        mapping_rows: list[dict] = []
        all_guids: set[str] = set()

        for mp_guid in tqdm(mp_ids, desc="MPs"):
            mp_guid = mp_guid.strip().lower()
            if not mp_guid:
                continue
            try:
                profile = self.fetch_mp_profile(mp_guid)
            except Exception as exc:  # noqa: BLE001
                logger.error("profile fetch failed %s :: %s", mp_guid, exc)
                index_rows.append(
                    {
                        "mp_guid": mp_guid,
                        "name": "",
                        "party": "",
                        "province": "",
                        "donem": "",
                        "n_yazili_soru": 0,
                        "error": str(exc),
                    }
                )
                continue

            listing_url = profile.get("yazili_soru_listing_url")
            guids: list[str] = []
            if listing_url:
                try:
                    guids = self.fetch_yazili_soru_guids(listing_url)
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "yazili-soru listing failed mp=%s :: %s", mp_guid, exc
                    )
            else:
                logger.info("no yazili-soru listing for mp=%s", mp_guid)

            for g in guids:
                all_guids.add(g)
                mapping_rows.append({"mp_guid": mp_guid, "onerge_guid": g})

            index_rows.append(
                {
                    "mp_guid": mp_guid,
                    "name": profile.get("name", ""),
                    "party": profile.get("party", ""),
                    "province": profile.get("province", ""),
                    "donem": profile.get("donem", ""),
                    "n_yazili_soru": len(guids),
                    "error": "",
                }
            )

            # Incremental persistence — never lose progress
            self._write_outputs(out, index_rows, mapping_rows, all_guids)

        logger.info(
            "run complete :: %d MPs, %d unique önerge GUIDs",
            len(index_rows),
            len(all_guids),
        )

    def _write_outputs(
        self,
        out: Path,
        index_rows: list[dict],
        mapping_rows: list[dict],
        all_guids: Iterable[str],
    ) -> None:
        try:
            pd.DataFrame(index_rows).to_parquet(
                out / "mp_index.parquet", index=False
            )
            pd.DataFrame(mapping_rows).to_parquet(
                out / "mp_to_guids.parquet", index=False
            )
            (out / "guids.txt").write_text(
                "\n".join(sorted(set(all_guids))) + "\n", encoding="utf-8"
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("write outputs failed :: %s", exc)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _read_id_file(path: Path) -> list[str]:
    return [
        line.strip().lower()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="TBMM MP crawler (Bronze layer)")
    ap.add_argument("--out", default="data/bronze", type=Path)
    ap.add_argument(
        "--donem", default=28, type=int, help="Yasama dönemi (default: 28)"
    )
    ap.add_argument(
        "--discover",
        action="store_true",
        help="Enumerate MPs from legacy TBMM listing endpoints",
    )
    ap.add_argument(
        "--mp-ids-from",
        type=Path,
        help="Newline-separated file of MP GUIDs to crawl",
    )
    ap.add_argument(
        "--test", metavar="MP_GUID", help="Single-profile dry run; prints parsed dict"
    )
    args = ap.parse_args()

    crawler = MPCrawler(out_dir=args.out)

    if args.test:
        profile = crawler.fetch_mp_profile(args.test)
        print(json.dumps(profile, indent=2, ensure_ascii=False))
        return

    mp_ids: list[str] = []
    if args.mp_ids_from:
        mp_ids = _read_id_file(args.mp_ids_from)
        logger.info("loaded %d MP GUIDs from %s", len(mp_ids), args.mp_ids_from)
    elif args.discover:
        discovered = crawler.discover_mp_list(donem=args.donem)
        mp_ids = [row["mp_guid"] for row in discovered]
        if not mp_ids:
            logger.error(
                "discover_mp_list returned 0 rows; rerun with --mp-ids-from"
            )
            return
    else:
        ap.print_help()
        return

    crawler.run(mp_ids=mp_ids, out_dir=args.out)


if __name__ == "__main__":
    main()
