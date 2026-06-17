"""TBMM Yazılı Soru Önergesi scraper — Bronze layer (refactored for bulk).

Two-phase pipeline:
  Phase A — fetch detail HTML, parse structured metadata, write incremental
            Parquet shards (every ``BATCH_SIZE`` rows). Resumes from
            ``done_meta.txt``.
  Phase B — read metadata Parquet, download önerge + cevap PDFs in a
            ThreadPool against ``cdn.tbmm.gov.tr`` (separate host, CORS
            open, supports concurrent connections). Resumes by checking
            file existence.

Two rate limiters: one per HTML host (1 req/s) and one per CDN host
(8 concurrent + 0.2s gap) — they don't share quota.

CLI:
  python src/scraper.py --test                              # single GUID smoke
  python src/scraper.py --phase a --guids guids.txt         # bulk metadata
  python src/scraper.py --phase b --meta meta.parquet       # bulk PDF download
  python src/scraper.py --phase all --guids guids.txt       # A then B
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

BASE = "https://www.tbmm.gov.tr"
DETAIL_TPL = f"{BASE}/Denetim/Yazili-Soru-Onergesi-Detay/{{guid}}"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept-Language": "tr-TR,tr;q=0.9,en;q=0.8",
}

HTML_RATE_SEC = 1.5   # TBMM main host — moderate (IP clean, small remaining batch)
CDN_GAP_SEC = 0.2     # CDN — relaxed
CDN_WORKERS = 8
BATCH_SIZE = 500

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)
logger = logging.getLogger("scraper")


@dataclass
class Onerge:
    guid: str
    donem: str
    yasama_yili: str
    esas_no: str
    geliş_tarihi: str
    özet: str
    sahip_il: str
    sahip_isim: str
    muhatap_bakanlık: str
    muhatap_bakan: str
    durum: str
    onerge_pdf_url: str | None
    cevap_pdf_url: str | None
    detail_html_hash: str


# --- shared per-host throttle helpers ---------------------------------

class HostThrottle:
    """Thread-safe minimum gap between requests for a given host."""

    def __init__(self, min_gap: float) -> None:
        self.min_gap = min_gap
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delta = now - self._last
            if delta < self.min_gap:
                time.sleep(self.min_gap - delta)
            self._last = time.monotonic()


# --- core scraper -----------------------------------------------------

class TBMMScraper:
    def __init__(self, out_dir: Path) -> None:
        self.out = Path(out_dir)
        self.pdf_dir = self.out / "pdf"
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir = self.out / "yazili_soru_meta_shards"
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.done_meta_file = self.out / "done_meta.txt"
        self.errors_file = self.out / "scraper_errors.parquet"

        self.html_session = requests.Session()
        self.html_session.headers.update(HEADERS)
        self.html_throttle = HostThrottle(HTML_RATE_SEC)

        self.cdn_session = requests.Session()
        self.cdn_session.headers.update(HEADERS)
        self.cdn_throttle = HostThrottle(CDN_GAP_SEC)

    # ----- HTTP -------------------------------------------------------

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=60))
    def _get_html(self, url: str) -> requests.Response:
        self.html_throttle.wait()
        r = self.html_session.get(url, timeout=30)
        if r.status_code == 429:
            retry_after = int(r.headers.get("Retry-After", 30))
            time.sleep(retry_after)
            raise requests.HTTPError(f"429 rate-limited; sleeping {retry_after}s")
        r.raise_for_status()
        return r

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=30))
    def _get_pdf(self, url: str) -> requests.Response:
        self.cdn_throttle.wait()
        r = self.cdn_session.get(url, timeout=60)
        if r.status_code == 429:
            retry_after = int(r.headers.get("Retry-After", 30))
            time.sleep(retry_after)
            raise requests.HTTPError(f"429 rate-limited; sleeping {retry_after}s")
        r.raise_for_status()
        return r

    # ----- HTML parse -------------------------------------------------

    def parse_detail(self, guid: str) -> Onerge:
        url = DETAIL_TPL.format(guid=guid)
        r = self._get_html(url)
        html = r.text
        h = hashlib.sha256(html.encode()).hexdigest()[:16]
        soup = BeautifulSoup(html, "lxml")

        fields: dict[str, str] = {}
        for row in soup.select("tr"):
            cells = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
            if len(cells) == 2:
                fields[cells[0]] = cells[1]

        def f(key: str) -> str:
            return fields.get(key, "").strip()

        sahip_raw = f("Önergenin Sahibi")
        sahip_il, sahip_isim = "", sahip_raw
        if " Milletvekili " in sahip_raw:
            sahip_il, sahip_isim = sahip_raw.split(" Milletvekili ", 1)

        muh_raw = f("Önergenin Muhatabı")
        muh_bak, muh_isim = muh_raw, ""
        if " Bakanı " in muh_raw:
            muh_bak, muh_isim = muh_raw.split(" Bakanı ", 1)
            muh_bak = muh_bak + " Bakanlığı"

        pdf_urls = [
            urljoin(BASE, a["href"])
            for a in soup.select("a[href$='.pdf'], a[href*='.pdf?']")
        ]
        onerge_pdf = pdf_urls[0] if pdf_urls else None
        cevap_pdf = pdf_urls[1] if len(pdf_urls) > 1 else None

        donem_str = f("Dönemi ve Yasama Yılı")
        donem, yyil = (donem_str.split("/") + ["", ""])[:2]

        return Onerge(
            guid=guid,
            donem=donem.strip(),
            yasama_yili=yyil.strip(),
            esas_no=f("Esas Numarası"),
            geliş_tarihi=f("Başkanlığa Geliş Tarihi"),
            özet=f("Önergenin Özeti"),
            sahip_il=sahip_il.strip(),
            sahip_isim=sahip_isim.strip(),
            muhatap_bakanlık=muh_bak.strip(),
            muhatap_bakan=muh_isim.strip(),
            durum=f("Son Durumu"),
            onerge_pdf_url=onerge_pdf,
            cevap_pdf_url=cevap_pdf,
            detail_html_hash=h,
        )

    # ----- Phase A: metadata ------------------------------------------

    def _load_done_meta(self) -> set[str]:
        if not self.done_meta_file.exists():
            return set()
        return set(self.done_meta_file.read_text().splitlines())

    def _append_done(self, guids: list[str]) -> None:
        with self.done_meta_file.open("a") as f:
            for g in guids:
                f.write(g + "\n")

    def _flush_shard(self, rows: list[dict], idx: int) -> None:
        if not rows:
            return
        path = self.meta_dir / f"part-{idx:04d}.parquet"
        pd.DataFrame(rows).to_parquet(path, index=False)
        logger.info("metadata shard flushed: %s (%d rows)", path.name, len(rows))

    def _append_errors(self, errs: list[dict]) -> None:
        if not errs:
            return
        new = pd.DataFrame(errs)
        if self.errors_file.exists():
            new = pd.concat([pd.read_parquet(self.errors_file), new], ignore_index=True)
        new.to_parquet(self.errors_file, index=False)

    def phase_a(self, guids: list[str]) -> None:
        done = self._load_done_meta()
        todo = [g for g in guids if g not in done]
        logger.info("phase-A: %d total, %d done, %d todo", len(guids), len(done), len(todo))
        if not todo:
            logger.info("nothing to do")
            return

        existing = sorted(self.meta_dir.glob("part-*.parquet"))
        next_idx = int(existing[-1].stem.split("-")[1]) + 1 if existing else 0

        buffer: list[dict] = []
        flushed: list[str] = []
        errors: list[dict] = []

        with tqdm(total=len(todo), desc="phase-A meta", unit="onerge") as pbar:
            for guid in todo:
                try:
                    o = self.parse_detail(guid)
                    buffer.append(asdict(o))
                    flushed.append(guid)
                except Exception as exc:  # noqa: BLE001
                    errors.append({"guid": guid, "error": repr(exc)[:300]})
                pbar.update(1)
                if len(flushed) >= BATCH_SIZE:
                    self._flush_shard(buffer, next_idx)
                    self._append_done(flushed)
                    self._append_errors(errors)
                    next_idx += 1
                    buffer.clear()
                    flushed.clear()
                    errors.clear()

        self._flush_shard(buffer, next_idx)
        self._append_done(flushed)
        self._append_errors(errors)
        logger.info("phase-A complete; %d errors logged", self._count_errors())

        # consolidate shards into single parquet for downstream consumers
        self._consolidate_meta()

    def _count_errors(self) -> int:
        if not self.errors_file.exists():
            return 0
        return len(pd.read_parquet(self.errors_file))

    def _consolidate_meta(self) -> Path:
        shards = sorted(self.meta_dir.glob("part-*.parquet"))
        if not shards:
            logger.warning("no metadata shards to consolidate")
            return self.out / "yazili_soru_meta.parquet"
        df = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
        out = self.out / "yazili_soru_meta.parquet"
        df.to_parquet(out, index=False)
        logger.info("consolidated %d shards → %s (%d rows)", len(shards), out, len(df))
        return out

    # ----- Phase B: PDF download in parallel --------------------------

    def _download_one(self, url: str, fname: str) -> tuple[str, bool, str]:
        path = self.pdf_dir / fname
        if path.exists() and path.stat().st_size > 0:
            return fname, True, "skip-exists"
        try:
            r = self._get_pdf(url)
            path.write_bytes(r.content)
            return fname, True, "ok"
        except Exception as exc:  # noqa: BLE001
            return fname, False, repr(exc)[:200]

    def phase_b(self, meta_path: Path | None = None) -> None:
        meta_path = meta_path or self._consolidate_meta()
        df = pd.read_parquet(meta_path)
        jobs: list[tuple[str, str]] = []
        for _, row in df.iterrows():
            g = row["guid"]
            if row["onerge_pdf_url"]:
                jobs.append((row["onerge_pdf_url"], f"{g}_onerge.pdf"))
            if row["cevap_pdf_url"]:
                jobs.append((row["cevap_pdf_url"], f"{g}_cevap.pdf"))
        logger.info("phase-B: %d PDF jobs (%d workers, %.1fs gap)", len(jobs), CDN_WORKERS, CDN_GAP_SEC)

        errors: list[dict] = []
        with tqdm(total=len(jobs), desc="phase-B pdf", unit="pdf") as pbar:
            with ThreadPoolExecutor(max_workers=CDN_WORKERS) as ex:
                futures = {ex.submit(self._download_one, u, f): f for (u, f) in jobs}
                for fut in as_completed(futures):
                    fname, ok, msg = fut.result()
                    if not ok and msg != "skip-exists":
                        errors.append({"file": fname, "error": msg})
                    pbar.update(1)
        self._append_errors(errors)
        logger.info("phase-B complete; %d new errors", len(errors))


# --- CLI --------------------------------------------------------------

def _read_guids(path: Path) -> list[str]:
    return [g.strip() for g in path.read_text().splitlines() if g.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="TBMM Bronze scraper (Phase A=meta, Phase B=PDF)")
    ap.add_argument("--out", default=Path("data/bronze"), type=Path)
    ap.add_argument("--phase", choices=["a", "b", "all"], default=None)
    ap.add_argument("--guids", type=Path, help="newline-separated GUID file (phase a / all)")
    ap.add_argument("--meta", type=Path, help="metadata parquet (phase b only)")
    ap.add_argument("--test", action="store_true", help="single-GUID smoke test")
    args = ap.parse_args()

    s = TBMMScraper(args.out)
    if args.test:
        sample = "eea2e48d-d119-4df8-98cb-018d1179fa67"
        o = s.parse_detail(sample)
        print(json.dumps(asdict(o), indent=2, ensure_ascii=False))
        return

    if args.phase in ("a", "all"):
        if not args.guids:
            ap.error("--phase a requires --guids")
        s.phase_a(_read_guids(args.guids))

    if args.phase in ("b", "all"):
        s.phase_b(meta_path=args.meta)


if __name__ == "__main__":
    main()
