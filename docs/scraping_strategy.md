# TBMM Scraping Strategy

## Verified Endpoints (live fetched)

| Endpoint | URL | Status |
|----------|-----|--------|
| Yazılı Soru Detail | `tbmm.gov.tr/Denetim/Yazili-Soru-Onergesi-Detay/{GUID}` | ✓ HTML structured |
| MP Profile | `tbmm.gov.tr/milletvekili/MilletvekiliDetay?Id={GUID}` | ✓ HTML structured |
| Tutanak | `www5.tbmm.gov.tr/develop/owa/tutanak_sd.son_tutanak` | ✓ full text |
| Komisyon Tutanakları | `tbmm.gov.tr/Tutanaklar/KomisyonTutanaklari` | ✓ navigable list |

## Open Question — GUID Collection

We have:
- A working **detail page** (given a GUID).
- An MP profile that lists **"Sahibi Olduğu Yazılı Soru Önergeleri"** with detail links.

We need to enumerate **all 28th-term yazılı soru GUIDs**. Three viable paths:

### Path A — Form POST on listing page (preferred)
Endpoint: `tbmm.gov.tr/denetim/yazili-soru-onergeleri` (or `www5.tbmm.gov.tr/develop/owa/yazili_soru_sd.sorgu_baslangic`).
Inspect Chrome DevTools → Network tab → submit empty form for "Son Dönem Tüm Yasama Yılları" → capture the POST request (URL, payload, cookies, `__VIEWSTATE`).
Replay with `requests.Session` + `BeautifulSoup`. Parse response table → extract `<a href="...Detay/{GUID}">`.

### Path B — Crawl MP profiles (fallback)
1. Get MP list from `tbmm.gov.tr/milletvekili-arama/form` (also form-based).
2. For each MP → follow "Sahibi Olduğu Yazılı Soru Önergeleri" link.
3. Collect all GUIDs.
Pros: each MP page is a fixed URL. Cons: O(N_MP × avg_questions) HTTP calls, slower but deterministic.

### Path C — Selenium/Playwright (last resort)
If both form endpoints rely on heavy JS state (`__VIEWSTATE`, anti-CSRF, async XHR), use headless browser.
Cost: 10× slower, more brittle, but works.

## Recommendation

Try in order: **A → B → C**. Each step has a 1-hour timebox before falling back.

## Rate Limiting

- 1 req/sec single thread.
- `Retry-After` header respected.
- Exponential backoff on 5xx and 429.
- `User-Agent` mimics regular Chrome.
- `Accept-Language: tr-TR`.

## Storage

- Detail metadata → Parquet (`data/bronze/yazili_soru_meta.parquet`).
- Raw PDFs → `data/bronze/pdf/{GUID}_{onerge|cevap}.pdf`.
- Raw HTML snapshots → optional, for debugging.

## OCR Path

Validated path: Colab Pro GPU + PaddleOCR (`lang='latin'`) → batch 50 PDFs → Delta Silver.
Fallback if Turkish accuracy < 0.7: Tesseract `tur` + image preprocessing (deskew, denoise, binarize).

## Tutanak Path

- Direct HTML scrape from owa endpoint per `birleşim` ID.
- Pattern: speaker `AD SOYAD (İl)` already gives MP + province (no NER needed).
- 81-il dictionary regex for province mention counts.
