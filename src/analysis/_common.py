"""Shared reference data + helpers for the RQ1/RQ2 Spark analyses.

Centralises the things both research questions need:

* canonical (short) party labels,
* MP -> (party, electoral province) lookup built from the Wikipedia parse,
* a Turkish stop-word list tuned for parliamentary text,
* the 81-province gazetteer used to detect province *mentions* in question
  summaries (RQ2 attention map).

Kept dependency-free (pure Python / stdlib) so it can be imported both inside a
Spark driver and from plain pandas utilities.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

REF_DIR = Path(__file__).resolve().parents[2] / "data" / "reference"

# --------------------------------------------------------------------------
# Party labels
# --------------------------------------------------------------------------
# The Wikipedia parse stores full legal names; collapse them to the short
# labels used in TBMM reporting (and merge the HDP successor names into DEM).
PARTY_CANON = {
    "Adalet ve Kalkınma Partisi": "AK Parti",
    "Cumhuriyet Halk Partisi": "CHP",
    "Halkların Eşitlik ve Demokrasi Partisi": "DEM Parti",
    "Yeşiller ve Sol Gelecek Partisi": "DEM Parti",
    "Halkların Demokratik Partisi": "DEM Parti",
    "Milliyetçi Hareket Partisi": "MHP",
    "İYİ Parti": "İYİ Parti",
    "Türkiye İşçi Partisi": "TİP",
    "Yeniden Refah Partisi": "Yeniden Refah",
    "Hür Dava Partisi": "HÜDA PAR",
    "Demokrasi ve Atılım Partisi": "DEVA",
    "Saadet Partisi": "Saadet",
    "Gelecek Partisi": "Gelecek",
    "Demokratik Sol Parti": "DSP",
    "Bağımsız": "Bağımsız",
}


def canon_party(raw: str | None) -> str:
    if not raw:
        return "Bilinmiyor"
    raw = raw.strip()
    return PARTY_CANON.get(raw, raw)


# --------------------------------------------------------------------------
# MP -> party / province lookup
# --------------------------------------------------------------------------
# Four CSV names differ from the Wikipedia spelling; resolved the same way the
# RQ3 network analysis did, so all three research questions share one mapping.
_MANUAL = {
    "Hakkı Saruhan OLUÇ": ("DEM Parti", "Antalya"),
    "Mehmet Selim ENSARİOĞLU": ("AK Parti", "İstanbul"),
    "Selcan TAŞCI": ("AK Parti", "Tekirdağ"),
    "Şahzade DEMİR": ("HÜDA PAR", "Gaziantep"),
}


def normalize_province(prov: str | None) -> str:
    """``"İstanbul (II)"`` -> ``"İstanbul"`` (drop the multi-district suffix)."""
    if not prov:
        return "Bilinmiyor"
    return re.sub(r"\s*\(.*?\)\s*$", "", prov).strip() or "Bilinmiyor"


def build_mp_lookup() -> dict[str, dict[str, str]]:
    """name (as it appears in the question CSV) -> {party, province}."""
    pm = json.loads((REF_DIR / "mp_party_mapping.json").read_text(encoding="utf-8"))
    out: dict[str, dict[str, str]] = {}
    for name, rec in pm.items():
        out[name] = {
            "party": canon_party(rec.get("elected_party")),
            "party_current": canon_party(rec.get("current_party")),
            "province": normalize_province(rec.get("province")),
        }
    for name, (party, prov) in _MANUAL.items():
        out[name] = {"party": party, "party_current": party, "province": prov}
    return out


# --------------------------------------------------------------------------
# Turkish stop words (parliamentary register)
# --------------------------------------------------------------------------
TURKISH_STOPWORDS = sorted({
    # function words
    "acaba", "ama", "ancak", "ara", "aslında", "az", "bana", "bazı", "belki",
    "ben", "beni", "benim", "beş", "bile", "bir", "birçok", "biri", "birkaç",
    "birşey", "biz", "bize", "bizi", "bizim", "böyle", "böylece", "bu", "buna",
    "bunda", "bundan", "bunlar", "bunları", "bunların", "bunu", "bunun", "burada",
    "çok", "çünkü", "da", "daha", "dahi", "de", "defa", "değil", "diğer", "diye",
    "doğru", "dolayı", "edilir", "eğer", "en", "etmesi", "etti", "ettiği", "gibi",
    "göre", "halen", "hangi", "hani", "hatta", "hem", "henüz", "hep", "hepsi",
    "her", "herhangi", "hiç", "için", "ile", "ilgili", "ise", "işte", "kadar",
    "karşın", "kendi", "kendine", "ki", "kim", "kez", "madem", "mı", "mu", "mü",
    "nasıl", "ne", "neden", "nedenle", "nerede", "nereye", "niçin", "niye", "o",
    "olan", "olarak", "oldu", "olduğu", "olduğunu", "olmadı", "olması", "olup",
    "ona", "onlar", "onları", "onların", "onu", "onun", "orada", "öyle", "oysa",
    "para", "rağmen", "sanki", "şayet", "şey", "şu", "şöyle", "siz", "sizin",
    "sonra", "tabi", "tarafından", "tüm", "üzere", "var", "vardı", "ve", "veya",
    "ya", "yani", "yine", "yoksa",
    # parliamentary boilerplate that drowns out topics in 'özet'
    "ilişkin", "hakkında", "dair", "soru", "önerge", "önergesi", "yazılı",
    "bakan", "bakanı", "bakanlık", "bakanlığı", "bakanlığına", "tarafından",
    "edilmesi", "yapılması", "yapılan", "alınması", "verilen", "konusunda",
    "iddia", "iddiası", "edilen", "ettiği", "edip", "olup", "açıklaması",
    # Turkish case-suffix fragments left behind when an apostrophe splits a
    # proper noun ("Şanlıurfa'nın" -> "şanlıurfa" + "nın"); pure noise as tokens.
    "nin", "nın", "nun", "nün", "nda", "nde", "ndan", "nden", "deki", "daki",
    "den", "dan", "tan", "ten", "lar", "ler", "ları", "leri", "lığı", "liği",
    "nan", "nen", "ında", "inde", "ına", "ine", "nın", "için",
    # Parliamentary written-question boilerplate — the formal letter opening /
    # closing repeated in EVERY full-text body ("Türkiye Büyük Millet Meclisi
    # Başkanlığına … aşağıdaki sorularımın … tarafından yazılı olarak
    # cevaplandırılmasını arz ederim"). Without these, full-text LDA is drowned.
    "türkiye", "büyük", "millet", "meclisi", "meclis", "başkanlığına",
    "başkanlık", "başkanlığı", "milletvekili", "milletvekilleri", "tbmm",
    "arz", "ederim", "ederiz", "mıdır", "mudur", "müdür", "midir", "nedir",
    "kaçtır", "sorularımın", "sorularımı", "soruların", "sorum", "sorularım",
    "soruları", "cevaplandırılması", "cevaplandırılmasını", "cevaplanmasını",
    "cevaplandırılmak", "sayın", "aşağıdaki", "gereğini", "talep", "ediyorum",
    "hususunda", "gerekli", "grup", "grubu", "parti", "partisi", "vekili",
    "ben", "adına", "üzere", "kanunu", "sayılı", "madde", "maddesi", "fıkra",
    "anayasa", "anayasanın", "içtüzük", "içtüzüğünün", "uyarınca", "gereği",
    "olmak", "bağlamda", "söz", "konusu", "halkla", "ilişkiler", "devam",
    "gov", "posta", "tel", "telefon", "adres", "faks", "eposta",
    # Party / group identifiers in the letterhead + signature block. These leak
    # the label into the text (full-text party classification jumps to ~99%
    # otherwise) and dominate FP-Growth; remove so topics reflect content.
    # party abbreviations only (the full letterhead is stripped into `govde`,
    # so content words like "halk"/"cumhuriyet"/"genel" are kept — "halk
    # sağlığı" etc. are legitimate signal).
    "chp", "mhp", "akp", "dem", "hdp", "yeniden", "refah",
    "başkanvekili", "sayın", "vekilleri",
    # "cumhuriyet" kept here (party name) because a minority of closings aren't
    # "saygılarımla"-terminated so the signature "Cumhuriyet Halk Partisi" can
    # survive into govde. "halk" is left IN the vocabulary (content: halk sağlığı).
    "cumhuriyet",
    # High-frequency content-free filler that dominates full-text topics
    # (and their common case-inflected forms). maxDF in the vectoriser catches
    # the rest automatically.
    "bulunan", "bulunmaktadır", "iddiasına", "iddiası", "iddialar", "iddialara",
    "talebine", "talebi", "talep", "yaşanan", "yaşandığı", "yılda", "yıl",
    "yılında", "arasında", "ilçesinde", "ilçesi", "ilçe", "ilinde", "ili",
    "ilin", "edildiği", "edilmiş", "edilmesi", "tespit", "tespiti", "çeşitli",
    "durumu", "durum", "kişinin", "kişi", "kişiye", "sayısı", "sayı", "kaç",
    "nelerdir", "yapılan", "yapılması", "yönelik", "nedeniyle", "ilgili",
    "konusunda", "alan", "devam", "ettiği",
})


# --------------------------------------------------------------------------
# 81-province gazetteer (RQ2 mention detection)
# --------------------------------------------------------------------------
PROVINCES_81 = [
    "Adana", "Adıyaman", "Afyonkarahisar", "Ağrı", "Aksaray", "Amasya", "Ankara",
    "Antalya", "Ardahan", "Artvin", "Aydın", "Balıkesir", "Bartın", "Batman",
    "Bayburt", "Bilecik", "Bingöl", "Bitlis", "Bolu", "Burdur", "Bursa",
    "Çanakkale", "Çankırı", "Çorum", "Denizli", "Diyarbakır", "Düzce", "Edirne",
    "Elazığ", "Erzincan", "Erzurum", "Eskişehir", "Gaziantep", "Giresun",
    "Gümüşhane", "Hakkari", "Hatay", "Iğdır", "Isparta", "İstanbul", "İzmir",
    "Kahramanmaraş", "Karabük", "Karaman", "Kars", "Kastamonu", "Kayseri",
    "Kırıkkale", "Kırklareli", "Kırşehir", "Kilis", "Kocaeli", "Konya", "Kütahya",
    "Malatya", "Manisa", "Mardin", "Mersin", "Muğla", "Muş", "Nevşehir", "Niğde",
    "Ordu", "Osmaniye", "Rize", "Sakarya", "Samsun", "Siirt", "Sinop", "Sivas",
    "Şanlıurfa", "Şırnak", "Tekirdağ", "Tokat", "Trabzon", "Tunceli", "Uşak",
    "Van", "Yalova", "Yozgat", "Zonguldak",
]

# A few common alternate spellings seen in TBMM text -> canonical province.
PROVINCE_ALIASES = {
    "Maraş": "Kahramanmaraş",
    "Urfa": "Şanlıurfa",
    "Antep": "Gaziantep",
    "Afyon": "Afyonkarahisar",
    "İçel": "Mersin",
}


def _casefold_tr(text: str) -> str:
    """Turkish-aware lower-casing (handles I/İ) for matching only."""
    text = text.replace("İ", "i").replace("I", "ı")
    return text.lower()


def build_province_pattern() -> "re.Pattern[str]":
    """Regex that matches any province name as a whole token, allowing a Turkish
    case suffix (``Diyarbakır'da``, ``İzmir'in``, ``Vanlılar`` is *not* matched
    because we anchor on a word boundary + optional apostrophe-suffix)."""
    names = sorted(set(PROVINCES_81) | set(PROVINCE_ALIASES), key=len, reverse=True)
    folded = [re.escape(_casefold_tr(n)) for n in names]
    # name, optionally followed by "'<suffix>" or directly a lowercase suffix,
    # bounded so 'van' inside 'vana' / 'kars' inside 'karşı' do not match.
    pattern = r"(?<![a-zçğıöşü])(" + "|".join(folded) + r")(?:['’][a-zçğıöşü]+|(?=[^a-zçğıöşü]|$))"
    return re.compile(pattern)


def extract_province_mentions(text: str | None, pattern: "re.Pattern[str]") -> list[str]:
    """Return the canonical provinces mentioned in *text* (deduplicated)."""
    if not text:
        return []
    folded = _casefold_tr(text)
    fold_to_canon = {_casefold_tr(p): p for p in PROVINCES_81}
    fold_to_canon.update({_casefold_tr(a): c for a, c in PROVINCE_ALIASES.items()})
    found: list[str] = []
    for m in pattern.finditer(folded):
        canon = fold_to_canon.get(m.group(1))
        if canon and canon not in found:
            found.append(canon)
    return found
