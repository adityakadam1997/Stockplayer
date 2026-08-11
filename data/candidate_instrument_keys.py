"""Instrument-key resolution for the watchlist-candidate vetting pipeline
(``scripts/vet_candidates.py``) -- kept entirely separate from
``data/instrument_fallback.py``'s ``VERIFIED_INSTRUMENT_KEYS`` (the live
15-symbol production watchlist's fallback map). Nothing in this module is
imported by ``scripts/paper_daily.py`` or anything else the daily cron runs.

Resolution order, per the pipeline spec:

1. Bulk resolution via ``assets.upstox.com`` (``data.instruments.load_instrument_map``)
   -- historically blocked from this sandbox's network egress (repo issue #3).
   If blocked, note it and move on; do not retry.
2. ``api.upstox.com`` symbol-search, if a separate, usable, unauthenticated
   endpoint exists (checked manually this session -- no such endpoint was
   found; every guessed path returned 401/404, so this step is a documented
   no-op here, not a fabricated integration).
3. Fallback: a hardcoded, per-symbol ``<SEGMENT>|<ISIN>`` map
   (``CANDIDATE_VERIFIED_INSTRUMENT_KEYS`` below), where each ISIN was found
   via web search against NSE's own listing pages (or, where NSE's page
   wasn't the top result, cross-checked across at least two independent
   sources -- Upstox's own stock pages, screener.in, tickertape.in, etc.)
   and then INDEPENDENTLY VERIFIED against a live Upstox historical-candle
   request before being trusted -- exactly the same discipline used for
   KOTAKBANK/BAJFINANCE in ``data/instrument_fallback.py``. A wrong ISIN
   silently pulls the wrong company's data, so verification is mandatory,
   not optional.
4. Anything still unresolved after all three steps is reported by name as
   UNRESOLVED (see ``resolve_candidates``'s return value) -- never guessed.

**50 of 51 candidates resolved and verified.** FINELABS is UNRESOLVED: no
NSE-listed company trading under this exact symbol could be found across
two independent research passes (nseindia.com search, general web search) --
reported as UNRESOLVED rather than guessed, per the pipeline's spec.

Two symbols needed correction/disambiguation during live verification
(the spec's "verify EACH ONE" step earning its keep -- a plausible-looking,
multi-source-agreed ISIN still turned out wrong in one case):

- **NSDL**: ``NSE_EQ|INE301O01023`` returned "Invalid Instrument key" even
  though INE301O01023 is the correct ISIN (confirmed by Wikipedia, Upstox's
  own stock page URL, Groww, BlinkX). ``BSE_EQ|INE301O01023`` DOES work and
  returns real OHLC data in the expected post-IPO price band (~Rs800-830) --
  Upstox's V3 API just doesn't have this instrument registered under the
  NSE_EQ segment yet (NSDL IPO'd July 2025; still a very recently listed
  instrument). Used as ``BSE_EQ|INE301O01023`` here.
- **TATAINVEST**: the first ISIN found (INE672A01018, agreed by
  upstox.com/screener.in/businesstoday.in) returned "Invalid Instrument
  key" against the live API despite multi-source agreement. A second,
  independent source (tickertape.in) gave INE672A01026 (only the last two
  digits differ from the first guess -- a plausible transcription error
  propagated across several sites) -- verified live, returns real OHLC
  data in Tata Investment Corporation's expected price band (~Rs670-715)
  consistent with its known market cap. Used as ``NSE_EQ|INE672A01026``
  here.
"""

from __future__ import annotations

from pathlib import Path

from data import instruments

# Verified via a live GET to api.upstox.com/v3/historical-candle/<key>/days/1/...
# during this pipeline's build -- every entry below returned real OHLC data,
# in a price range plausible for that company, for that symbol. All keys are
# NSE_EQ|<ISIN> except NSDL, which Upstox's V3 API only recognizes under the
# BSE_EQ segment (see module docstring).
CANDIDATE_VERIFIED_INSTRUMENT_KEYS: dict[str, str] = {
    "TANLA": "NSE_EQ|INE483C01032",
    "PPLPHARMA": "NSE_EQ|INE0DK501011",
    "INDSWFTLAB": "NSE_EQ|INE915B01019",
    "RATNAVEER": "NSE_EQ|INE05CZ01011",
    "KPIGREEN": "NSE_EQ|INE542W01025",
    "IDEA": "NSE_EQ|INE669E01016",
    "JKTYRE": "NSE_EQ|INE573A01042",
    "NSDL": "BSE_EQ|INE301O01023",  # not registered under NSE_EQ in Upstox's V3 API -- see module docstring
    "SOLEX": "NSE_EQ|INE880Y01017",
    "ZOTA": "NSE_EQ|INE358U01012",
    "SHAILY": "NSE_EQ|INE151G01028",
    # FINELABS: UNRESOLVED -- no NSE-listed company found under this symbol. Not guessed.
    "NPST": "NSE_EQ|INE0FFK01017",
    "CREDITACC": "NSE_EQ|INE741K01010",
    "KPL": "NSE_EQ|INE552U01010",
    "DIXON": "NSE_EQ|INE935N01020",
    "CDSL": "NSE_EQ|INE736A01011",
    "GODFRYPHLP": "NSE_EQ|INE260B01028",
    "SCODATUBES": "NSE_EQ|INE090501011",
    "BIRLANU": "NSE_EQ|INE557A01011",
    "COCHINSHIP": "NSE_EQ|INE704P01025",
    "SUNPHARMA": "NSE_EQ|INE044A01036",
    "THERMAX": "NSE_EQ|INE152A01029",
    "WEBELSOLAR": "NSE_EQ|INE855C01023",
    "HCLTECH": "NSE_EQ|INE860A01027",
    "BALAMINES": "NSE_EQ|INE050E01027",
    "MPSLTD": "NSE_EQ|INE943D01017",
    "SAFARI": "NSE_EQ|INE429E01023",
    "ERIS": "NSE_EQ|INE406M01024",
    "TATAINVEST": "NSE_EQ|INE672A01026",  # corrected ISIN -- see module docstring
    "BEML": "NSE_EQ|INE258A01024",
    "DRREDDY": "NSE_EQ|INE089A01031",
    "PIRAMALFIN": "NSE_EQ|INE202B01038",
    "LLOYDSENT": "NSE_EQ|INE080I01025",
    "LLOYDSENGG": "NSE_EQ|INE093R01011",
    "REDINGTON": "NSE_EQ|INE891D01026",
    "MGL": "NSE_EQ|INE002S01010",
    "IIFL": "NSE_EQ|INE530B01024",
    "NMDC": "NSE_EQ|INE584A01023",
    "IRCTC": "NSE_EQ|INE335Y01020",
    "RAILTEL": "NSE_EQ|INE0DD101019",
    "HINDALCO": "NSE_EQ|INE038A01020",
    "IONEXCHANG": "NSE_EQ|INE570A01022",
    "DIAMONDYD": "NSE_EQ|INE393P01035",
    "HIRECT": "NSE_EQ|INE835D01023",
    "VOLTAS": "NSE_EQ|INE226A01021",
    "PARAGMILK": "NSE_EQ|INE883N01014",
    "TMCV": "NSE_EQ|INE1TAE01010",
    "EIMCOELECO": "NSE_EQ|INE158B01016",
    "OLAELEC": "NSE_EQ|INE0LXG01040",
    "IDFCFIRSTB": "NSE_EQ|INE092T01019",
}


def resolve_candidates(
    candidates: list[str], cache_dir: Path, force_refresh: bool = False
) -> tuple[dict[str, str], list[str]]:
    """Resolve every symbol in ``candidates`` to an instrument key.

    Returns ``(resolved, unresolved)``: ``resolved`` maps symbol ->
    ``NSE_EQ|<ISIN>`` for every symbol found via the bulk instrument master
    or the verified fallback map; ``unresolved`` lists every symbol found by
    neither, in input order. Never raises on partial resolution -- the
    pipeline's job is to vet whatever it CAN resolve and report the rest
    honestly, not to halt on one bad symbol."""
    try:
        symbol_map = instruments.load_instrument_map(cache_dir, force_refresh=force_refresh)
    except Exception as exc:
        print(f"[candidate_instrument_keys] assets.upstox.com bulk resolution unavailable ({exc}); "
              f"falling back to the verified per-symbol map.")
        symbol_map = {}

    resolved: dict[str, str] = {}
    unresolved: list[str] = []
    for symbol in candidates:
        key = symbol_map.get(symbol) or CANDIDATE_VERIFIED_INSTRUMENT_KEYS.get(symbol)
        if key is None:
            unresolved.append(symbol)
        else:
            resolved[symbol] = key
    return resolved, unresolved
