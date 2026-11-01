"""Grounding check for rule constraints (utils/ontology_grounding.py).

A constraint is "ungrounded" when its value cannot be found in the source text:
  - ISO dates (YYYY-MM-DD): the full date must appear in the text, in any of the
    usual forms ("14 December 2016", "December 14, 2016", "14.12.2016", "2016-12-14");
  - numbers: the number must appear in the text (separators and the words
    thousand / million / billion are handled, e.g. "EUR 150 million" -> 150000000);
  - currency units (EUR, USD, GBP): the code must appear in the text.

A date normalised from prose ("14 December 2016" -> 2016-12-14) is kept; a date
whose day was made up (text says "in 2017", value 2017-12-31) is dropped.
"""
from __future__ import annotations

import re

CURRENCIES = ("EUR", "USD", "GBP")

_ISO_DATE = re.compile(r"^\s*(\d{4})-(\d{2})-(\d{2})")
_MULTIPLIERS = {"thousand": 1e3, "million": 1e6, "billion": 1e9}
# "1 000", "50,000,000", "1.500", "1,3" (decimal comma), "150"
_NUMBER = re.compile(
    r"(?<![\w.])(\d{1,3}(?:[ ,.]\d{3})+|\d+)(?:[.,](\d+))?\s*(thousand|million|billion)?",
    re.IGNORECASE,
)


_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
           "september", "october", "november", "december"]


def _date_in_text(year: str, month: str, day: str, text: str) -> bool:
    """True if the date appears in the text in one of the usual forms."""
    low = re.sub(r"\s+", " ", text.lower())
    m, d = int(month), int(day)
    if not 1 <= m <= 12:
        return False
    name = _MONTHS[m - 1]
    forms = [
        rf"(?<!\d){d}(?:st|nd|rd|th)? {name},? {year}(?!\d)",
        rf"{name} {d}(?:st|nd|rd|th)?,? {year}(?!\d)",
        rf"(?<!\d)0?{d}[./]0?{m}[./]{year}(?!\d)",
        rf"{year}-{month}-{day}",
    ]
    return any(re.search(f, low) for f in forms)


def _numbers_in_text(text: str) -> set[float]:
    found: set[float] = set()
    for m in _NUMBER.finditer(text):
        digits = re.sub(r"[ ,.]", "", m.group(1))
        value = float(digits + ("." + m.group(2) if m.group(2) else ""))
        found.add(round(value, 6))
        if m.group(3):
            found.add(round(value * _MULTIPLIERS[m.group(3).lower()], 6))
    return found


def constraint_is_ungrounded(constraint: dict, text: str, currencies=CURRENCIES) -> str | None:
    """Return a short reason if the constraint's value is not supported by `text`, else None."""
    # "EUR-Lex" is the database name, not the currency
    clean = re.sub(r"EUR-Lex", "", text or "", flags=re.IGNORECASE)

    unit = str(constraint.get("unit") or "").strip()
    if unit.upper() in currencies and not re.search(rf"\b{re.escape(unit)}\b", clean, re.IGNORECASE):
        return f"currency {unit} not in text"

    raw = constraint.get("right_operand")
    if raw is None or str(raw).strip() == "":
        return None
    value = str(raw).strip()

    date = _ISO_DATE.match(value)
    if date:
        if not _date_in_text(*date.groups(), clean):
            return f"date {date.group(0)} not in text"
        return None

    try:
        number = float(re.sub(r"[ ,]", "", value))
    except ValueError:
        # free text ("Syria", "maintain macroeconomic stability"): every word of 4+ letters
        # must occur in the text, compared on its stem (word without the last two letters)
        words = [w.lower() for w in re.findall(r"[A-Za-z]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", value))]
        low = clean.lower()
        for w in words:
            if len(w) >= 4 and w[:max(4, len(w) - 2)] not in low:
                return f"word '{w}' not in text"
        return None
    if round(number, 6) not in _numbers_in_text(clean):
        return f"value {value} not in text"
    return None