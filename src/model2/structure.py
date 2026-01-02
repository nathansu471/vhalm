import regex as re
from syllables import structure_575

def normalize_haiku(text: str) -> list[str]:
    # keep exactly 3 non-empty lines
    lines = [l.strip(" -\t") for l in text.strip().splitlines() if l.strip()]
    if len(lines) > 3:
        lines = lines[:3]
    while len(lines) < 3:
        lines.append("")  # pad if needed
    return lines

def try_fix(lines: list[str]) -> list[str]:
    # conservative micro-fixes: remove trailing punctuation clusters; collapse double spaces
    fx = []
    for l in lines:
        l = re.sub(r"\s{2,}", " ", l)
        l = re.sub(r"[\.!,;\-—]+$", "", l).strip()
        fx.append(l)
    return fx

def ensure_575(text: str):
    lines = normalize_haiku(text)
    ok, syl = structure_575(lines)
    if ok:
        return lines, syl, True
    # One gentle pass of trimming/fixing
    lines = try_fix(lines)
    ok2, syl2 = structure_575(lines)
    return lines, (syl2 if ok2 else syl), ok2
