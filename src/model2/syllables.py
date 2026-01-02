import re

VOWEL_GROUP = re.compile(r"[aeiouy]+", re.I)

def rough_syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    # Basic English heuristics
    groups = len(VOWEL_GROUP.findall(w))
    # silent 'e'
    if w.endswith("e"):
        groups = max(1, groups - 1)
    # diphthong exceptions
    groups = max(1, groups)
    return groups

def count_syllables_line(line: str) -> int:
    return sum(rough_syllables(w) for w in re.findall(r"[A-Za-z']+", line))

def structure_575(lines):
    if len(lines) != 3:
        return False, (0,0,0)
    s = tuple(count_syllables_line(line) for line in lines)
    return (s == (5,7,5)), s
