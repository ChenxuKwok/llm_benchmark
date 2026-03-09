import re
from typing import List, Optional, Tuple


_WORD_RE = re.compile(r"[a-z0-9']+")


def normalize_words(text: str) -> List[str]:
    if not text:
        return []
    text = text.lower()
    return _WORD_RE.findall(text)


def edit_distance(ref: List[str], hyp: List[str]) -> Tuple[int, int, int]:
    n = len(ref)
    m = len(hyp)
    if n == 0:
        return (0, 0, m)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    # backtrace for S, D, I
    i, j = n, m
    subs = dels = ins = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (
            0 if ref[i - 1] == hyp[j - 1] else 1
        ):
            if ref[i - 1] != hyp[j - 1]:
                subs += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            dels += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return subs, dels, ins


def wer(reference: str, hypothesis: str) -> Optional[float]:
    ref_words = normalize_words(reference or "")
    hyp_words = normalize_words(hypothesis or "")
    if not ref_words:
        if not hyp_words:
            return 0.0
        return 1.0
    subs, dels, ins = edit_distance(ref_words, hyp_words)
    return (subs + dels + ins) / max(1, len(ref_words))
