from collections import Counter
from .principles import PRINCIPLES

class ErrorMonitor:
    def __init__(self, dataset_name: str):
        self.dataset = dataset_name
        triples = PRINCIPLES[dataset_name]                 # [["1","Name","Def"], ...]
        self.id2name   = {str(t[0]): t[1] for t in triples}
        self.principles = [t[1] for t in triples]          # names only
        self.counter  = Counter({name: 0 for name in self.principles})

    # ----- called during the OFFLINE pre-scan -------
    def log(self, principle):
        """
        Accepts a principle NAME, numeric ID, or a (id,name,...) triple.
        """
        key = principle
        # normalize triple/list/tuple -> prefer name, else id
        if isinstance(key, (list, tuple)):
            if len(key) > 1:
                key = key[1]                # name
            elif len(key) == 1:
                key = key[0]                # id
        # map id -> name if needed
        if key not in self.counter:
            key = self.id2name.get(str(key), None)
        if key is None:
            return                          # unknown: ignore
        self.counter[key] += 1

    # ----- after the pre-scan is finished -----------
    def make_linear_bonus(self) -> dict[str, float]:
        """
        Map counts linearly onto [-0.30, +0.30].
        max(count)  →  +0.30
        min(count)  →  -0.30
        """
        counts = self.counter
        if not counts:
            return {}
        hi = max(counts.values())
        lo = min(counts.values())
        if hi == lo:                          # all equally frequent
            return {p: 0.0 for p in self.principles}
        span = hi - lo
        return {p: (-0.30 + 0.60 * (counts[p] - lo) / span) for p in self.principles}
