class ErrorUtilityMixin:
    """Adds the linear bonus (–0.30 … +0.30) to the base utility."""
    def __init__(self, monitor, mapping):
        self.p_map   = mapping                 # sample -> principle (id or name)
        self.bonus   = monitor.make_linear_bonus()
        self._id2name = getattr(monitor, "id2name", {})

    def weighted(self, sample, base_u: float) -> float:
        key = self.p_map(sample)
        # accept tuple/list (id,name,...) or raw id
        if isinstance(key, (list, tuple)) and len(key) > 1:
            key = key[1]
        if key not in self.bonus:
            key = self._id2name.get(str(key), key)
        u = base_u + self.bonus.get(key, 0.0)
        return max(-1.0, min(1.0, u))
