from dataclasses import dataclass


@dataclass(frozen=True)
class TriggerDecision:
    triggered: bool
    layer: int
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class TriggerPolicy:
    max_interval: int | None
    weather_threshold: float
    risk_threshold: float
    delay_threshold: float
    energy_threshold: float = 0.2

    def __post_init__(self) -> None:
        if self.max_interval is not None and self.max_interval <= 0:
            raise ValueError("max interval must be positive")

    def evaluate(
        self,
        last_layer: int,
        current_layer: int,
        events: list[tuple[str, int, float]],
    ) -> TriggerDecision:
        reasons: list[str] = []
        if (
            self.max_interval is not None
            and current_layer - last_layer >= self.max_interval
        ):
            reasons.append("regular")
        thresholds = {
            "weather": self.weather_threshold,
            "risk": self.risk_threshold,
            "vehicle_delay": self.delay_threshold,
            "uav_energy": self.energy_threshold,
        }
        for kind, layer, magnitude in events:
            if layer != current_layer:
                continue
            strong = kind in {
                "road",
                "road_closed",
                "uav_failure",
                "locked_tuple_unsafe",
            }
            threshold = thresholds.get(kind, self.risk_threshold)
            if strong or magnitude >= threshold:
                reasons.append(kind)
        unique = tuple(dict.fromkeys(reasons))
        return TriggerDecision(bool(unique), current_layer, unique)
