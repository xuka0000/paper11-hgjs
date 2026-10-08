from dataclasses import dataclass


@dataclass(frozen=True)
class WeatherSlot:
    layer: int
    grid_id: int
    severity: float
    wind_speed: float
    wind_direction: float
    temperature: float
    uncertainty: float


def road_effective_speed(
    nominal: float,
    structure_factor: float,
    weather: WeatherSlot,
    beta_w: float,
    beta_eta: float,
) -> float:
    factor = 1.0 + beta_w * weather.wind_speed + beta_eta * weather.uncertainty
    return nominal / (structure_factor * factor)
