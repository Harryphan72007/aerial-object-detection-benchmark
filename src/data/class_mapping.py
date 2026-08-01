"""Canonical VisDrone category definitions and two-class mapping."""

from __future__ import annotations

from dataclasses import dataclass

VISDRONE_CLASSES = {
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
}
PERSON = frozenset({"pedestrian", "people"})
VEHICLE = frozenset(
    {"bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"}
)
LIGHT_VEHICLES = frozenset({"bicycle", "tricycle", "awning-tricycle"})
IGNORED_CATEGORY_IDS = frozenset({0, 11})


@dataclass(frozen=True)
class ClassMapping:
    """Map official VisDrone IDs to a versioned benchmark class space."""

    track: str = "2class"
    exclude_light_vehicles: bool = False

    def __post_init__(self) -> None:
        if self.track not in {"2class", "10class"}:
            raise ValueError(f"unsupported track: {self.track}")

    @property
    def class_names(self) -> list[str]:
        return (
            ["person", "vehicle"]
            if self.track == "2class"
            else list(VISDRONE_CLASSES.values())
        )

    def map_category(self, original_id: int) -> int | None:
        if original_id not in VISDRONE_CLASSES:
            return None
        if self.track == "10class":
            return original_id
        name = VISDRONE_CLASSES[original_id]
        if name in PERSON:
            return 1
        if name in VEHICLE:
            if self.exclude_light_vehicles and name in LIGHT_VEHICLES:
                return None
            return 2
        return None

    def coco_categories(self) -> list[dict[str, object]]:
        return [
            {"id": index + 1, "name": name, "supercategory": name}
            for index, name in enumerate(self.class_names)
        ]
