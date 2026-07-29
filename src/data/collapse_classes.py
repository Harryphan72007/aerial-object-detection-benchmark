"""VisDrone category definitions and two-class collapse logic."""
from __future__ import annotations
from dataclasses import dataclass

VISDRONE_CLASSES = {
    1: "pedestrian", 2: "people", 3: "bicycle", 4: "car", 5: "van",
    6: "truck", 7: "tricycle", 8: "awning-tricycle", 9: "bus", 10: "motor",
}
PERSON = {"pedestrian", "people"}
VEHICLE = {"bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"}
LIGHT_VEHICLES = {"bicycle", "tricycle", "awning-tricycle"}

@dataclass(frozen=True)
class ClassMapping:
    track: str = "2class"
    exclude_light_vehicles: bool = False

    @property
    def class_names(self) -> list[str]:
        return ["person", "vehicle"] if self.track == "2class" else list(VISDRONE_CLASSES.values())

    def map_category(self, original_id: int) -> int | None:
        if original_id not in VISDRONE_CLASSES: return None
        if self.track == "10class": return original_id
        if self.track != "2class": raise ValueError(f"unsupported track: {self.track}")
        name = VISDRONE_CLASSES[original_id]
        if name in PERSON: return 1
        if name in VEHICLE:
            if self.exclude_light_vehicles and name in LIGHT_VEHICLES: return None
            return 2
        return None

    def coco_categories(self) -> list[dict[str, object]]:
        return [{"id": i + 1, "name": n, "supercategory": n} for i, n in enumerate(self.class_names)]
