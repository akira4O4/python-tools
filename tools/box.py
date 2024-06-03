from typing import Optional, List
from dataclasses import dataclass, field


@dataclass
class Location:
    x: Optional[int] = 0
    y: Optional[int] = 0
    w: Optional[int] = 0
    h: Optional[int] = 0

    def raw(self) -> List[int]:
        return [self.x, self.y, self.w, self.h]


@dataclass
class Box:
    location: Optional[Location] = None
    score: Optional[float] = 0.0
    label_id: Optional[int] = -1

    def raw(self) -> dict:
        return {
            "box": self.location.raw(),
            "score": self.score,
            "label_id": self.label_id
        }


@dataclass
class Detection:
    image: Optional[str] = ''
    is_empty: Optional[bool] = True
    boxes: Optional[List[Box]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.boxes)
