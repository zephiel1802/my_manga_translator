"""Small render-stage models and config helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class RenderConfig:
    """Serializable configuration for page rendering."""

    font_name: str = ""
    font_path: str = ""
    font_size_mode: str = "fit"
    min_font_size: int = 12
    max_font_size: int = 72
    text_color: tuple[int, int, int] | None = None
    stroke_enabled: bool = True
    stroke_color: tuple[int, int, int] | None = None
    stroke_width: float | None = None
    auto_color: bool = True
    auto_direction: bool = True
    vertical_cjk: bool = True
    save_sprites: bool = True
    force: bool = False

    @classmethod
    def from_value(cls, value: "RenderConfig | dict[str, Any] | None") -> "RenderConfig":
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            return cls()

        return cls(
            font_name=str(value.get("font_name", "") or ""),
            font_path=str(value.get("font_path", "") or ""),
            font_size_mode=str(value.get("font_size_mode", "fit") or "fit"),
            min_font_size=_coerce_positive_int(value.get("min_font_size"), 12),
            max_font_size=max(
                _coerce_positive_int(value.get("min_font_size"), 12),
                _coerce_positive_int(value.get("max_font_size"), 72),
            ),
            text_color=_coerce_color_tuple(value.get("text_color")),
            stroke_enabled=bool(value.get("stroke_enabled", True)),
            stroke_color=_coerce_color_tuple(value.get("stroke_color")),
            stroke_width=_coerce_optional_float(value.get("stroke_width")),
            auto_color=bool(value.get("auto_color", True)),
            auto_direction=bool(value.get("auto_direction", True)),
            vertical_cjk=bool(value.get("vertical_cjk", True)),
            save_sprites=bool(value.get("save_sprites", True)),
            force=bool(value.get("force", False)),
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "font_name": self.font_name,
            "font_path": self.font_path,
            "font_size_mode": self.font_size_mode,
            "min_font_size": int(self.min_font_size),
            "max_font_size": int(self.max_font_size),
            "text_color": list(self.text_color) if self.text_color is not None else None,
            "stroke_enabled": bool(self.stroke_enabled),
            "stroke_color": list(self.stroke_color) if self.stroke_color is not None else None,
            "stroke_width": self.stroke_width,
            "auto_color": bool(self.auto_color),
            "auto_direction": bool(self.auto_direction),
            "vertical_cjk": bool(self.vertical_cjk),
            "save_sprites": bool(self.save_sprites),
            "force": bool(self.force),
        }


def _coerce_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        return int(default)
    return parsed if parsed > 0 else int(default)


def _coerce_optional_float(value: Any) -> float | None:
    if value in (None, "", False):
        return None
    try:
        parsed = float(value)
    except Exception:
        return None
    return parsed if parsed > 0 else None


def _coerce_color_tuple(value: Any) -> tuple[int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        red, green, blue = [max(0, min(255, int(channel))) for channel in value[:3]]
    except Exception:
        return None
    return (red, green, blue)


__all__ = ["RenderConfig"]
