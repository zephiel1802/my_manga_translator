"""Font and style resolution helpers for render-stage workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .image_io import ensure_path


def list_project_fonts(project_root: Path | str) -> list[tuple[str, Path]]:
    """Return available repo/project fonts as ``(display_name, path)`` pairs."""

    root_path = ensure_path(project_root)
    candidate_dirs = [root_path / "fonts", Path(__file__).resolve().parents[1] / "fonts"]
    fonts: list[tuple[str, Path]] = []
    seen_paths: set[Path] = set()
    for font_dir in candidate_dirs:
        if not font_dir.exists():
            continue
        for pattern in ("*.ttf", "*.otf", "*.ttc"):
            for font_file in sorted(font_dir.glob(pattern)):
                resolved = font_file.resolve()
                if resolved in seen_paths:
                    continue
                seen_paths.add(resolved)
                fonts.append((font_file.stem, resolved))
    return fonts


def resolve_font_path(
    project_root: Path | str,
    *,
    font_name: str | None = None,
    font_path: str | None = None,
) -> str:
    """Resolve a selected font name/path to a concrete file path when possible."""

    explicit_path = str(font_path or "").strip()
    if explicit_path:
        resolved = ensure_path(explicit_path)
        if not resolved.exists():
            raise FileNotFoundError(f"Font file is missing: {resolved}")
        return str(resolved)

    normalized_name = str(font_name or "").strip().lower()
    project_fonts = list_project_fonts(project_root)
    if normalized_name:
        for display_name, candidate_path in project_fonts:
            if display_name.lower() == normalized_name or candidate_path.name.lower() == normalized_name:
                return str(candidate_path)

    if project_fonts:
        return str(project_fonts[0][1])

    return ""


def parse_color_value(value: str | None) -> tuple[int, int, int] | None:
    """Parse a simple user-entered RGB value.

    Supported formats:
    - ``#RRGGBB``
    - ``R,G,B``
    - ``auto`` / empty -> ``None``
    """

    raw_value = str(value or "").strip()
    if not raw_value or raw_value.lower() == "auto":
        return None

    if raw_value.startswith("#") and len(raw_value) == 7:
        try:
            return (
                int(raw_value[1:3], 16),
                int(raw_value[3:5], 16),
                int(raw_value[5:7], 16),
            )
        except ValueError as exc:
            raise ValueError(f"Invalid color value: {raw_value}") from exc

    if "," in raw_value:
        parts = [part.strip() for part in raw_value.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Invalid RGB color value: {raw_value}")
        try:
            red, green, blue = [max(0, min(255, int(part))) for part in parts]
        except ValueError as exc:
            raise ValueError(f"Invalid RGB color value: {raw_value}") from exc
        return (red, green, blue)

    raise ValueError(f"Unsupported color value: {raw_value}")


def coerce_serializable_color(color: tuple[int, int, int] | None) -> list[int] | None:
    if color is None:
        return None
    return [int(color[0]), int(color[1]), int(color[2])]


__all__ = [
    "coerce_serializable_color",
    "list_project_fonts",
    "parse_color_value",
    "resolve_font_path",
]
