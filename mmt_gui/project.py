"""Lightweight project storage for the PyQt6 desktop shell."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import shutil
from typing import Any, Sequence

PROJECT_FILENAME = "project.json"
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
CACHE_STAGES = ("detection", "ocr", "ocr_crops", "translation", "inpaint", "render", "render_sprites", "masks")


@dataclass(slots=True)
class ProjectPage:
    """Serializable per-page project state."""

    source_path: str
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProjectPage" | None:
        source_path = payload.get("source_path")
        if not isinstance(source_path, str) or not source_path.strip():
            return None

        stages = payload.get("stages", {})
        if not isinstance(stages, dict):
            stages = {}

        normalized_stages: dict[str, dict[str, Any]] = {}
        for stage_name, stage_payload in stages.items():
            if not isinstance(stage_name, str):
                continue
            if isinstance(stage_payload, dict):
                normalized_stages[stage_name] = {
                    str(key): value for key, value in stage_payload.items()
                }
            else:
                normalized_stages[stage_name] = {}

        return cls(source_path=source_path, stages=normalized_stages)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "stages": dict(self.stages),
        }


@dataclass(slots=True)
class ProjectData:
    """Serializable GUI project state."""

    name: str
    pages: list[ProjectPage] = field(default_factory=list)
    current_page_index: int = 0
    settings: dict[str, Any] = field(default_factory=dict)

    @property
    def source_images(self) -> list[str]:
        return [page.source_path for page in self.pages]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProjectData":
        settings = payload.get("settings", {})
        if not isinstance(settings, dict):
            settings = {}

        current_page_index = payload.get("current_page_index", 0)
        if not isinstance(current_page_index, int):
            current_page_index = 0

        name = payload.get("name", "")
        if not isinstance(name, str) or not name.strip():
            name = "Untitled Project"

        pages = _load_pages(payload.get("pages"), payload.get("source_images"))

        return cls(
            name=name.strip(),
            pages=pages,
            current_page_index=current_page_index,
            settings=settings,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_images": self.source_images,
            "pages": [page.to_dict() for page in self.pages],
            "current_page_index": self.current_page_index,
            "settings": dict(self.settings),
        }


class MangaProject:
    """Represents a project folder and its JSON-backed UI state."""

    def __init__(self, root_dir: Path, data: ProjectData) -> None:
        self.root_dir = root_dir.resolve()
        self.data = data
        self.ensure_structure()
        self._normalize_state()

    @property
    def project_file(self) -> Path:
        return self.root_dir / PROJECT_FILENAME

    @property
    def source_dir(self) -> Path:
        return self.root_dir / "source"

    @property
    def cache_dir(self) -> Path:
        return self.root_dir / "cache"

    @property
    def page_count(self) -> int:
        return len(self.data.pages)

    @classmethod
    def create(cls, root_dir: Path, name: str | None = None) -> "MangaProject":
        project_name = name.strip() if isinstance(name, str) and name.strip() else root_dir.name
        project = cls(root_dir, ProjectData(name=project_name))
        project.save()
        return project

    @classmethod
    def load(cls, project_file: Path) -> "MangaProject":
        project_path = project_file.resolve()
        payload = json.loads(project_path.read_text(encoding="utf-8"))
        data = ProjectData.from_dict(payload)
        return cls(project_path.parent, data)

    def ensure_structure(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.source_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        for stage in CACHE_STAGES:
            (self.cache_dir / stage).mkdir(parents=True, exist_ok=True)

    def save(self) -> None:
        self._normalize_state()
        payload = self.data.to_dict()
        self.project_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def import_images(self, image_paths: Sequence[Path]) -> list[str]:
        imported_images: list[str] = []
        had_pages = self.page_count > 0

        for image_path in image_paths:
            source_path = Path(image_path)
            if not source_path.exists():
                continue

            if source_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                continue

            destination_name = self._build_unique_source_name(source_path.name)
            destination_path = self.source_dir / destination_name
            shutil.copy2(source_path, destination_path)

            relative_path = destination_path.relative_to(self.root_dir).as_posix()
            self.data.pages.append(ProjectPage(source_path=relative_path))
            imported_images.append(relative_path)

        if imported_images and not had_pages:
            self.data.current_page_index = 0

        self._normalize_state()
        return imported_images

    def page_display_names(self) -> list[str]:
        return [Path(page.source_path).name for page in self.data.pages]

    def page_relative_path_for_index(self, index: int) -> str | None:
        page = self.page_for_index(index)
        return page.source_path if page is not None else None

    def image_path_for_index(self, index: int) -> Path | None:
        relative_path = self.page_relative_path_for_index(index)
        if relative_path is None:
            return None

        return self.root_dir / relative_path

    def page_for_index(self, index: int) -> ProjectPage | None:
        if index < 0 or index >= self.page_count:
            return None
        return self.data.pages[index]

    def all_image_paths(self) -> list[Path]:
        return [self.root_dir / page.source_path for page in self.data.pages]

    def relative_source_path(self, image_path: Path | str) -> str | None:
        target_path = Path(image_path).resolve()

        try:
            relative_path = target_path.relative_to(self.root_dir)
        except ValueError:
            return None

        normalized = self._normalize_source_entry(relative_path.as_posix())
        for page in self.data.pages:
            if page.source_path == normalized:
                return normalized
        return None

    def stage_metadata(self, image_relative_path: str, stage_name: str) -> dict[str, Any] | None:
        page = self.page_for_source_path(image_relative_path)
        if page is None:
            return None
        return page.stages.get(stage_name)

    def update_stage_status(
        self,
        image_relative_path: str,
        stage_name: str,
        *,
        status: str,
        cache_path: str | None = None,
        error: str | None = None,
    ) -> None:
        page = self.page_for_source_path(image_relative_path)
        if page is None:
            return

        stage_payload: dict[str, Any] = {"status": status}
        if cache_path:
            stage_payload["cache_path"] = str(cache_path)
        if error:
            stage_payload["error"] = str(error)

        page.stages[stage_name] = stage_payload

    def page_for_source_path(self, image_relative_path: str) -> ProjectPage | None:
        normalized = self._normalize_source_entry(image_relative_path)
        for page in self.data.pages:
            if page.source_path == normalized:
                return page
        return None

    def set_current_page(self, index: int) -> None:
        self.data.current_page_index = index
        self._normalize_state()

    def _normalize_state(self) -> None:
        normalized_pages: list[ProjectPage] = []
        seen_source_paths: set[str] = set()

        for page in self.data.pages:
            normalized_source = self._normalize_source_entry(page.source_path)
            if normalized_source in seen_source_paths:
                continue
            seen_source_paths.add(normalized_source)
            normalized_pages.append(
                ProjectPage(
                    source_path=normalized_source,
                    stages=_normalize_stage_payload(page.stages),
                )
            )

        self.data.pages = normalized_pages

        if not self.data.pages:
            self.data.current_page_index = 0
            return

        self.data.current_page_index = max(0, min(self.data.current_page_index, self.page_count - 1))

    def _normalize_source_entry(self, source_image: str) -> str:
        relative_path = Path(source_image)

        if relative_path.is_absolute():
            try:
                relative_path = relative_path.relative_to(self.root_dir)
            except ValueError:
                relative_path = Path("source") / relative_path.name
        elif not relative_path.parts or relative_path.parts[0] != "source":
            relative_path = Path("source") / relative_path.name

        return relative_path.as_posix()

    def _build_unique_source_name(self, original_name: str) -> str:
        original_path = Path(original_name)
        stem = original_path.stem
        suffix = original_path.suffix
        candidate_name = original_path.name
        counter = 1

        while (self.source_dir / candidate_name).exists():
            candidate_name = f"{stem}_{counter}{suffix}"
            counter += 1

        return candidate_name


def _load_pages(
    raw_pages: Any,
    raw_source_images: Any,
) -> list[ProjectPage]:
    pages: list[ProjectPage] = []

    if isinstance(raw_pages, list):
        for raw_page in raw_pages:
            if isinstance(raw_page, dict):
                page = ProjectPage.from_dict(raw_page)
                if page is not None:
                    pages.append(page)

    if pages:
        return pages

    source_images: list[str] = []
    if isinstance(raw_source_images, list):
        source_images = [
            str(item)
            for item in raw_source_images
            if isinstance(item, str) and item.strip()
        ]

    return [ProjectPage(source_path=source_path) for source_path in source_images]


def _normalize_stage_payload(stages: dict[str, Any]) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}

    for stage_name, stage_payload in stages.items():
        if not isinstance(stage_name, str):
            continue

        if isinstance(stage_payload, dict):
            normalized[stage_name] = {
                str(key): value for key, value in stage_payload.items()
            }
        else:
            normalized[stage_name] = {}

    return normalized
