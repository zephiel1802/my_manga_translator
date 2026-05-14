from __future__ import annotations

import inspect
import math
from typing import Callable


def parse_translation_batch_size(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        if not cleaned.isdigit():
            return None
        parsed = int(cleaned)
        return parsed if parsed > 0 else None
    return None


def build_page_chunks(
    page_names: list[str],
    batch_size: int | None,
    default_size: int = 5,
) -> list[list[str]]:
    if not page_names:
        return []

    if batch_size is not None:
        return [
            page_names[index:index + batch_size]
            for index in range(0, len(page_names), batch_size)
            if page_names[index:index + batch_size]
        ]

    target_size = max(1, int(default_size))
    chunk_count = max(1, math.ceil(len(page_names) / target_size))
    base_size = len(page_names) // chunk_count
    remainder = len(page_names) % chunk_count

    chunks: list[list[str]] = []
    cursor = 0
    for chunk_index in range(chunk_count):
        current_size = base_size + (1 if chunk_index < remainder else 0)
        chunk = page_names[cursor:cursor + current_size]
        if chunk:
            chunks.append(chunk)
        cursor += current_size

    return chunks


def validate_page_translation(
    page_name: str,
    original_lines: list[str],
    translated_lines,
) -> bool:
    if not isinstance(translated_lines, list):
        return False
    if len(translated_lines) != len(original_lines):
        return False
    try:
        for item in translated_lines:
            str(item)
    except Exception:
        return False
    return True


def split_valid_and_failed_pages(
    original_chunk: dict[str, list[str]],
    translated_chunk,
) -> tuple[dict[str, list[str]], list[str]]:
    valid_pages: dict[str, list[str]] = {}
    failed_pages: list[str] = []

    if not isinstance(translated_chunk, dict):
        return {}, list(original_chunk.keys())

    for page_name, original_lines in original_chunk.items():
        if page_name not in translated_chunk:
            failed_pages.append(page_name)
            continue
        translated_lines = translated_chunk[page_name]
        if not validate_page_translation(page_name, original_lines, translated_lines):
            failed_pages.append(page_name)
            continue
        valid_pages[page_name] = [str(item) for item in translated_lines]

    return valid_pages, failed_pages


def _filter_supported_kwargs(callable_obj, kwargs: dict) -> dict:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return kwargs

    supported = {}
    for key, value in kwargs.items():
        if key in signature.parameters:
            supported[key] = value
    return supported


def _call_translate_pages_batch(
    translator,
    pages_texts: dict[str, list[str]],
    *,
    source: str,
    target: str,
    custom_prompt: str | None,
    context_memory=None,
    strict: bool,
):
    kwargs = {
        "source": source,
        "target": target,
        "custom_prompt": custom_prompt,
        "context_memory": context_memory,
    }
    if strict:
        kwargs["allow_internal_fallback"] = False
        kwargs["repair_shape"] = False

    filtered_kwargs = _filter_supported_kwargs(
        translator.translate_pages_batch,
        kwargs,
    )
    return translator.translate_pages_batch(
        pages_texts,
        **filtered_kwargs,
    )


def _call_translate_batch(
    translator,
    texts: list[str],
    *,
    source: str,
    target: str,
    custom_prompt: str | None,
):
    kwargs = _filter_supported_kwargs(
        translator.translate_batch,
        {
            "source": source,
            "target": target,
            "custom_prompt": custom_prompt,
        },
    )
    return translator.translate_batch(texts, **kwargs)


def _repair_translation_list(
    translated_lines,
    original_lines: list[str],
) -> list[str]:
    if not isinstance(translated_lines, list):
        translated_lines = []

    repaired = [str(item) for item in translated_lines[:len(original_lines)]]
    while len(repaired) < len(original_lines):
        repaired.append(original_lines[len(repaired)])
    return repaired


def translate_pages_chunked_with_recovery(
    translator,
    pages_texts: dict[str, list[str]],
    *,
    source: str,
    target: str,
    custom_prompt: str | None = None,
    context_memory=None,
    batch_size: int | None = 5,
    logger=print,
    chunk_callback: Callable[[int, int, list[str], str], None] | None = None,
) -> dict[str, list[str]]:
    if not pages_texts:
        return {}

    ordered_page_names = list(pages_texts.keys())
    parsed_batch_size = parse_translation_batch_size(batch_size)
    chunks = build_page_chunks(
        ordered_page_names,
        parsed_batch_size,
        default_size=5,
    )

    if logger:
        if parsed_batch_size is not None:
            logger(f"[Translation] Batch size: {parsed_batch_size} pages")
        else:
            logger("[Translation] Batch size: balanced default 5 pages")
        logger(f"[Translation] Chunks: {len(chunks)}")

    translated_results: dict[str, list[str]] = {}

    for chunk_index, chunk_names in enumerate(chunks, start=1):
        if chunk_callback is not None:
            chunk_callback(chunk_index, len(chunks), list(chunk_names), "start")
        if logger:
            logger(
                f"[Translation] Chunk {chunk_index}/{len(chunks)}: "
                + ", ".join(chunk_names)
            )

        original_chunk = {name: pages_texts[name] for name in chunk_names}
        translated_chunk = None
        valid_pages: dict[str, list[str]] = {}
        failed_pages: list[str] = []

        try:
            translated_chunk = _call_translate_pages_batch(
                translator,
                original_chunk,
                source=source,
                target=target,
                custom_prompt=custom_prompt,
                context_memory=context_memory,
                strict=True,
            )
        except Exception as exc:
            if logger:
                logger(
                    f"[Translation] Chunk {chunk_index} strict batch failed: {exc}"
                )

        valid_pages, failed_pages = split_valid_and_failed_pages(
            original_chunk,
            translated_chunk,
        )

        if logger:
            if not failed_pages:
                logger(
                    f"[Translation] Chunk {chunk_index} OK: "
                    f"{len(valid_pages)}/{len(chunk_names)} pages"
                )
            else:
                logger(
                    f"[Translation] Chunk {chunk_index} partial: "
                    f"valid={len(valid_pages)} failed={len(failed_pages)}"
                )

        final_chunk_translations = dict(valid_pages)

        for failed_page_name in failed_pages:
            original_lines = original_chunk[failed_page_name]
            recovered_lines = None

            if logger:
                logger(
                    f"[Translation] Recovering page {failed_page_name} as single-page batch..."
                )

            try:
                single_page_result = _call_translate_pages_batch(
                    translator,
                    {failed_page_name: original_lines},
                    source=source,
                    target=target,
                    custom_prompt=custom_prompt,
                    context_memory=context_memory,
                    strict=True,
                )
                single_valid, _single_failed = split_valid_and_failed_pages(
                    {failed_page_name: original_lines},
                    single_page_result,
                )
                recovered_lines = single_valid.get(failed_page_name)
                if recovered_lines is not None and logger:
                    logger(f"[Translation] Page {failed_page_name} recovered OK")
            except Exception as exc:
                if logger:
                    logger(
                        f"[Translation] Page {failed_page_name} single-page batch failed: {exc}"
                    )

            if recovered_lines is None:
                try:
                    fallback_lines = _call_translate_batch(
                        translator,
                        original_lines,
                        source=source,
                        target=target,
                        custom_prompt=custom_prompt,
                    )
                    if validate_page_translation(
                        failed_page_name,
                        original_lines,
                        fallback_lines,
                    ):
                        recovered_lines = [str(item) for item in fallback_lines]
                        if logger:
                            logger(
                                f"[Translation] Page {failed_page_name} recovered via translate_batch"
                            )
                    else:
                        recovered_lines = _repair_translation_list(
                            fallback_lines,
                            original_lines,
                        )
                        if logger:
                            logger(
                                f"[Translation] Page {failed_page_name} fallback repaired shape with originals"
                            )
                except Exception as exc:
                    recovered_lines = _repair_translation_list([], original_lines)
                    if logger:
                        logger(
                            f"[Translation] Page {failed_page_name} translate_batch failed: {exc}"
                        )
                        logger(
                            f"[Translation] Page {failed_page_name} fallback repaired shape with originals"
                        )

            final_chunk_translations[failed_page_name] = recovered_lines

        ordered_chunk_translations = {
            page_name: final_chunk_translations.get(page_name, list(original_chunk[page_name]))
            for page_name in chunk_names
        }
        translated_results.update(ordered_chunk_translations)
        if chunk_callback is not None:
            chunk_callback(chunk_index, len(chunks), list(chunk_names), "done")

        if context_memory is not None:
            try:
                context_memory.update_from_translation(
                    original_chunk,
                    ordered_chunk_translations,
                )
                if logger and hasattr(context_memory, "get_stats"):
                    stats = context_memory.get_stats()
                    logger(
                        f"[Translation] Context updated: "
                        f"{stats.get('tracked_words', 0)} terms tracked, "
                        f"{stats.get('recent_pages', 0)} pages in memory"
                    )
            except Exception as exc:
                if logger:
                    logger(f"[Translation] Context memory update failed: {exc}")

    return {
        page_name: translated_results.get(page_name, list(pages_texts[page_name]))
        for page_name in ordered_page_names
    }


__all__ = [
    "build_page_chunks",
    "parse_translation_batch_size",
    "split_valid_and_failed_pages",
    "translate_pages_chunked_with_recovery",
    "validate_page_translation",
]
