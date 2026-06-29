from __future__ import annotations

from pathlib import Path


def resolve_model_path(preferred_path: Path) -> Path:
    """새 구조(models/)와 기존 루트 폴더 양쪽을 모두 지원합니다."""
    preferred_path = Path(preferred_path)
    candidates = [
        preferred_path,
        preferred_path.parent.parent / preferred_path.name,
        Path.cwd() / preferred_path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return preferred_path
