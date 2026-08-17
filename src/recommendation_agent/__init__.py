"""Recommendation Agent package entrypoint — runs Part A on a sample profile."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    from config.settings import REPO_ROOT, get_settings
    from recommendation_graph import run_recommendation

    settings = get_settings()
    profiles_path = REPO_ROOT / "data" / "sample_profiles.json"
    profiles = json.loads(profiles_path.read_text(encoding="utf-8"))
    profile = profiles[0]
    catalog_path = Path(settings.catalog_path)
    if not catalog_path.is_absolute():
        catalog_path = REPO_ROOT / catalog_path

    result = run_recommendation(profile, catalog_path)
    print(json.dumps(
        {
            "profile": profile.get("id"),
            "validation_passed": result.get("validation_passed"),
            "validation_messages": result.get("validation_messages"),
            "ranked_items": result.get("ranked_items"),
            "narrative": result.get("narrative"),
            "errors": result.get("errors"),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
