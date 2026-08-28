from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_review_console_loads_both_segmentation_versions() -> None:
    app_path = Path(__file__).parents[1] / "src/evalprobe/review/app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=30)
    assert not app.exception

    methodology = next(
        control for control in app.segmented_control if control.label == "Local-unit methodology"
    )
    methodology.set_value("sentence-v1")
    app.run(timeout=30)
    assert not app.exception

    methodology = next(
        control for control in app.segmented_control if control.label == "Local-unit methodology"
    )
    methodology.set_value("sentence-v2")
    app.run(timeout=30)
    assert not app.exception
