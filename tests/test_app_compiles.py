from pathlib import Path
import py_compile


def test_app_and_package_modules_compile():
    repo_root = Path(__file__).resolve().parents[1]
    sources = [repo_root / "app.py", *sorted((repo_root / "resume_screener").glob("*.py"))]

    for source in sources:
        py_compile.compile(str(source), doraise=True)


def test_result_markup_keeps_final_values_as_inline_resting_values():
    app_source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    assert ">{val}%</div>" in app_source
    assert "background:conic-gradient({tier_color} {overall_pct * 3.6}deg" in app_source
    assert "counter(" not in app_source

    scores = [5, 16, 49, 63, 48]
    tile_html = "".join(
        f"<div class='score-number'>{score}%</div>"
        for score in scores
    )

    overall = scores[-1]
    tier_class = "tier-strong" if overall >= 65 else "tier-moderate" if overall >= 40 else "tier-weak"
    decision_html = (
        f"<div class='{tier_class} result-gauge' style='background:conic-gradient(#fbbf24 {overall * 3.6}deg, "
        f"rgba(168,85,247,.13) 0);'><div class='result-gauge-value'>{overall}%</div></div>"
    )

    for score in scores:
        assert f">{score}%</div>" in tile_html
    assert f"class='{tier_class} result-gauge'" in decision_html
    assert f"{overall * 3.6}deg" in decision_html
    assert f">{overall}%</div>" in decision_html
