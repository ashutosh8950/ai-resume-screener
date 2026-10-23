from resume_screener.text_utils import clean_jd_text


def test_clean_jd_text_replaces_bullets_and_removes_controls():
    source = "\uf0b7 Python\n\u2022 APIs\n\u25aa Testing\n\x00 Hidden\ue123 marker"

    assert clean_jd_text(source) == "-  Python\n-  APIs\n-  Testing\n Hidden marker"


def test_clean_jd_text_preserves_normal_whitespace_and_text():
    source = "Build models\n\tShip APIs\r\n"

    assert clean_jd_text(source) == source
