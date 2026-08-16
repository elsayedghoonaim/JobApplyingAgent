"""Focused tests for safe artifact paths and bounded collision-resistant identifier sanitization."""

import pytest

from jobapply.utils.paths import (
    MAX_COMPONENT_LENGTH,
    get_cover_letter_path,
    get_edited_resume_path,
    get_error_screenshot_path,
    get_run_output_dir,
    get_safe_artifact_path,
    sanitize_component,
    validate_run_id,
)


def test_sanitize_component_preserves_already_safe():
    assert sanitize_component("job123") == "job123"
    assert sanitize_component("run-uuid-1234-abcd") == "run-uuid-1234-abcd"
    assert sanitize_component("normal_job_12345") == "normal_job_12345"


def test_sanitize_component_collision_resistance_for_unsafe_inputs():
    input1 = "../job"
    input2 = "../../job"
    input3 = r"C:\data\job"
    input4 = "job/job"
    input5 = "job\x001"
    input6 = "job\x011"
    input7 = "job"

    results = {
        sanitize_component(inp) for inp in (input1, input2, input3, input4, input5, input6, input7)
    }
    assert len(results) == 7


def test_sanitize_component_long_inputs_and_extensions():
    # 1000-character input
    long_input = "a" * 1000
    res1 = sanitize_component(long_input)
    assert len(res1) <= MAX_COMPONENT_LENGTH
    assert res1.startswith("a")
    assert "_h" in res1

    # 1000-character extension
    long_ext_input = "document." + ("ext" * 300)
    res2 = sanitize_component(long_ext_input)
    assert len(res2) <= MAX_COMPONENT_LENGTH
    assert "_h" in res2

    # Multi-dot filenames
    multidot = "archive.tar.gz.backup.1"
    res3 = sanitize_component(multidot)
    assert len(res3) <= MAX_COMPONENT_LENGTH
    assert "_h" in res3

    # Long unsafe distinct inputs do not collide
    distinct_long1 = "prefix_" + ("x" * 500) + "_1"
    distinct_long2 = "prefix_" + ("x" * 500) + "_2"
    assert sanitize_component(distinct_long1) != sanitize_component(distinct_long2)
    assert len(sanitize_component(distinct_long1)) <= MAX_COMPONENT_LENGTH
    assert len(sanitize_component(distinct_long2)) <= MAX_COMPONENT_LENGTH


@pytest.mark.parametrize(
    "untrusted,expected_clean_stem,expected_ext",
    [
        ("../../../etc/passwd", "passwd", ""),
        (r"..\..\..\Windows\System32\cmd.exe", "cmd", ".exe"),
        (r"C:\Windows\System32\calc.exe", "calc", ".exe"),
        ("/var/log/syslog", "syslog", ""),
        (r"\\server\share\file.txt", "file", ".txt"),
        ("CON", "safe_CON", ""),
        ("NUL", "safe_NUL", ""),
        ("AUX.txt", "safe_AUX", ".txt"),
        ("COM1", "safe_COM1", ""),
        ("LPT3.pdf", "safe_LPT3", ".pdf"),
        ("test\x00run\x1f_id", "testrun_id", ""),
        ("..", "item", ""),
        (".", "item", ""),
        ("", "item", ""),
        ("   ", "item", ""),
    ],
)
def test_sanitize_component_edge_cases(untrusted, expected_clean_stem, expected_ext):
    result = sanitize_component(untrusted)
    assert len(result) <= MAX_COMPONENT_LENGTH
    assert "/" not in result
    assert "\\" not in result
    assert ":" not in result
    assert "\x00" not in result
    assert expected_clean_stem in result
    assert result.endswith(expected_ext)
    assert "_h" in result


def test_validate_run_id_valid():
    assert validate_run_id("abc-123_XYZ") == "abc-123_XYZ"
    assert (
        validate_run_id("d2c9778f-11df-4a92-9f1b-904b85bd89bc")
        == "d2c9778f-11df-4a92-9f1b-904b85bd89bc"
    )


@pytest.mark.parametrize(
    "invalid_id",
    [
        "../traversal",
        r"..\traversal",
        "foo/bar",
        r"foo\bar",
        "C:run",
        "CON",
        "NUL",
        "COM1",
        "run with spaces",
        "run@id!",
        "run\x00id",
        "a" * 200,
        "",
    ],
)
def test_validate_run_id_rejects_unsafe(invalid_id):
    with pytest.raises(ValueError) as excinfo:
        validate_run_id(invalid_id)
    assert "run_id" in str(excinfo.value).lower() or "run id" in str(excinfo.value).lower()


def test_get_run_output_dir_containment(tmp_path):
    base = tmp_path / "outputs"
    run_dir = get_run_output_dir("../../escaped_run", base_dir=base)
    assert run_dir.is_relative_to(base)
    assert run_dir.exists()


def test_get_safe_artifact_path_containment(tmp_path):
    base = tmp_path / "outputs"
    path = get_safe_artifact_path(
        run_id="../../../run1",
        filename="../../../secret.pdf",
        subfolder="../../../sub",
        base_dir=base,
    )
    assert path.is_relative_to(base)
    assert path.name.endswith(".pdf")
    assert len(path.name) <= MAX_COMPONENT_LENGTH
    assert path.exists() or path.parent.exists()


def test_artifact_helpers_resolve_under_outputs(tmp_path):
    base = tmp_path / "outputs"
    run_id = "test-run-123"
    job_id = "../job-456"
    label = r"..\..\step_error"

    cl_path = get_cover_letter_path(run_id, job_id, base_dir=base)
    assert cl_path.is_relative_to(base)
    assert cl_path.name.startswith("cover_letter_")
    assert cl_path.name.endswith(".txt")
    assert len(cl_path.name) <= MAX_COMPONENT_LENGTH

    resume_path = get_edited_resume_path(run_id, job_id, base_dir=base)
    assert resume_path.is_relative_to(base)
    assert resume_path.name.startswith("edited_resume_")
    assert resume_path.name.endswith(".pdf")
    assert len(resume_path.name) <= MAX_COMPONENT_LENGTH

    screenshot_path = get_error_screenshot_path(run_id, label, base_dir=base)
    assert screenshot_path.is_relative_to(base)
    assert screenshot_path.name.endswith(".png")
    assert screenshot_path.parent.name == "errors"
    assert len(screenshot_path.name) <= MAX_COMPONENT_LENGTH
