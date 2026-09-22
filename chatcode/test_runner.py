from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .context_builder import redact_sensitive_text
from .workspace import get_test_results_file


MAX_TEST_OUTPUT_CHARS = 100_000
TEST_PROFILE_FILE = Path(".chatcode") / "tests.toml"


class TestError(RuntimeError):
    pass


@dataclass(frozen=True)
class TestCommand:
    display: str
    args: list[str]


@dataclass(frozen=True)
class TestResult:
    command: str
    returncode: int
    duration_seconds: float
    output_file: Path
    failed_tests: frozenset[str] = frozenset()


def _configured_test_command(
    repo: Path,
    profile: str,
) -> TestCommand | None:
    config_file = repo / TEST_PROFILE_FILE
    if not config_file.exists():
        return None

    try:
        with config_file.open("rb") as stream:
            payload = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise TestError(
            f"Kunde inte läsa {TEST_PROFILE_FILE}: {exc}"
        ) from exc

    tests = payload.get("tests", {})
    if not isinstance(tests, dict):
        raise TestError(
            f"{TEST_PROFILE_FILE}: [tests] måste vara en tabell."
        )

    value = tests.get(profile)
    if value is None:
        return None

    if isinstance(value, str):
        display = value.strip()
        if not display:
            raise TestError(
                f"{TEST_PROFILE_FILE}: tests.{profile} får inte vara tomt."
            )
        lexer = shlex.shlex(display, posix=os.name != "nt")
        lexer.whitespace_split = True
        lexer.commenters = ""
        args = list(lexer)
        if os.name == "nt":
            args = [
                arg[1:-1]
                if len(arg) >= 2 and arg[0] == arg[-1] and arg[0] in {"'", '"'}
                else arg
                for arg in args
            ]
    elif (
        isinstance(value, list)
        and value
        and all(isinstance(arg, str) and arg for arg in value)
    ):
        args = list(value)
        display = (
            subprocess.list2cmdline(args)
            if os.name == "nt"
            else shlex.join(args)
        )
    else:
        raise TestError(
            f"{TEST_PROFILE_FILE}: tests.{profile} måste vara "
            "en kommandosträng eller en lista med argument."
        )

    if not args:
        raise TestError(
            f"{TEST_PROFILE_FILE}: tests.{profile} får inte vara tomt."
        )
    return TestCommand(display=display, args=args)


def _read_json(path: Path) -> dict:
    try:
        return json.loads(
            path.read_text(
                encoding="utf-8",
            )
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise TestError(
            f"Kunde inte läsa {path.name}: {exc}"
        ) from exc


def _detect_node_tests(
    repo: Path,
) -> TestCommand | None:
    package_file = repo / "package.json"

    if not package_file.exists():
        return None

    package = _read_json(package_file)
    scripts = package.get("scripts", {})

    if not isinstance(scripts, dict):
        return None

    test_script = scripts.get("test")

    if not isinstance(test_script, str):
        return None

    if not test_script.strip():
        return None

    if (repo / "pnpm-lock.yaml").exists():
        return TestCommand(
            display="pnpm test",
            args=["pnpm", "test"],
        )

    if (repo / "yarn.lock").exists():
        return TestCommand(
            display="yarn test",
            args=["yarn", "test"],
        )

    if (
        (repo / "bun.lock").exists()
        or (repo / "bun.lockb").exists()
    ):
        return TestCommand(
            display="bun run test",
            args=["bun", "run", "test"],
        )

    return TestCommand(
        display="npm test",
        args=["npm", "test"],
    )


def _detect_maven_tests(
    repo: Path,
) -> TestCommand | None:
    if not (repo / "pom.xml").exists():
        return None

    if os.name == "nt":
        wrapper = repo / "mvnw.cmd"
    else:
        wrapper = repo / "mvnw"

    if wrapper.exists():
        return TestCommand(
            display=f"{wrapper.name} test",
            args=[str(wrapper), "test"],
        )

    return TestCommand(
        display="mvn test",
        args=["mvn", "test"],
    )


def _detect_gradle_tests(
    repo: Path,
) -> TestCommand | None:
    has_gradle = (
        (repo / "build.gradle").exists()
        or (repo / "build.gradle.kts").exists()
    )

    if not has_gradle:
        return None

    if os.name == "nt":
        wrapper = repo / "gradlew.bat"
    else:
        wrapper = repo / "gradlew"

    if wrapper.exists():
        return TestCommand(
            display=f"{wrapper.name} test",
            args=[str(wrapper), "test"],
        )

    return TestCommand(
        display="gradle test",
        args=["gradle", "test"],
    )


def _is_python_project(
    repo: Path,
) -> bool:
    markers = (
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "requirements.txt",
        "Pipfile",
        "tox.ini",
        "pytest.ini",
        "conftest.py",
    )

    if any((repo / marker).exists() for marker in markers):
        return True

    try:
        return any(repo.glob("*.py"))
    except OSError:
        return False


def _config_mentions_pytest(
    path: Path,
) -> bool:
    if not path.exists():
        return False

    try:
        content = path.read_text(
            encoding="utf-8",
            errors="replace",
        ).lower()
    except OSError:
        return False

    return "pytest" in content


def _uses_pytest(
    repo: Path,
) -> bool:
    if (repo / "pytest.ini").exists():
        return True

    if (repo / "conftest.py").exists():
        return True

    for config_name in (
        "setup.cfg",
        "tox.ini",
    ):
        if _config_mentions_pytest(
            repo / config_name
        ):
            return True

    pyproject = repo / "pyproject.toml"

    if not pyproject.exists():
        return False

    try:
        content = pyproject.read_text(
            encoding="utf-8",
            errors="replace",
        ).lower()
    except OSError:
        return False

    return (
        "[tool.pytest" in content
        or "pytest" in content
    )


def _project_python(
    repo: Path,
) -> str:
    if os.name == "nt":
        candidates = [
            repo / ".venv" / "Scripts" / "python.exe",
            repo / "venv" / "Scripts" / "python.exe",
            repo / ".tools" / "python313" / "python.exe",
        ]
    else:
        candidates = [
            repo / ".venv" / "bin" / "python",
            repo / "venv" / "bin" / "python",
        ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    if os.name == "nt" and shutil.which("py"):
        return "py"

    if shutil.which("python"):
        return "python"

    return sys.executable


def _detect_python_tests(
    repo: Path,
) -> TestCommand | None:
    if not _is_python_project(repo):
        return None

    python = _project_python(repo)

    if _uses_pytest(repo):
        return TestCommand(
            display=f"{python} -m pytest -n auto",
            args=[python, "-m", "pytest", "-n", "auto"],
        )

    if not (repo / "tests").is_dir():
        return None

    return TestCommand(
        display=f"{python} -m unittest discover",
        args=[python, "-m", "unittest", "discover", "-s", "tests"],
    )


def _detect_composer_tests(
    repo: Path,
) -> TestCommand | None:
    composer_file = repo / "composer.json"

    if not composer_file.exists():
        return None

    composer = _read_json(composer_file)
    scripts = composer.get("scripts", {})

    if not isinstance(scripts, dict):
        return None

    if "test" not in scripts:
        return None

    return TestCommand(
        display="composer test",
        args=["composer", "test"],
    )


def detect_test_command(
    repo: Path,
) -> TestCommand:
    configured = _configured_test_command(repo, "full")
    if configured is not None:
        return configured

    detectors = [
        _detect_node_tests,
        _detect_maven_tests,
        _detect_gradle_tests,
        _detect_python_tests,
        _detect_composer_tests,
    ]

    for detector in detectors:
        command = detector(repo)

        if command is not None:
            return command

    raise TestError(
        "ChatCode kunde inte hitta något testkommando "
        "för det här projektet."
    )


def _prepare_command(
    command: TestCommand,
) -> list[str]:
    executable = command.args[0]

    executable_path = Path(executable)

    if executable_path.is_file():
        resolved = str(executable_path)
        command_name = resolved
    else:
        found = shutil.which(executable)

        if found is None:
            raise TestError(
                f"Testkommandot finns inte installerat: "
                f"{executable}"
            )

        resolved = found

        # Viktigt på Windows:
        # använd kommandonamnet istället för den absoluta
        # sökvägen när cmd.exe startar .cmd/.bat.
        command_name = executable

    if (
        os.name == "nt"
        and resolved.lower().endswith(
            (".cmd", ".bat")
        )
    ):
        return [
            "cmd.exe",
            "/d",
            "/c",
            command_name,
            *command.args[1:],
        ]

    return [
        resolved,
        *command.args[1:],
    ]


def _truncate_output(
    text: str,
) -> str:
    text = redact_sensitive_text(text)

    if len(text) <= MAX_TEST_OUTPUT_CHARS:
        return text

    return (
        "[Earlier test output truncated by ChatCode]\n\n"
        + text[-MAX_TEST_OUTPUT_CHARS:]
    )


def _indent_output(
    text: str,
) -> str:
    if not text.strip():
        return "    (no output)"

    return "\n".join(
        f"    {line}"
        for line in text.splitlines()
    )


def _write_test_report(
    repo: Path,
    command: TestCommand,
    returncode: int,
    duration_seconds: float,
    stdout: str,
    stderr: str,
    output_file: Path | None = None,
) -> Path:
    output_file = output_file or get_test_results_file(repo)

    safe_stdout = _truncate_output(stdout)
    safe_stderr = _truncate_output(stderr)

    status = (
        "PASSED"
        if returncode == 0
        else "FAILED"
    )

    timestamp = (
        datetime.now()
        .astimezone()
        .isoformat(timespec="seconds")
    )

    parts = [
        "# ChatCode Test Results",
        "",
        f"Status: {status}",
        f"Exit code: {returncode}",
        f"Command: {command.display}",
        f"Duration: {duration_seconds:.2f} seconds",
        f"Run at: {timestamp}",
        "",
        "## Standard output",
        "",
        _indent_output(safe_stdout),
        "",
        "## Standard error",
        "",
        _indent_output(safe_stderr),
        "",
    ]

    output_file.write_text(
        "\n".join(parts),
        encoding="utf-8",
    )

    return output_file


def _failed_test_ids(
    stdout: str,
    stderr: str,
) -> frozenset[str]:
    """Extract stable failure identifiers from common Python test runners.

    An empty result is intentional: a non-zero command without a recognizable
    test identifier must be reported as unclear, rather than guessed at.
    """
    failures: set[str] = set()
    for line in (stdout + "\n" + stderr).splitlines():
        unittest_match = re.match(
            r"^(?:FAIL|ERROR): .+ \((.+)\)$",
            line.strip(),
        )
        pytest_match = re.match(
            # pytest's summary line is e.g. ``FAILED (failures=1)``;
            # accept only node IDs, never that aggregate summary.
            r"^FAILED (\S+::\S+)",
            line.strip(),
        )
        match = unittest_match or pytest_match
        if match:
            failures.add(match.group(1))
    return frozenset(failures)


def _run_test_command(
    repo: Path,
    command: TestCommand,
    *,
    output_file: Path | None = None,
) -> TestResult:
    args = _prepare_command(command)

    started = time.perf_counter()

    try:
        process = subprocess.run(
            args,
            cwd=repo,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        raise TestError(
            f"Kunde inte starta testerna: {exc}"
        ) from exc

    duration = time.perf_counter() - started
    output_file = _write_test_report(
        repo=repo,
        command=command,
        returncode=process.returncode,
        duration_seconds=duration,
        stdout=process.stdout,
        stderr=process.stderr,
        output_file=output_file,
    )
    return TestResult(
        command=command.display,
        returncode=process.returncode,
        duration_seconds=duration,
        output_file=output_file,
        failed_tests=_failed_test_ids(process.stdout, process.stderr),
    )


def _relevant_python_test_files(
    repo: Path,
    paths: set[str],
) -> list[Path]:
    tests_root = repo / "tests"
    if not tests_root.is_dir():
        return []

    matches: set[Path] = set()
    for raw_path in paths:
        path = Path(raw_path.replace("\\", "/"))
        if path.parts and path.parts[0] == "tests" and path.name.startswith("test_"):
            candidate = repo / path
            if candidate.is_file():
                matches.add(candidate)
            continue

        candidate_name = f"test_{path.stem}.py"
        candidates = list(tests_root.rglob(candidate_name))
        if len(candidates) == 1:
            matches.add(candidates[0])

    return sorted(matches)


def unmapped_python_source_paths(
    repo: Path,
    changed_paths: set[str],
) -> list[str]:
    """Return changed Python source paths without a safe filename mapping.

    This is deliberately a warning heuristic, not a coverage assertion.
    """
    if not _is_python_project(repo):
        return []

    tests_root = repo / "tests"
    unmapped: list[str] = []
    for raw_path in sorted(changed_paths):
        normalized = raw_path.replace("\\", "/")
        path = Path(normalized)
        if path.suffix.lower() != ".py":
            continue
        if any(part.lower() in {"test", "tests"} for part in path.parts[:-1]):
            continue
        if (
            path.name.startswith("test_")
            or path.name.endswith("_test.py")
            or path.name in {"conftest.py", "setup.py"}
        ):
            continue

        candidates = (
            list(tests_root.rglob(f"test_{path.stem}.py"))
            if tests_root.is_dir()
            else []
        )
        if len(candidates) != 1:
            unmapped.append(normalized)

    return unmapped


def run_relevant_tests(
    repo: Path,
    changed_paths: set[str],
) -> TestResult | None:
    """Run the configured fast profile or a safe narrow Python selection.

    Unknown fallback mappings deliberately defer to the full-suite run.
    """
    configured = _configured_test_command(repo, "fast")
    if configured is not None:
        return _run_test_command(repo, configured)

    if not _is_python_project(repo):
        return None

    test_files = _relevant_python_test_files(repo, changed_paths)
    if not test_files:
        return None

    python = _project_python(repo)
    if _uses_pytest(repo):
        target_args = [path.relative_to(repo).as_posix() for path in test_files]
        targeted = TestCommand(
            display=(
                f"{python} -m pytest -n auto "
                f"{' '.join(target_args)}"
            ),
            args=[python, "-m", "pytest", "-n", "auto", *target_args],
        )
    elif len(test_files) == 1:
        targeted = TestCommand(
            display=f"{python} -m unittest discover -s tests -p {test_files[0].name}",
            args=[python, "-m", "unittest", "discover", "-s", "tests", "-p", test_files[0].name],
        )
    else:
        return None

    return _run_test_command(repo, targeted)


def run_project_tests(
    repo: Path,
    *,
    output_file: Path | None = None,
) -> TestResult:
    command = detect_test_command(repo)
    return _run_test_command(repo, command, output_file=output_file)
