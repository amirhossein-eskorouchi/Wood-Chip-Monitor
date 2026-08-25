from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_LFS = {
    "models/artifacts/detr_resnet101.onnx": {
        "oid": "0913062901c536dddae2de3ac98c8c0fa66964b76a67a72f226096a026a3720f",
        "size": 247214030,
    },
    "models/artifacts/detr_resnet101_simplified.onnx": {
        "oid": "6b01504525e653c2fe172c4788603ce5007c5a11b1d4e5289941facc214a2286",
        "size": 242602950,
    },
    "models/artifacts/moistnetlite_best_weights.h5": {
        "oid": "f52c6aa35ed82772b93c3c40806b631eba12214dc053362e4d7fe5b020a6eacf",
        "size": 1190104,
    },
    "models/artifacts/moistnetlite_dynamic.onnx": {
        "oid": "9c95d9a5155da373fb1a09d341be0ce9dde73153e68e82d57067e95e479546be",
        "size": 1164470,
    },
    "models/artifacts/moistnetlite_dynamic_simplified.onnx": {
        "oid": "f85873ee5c2cef714d1b8439f7ef576deb68332e284323f06f04ff90d3e2c71d",
        "size": 1165592,
    },
}

EXPECTED_CAD = {
    "hardware/cad/7_inch_LCD_CASE.SLDPRT",
    "hardware/cad/CAMERA_HOLDER.SLDPRT",
    "hardware/cad/Camera_Lock_System.SLDPRT",
    "hardware/cad/Handle.SLDPRT",
    "hardware/cad/Hook.SLDPRT",
    "hardware/cad/Intergrated_camera.SLDPRT",
    "hardware/cad/Jetson_nano_case.SLDPRT",
    "hardware/cad/NEW ASSEMBLY.SLDASM",
    "hardware/cad/port_lid.SLDPRT",
}

REQUIRED_FILES = {
    ".gitattributes",
    ".github/workflows/ci.yml",
    ".gitignore",
    "ACKNOWLEDGMENTS.md",
    "CHANGELOG.md",
    "CITATION.bib",
    "CITATION.cff",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "NOTICE",
    "README.md",
    "SECURITY.md",
    "app/__init__.py",
    "app/backend_app.py",
    "app/live_cam_trt.py",
    "app/web/index.html",
    "configs/device.env.example",
    "docs/CITATION.md",
    "docs/RELEASE.md",
    "docs/architecture.md",
    "docs/jetson-setup.md",
    "docs/related-research.md",
    "docs/software-environment.md",
    "docs/user-guide.md",
    "hardware/README.md",
    "hardware/cad/README.md",
    "models/README.md",
    "models/artifacts/moistnetlite_classes.txt",
    "models/detr_model.py",
    "models/moistnetlite.py",
    "scripts/audit_release.py",
    "tools/README.md",
    "tools/infer_trt.py",
    "validation/README.md",
    "validation/jetson_tensorrt.csv",
    "validation/manual_processing.csv",
    "validation/reference_processing.csv",
    "validation/size_statistics.csv",
}

FORBIDDEN_SUFFIXES = {
    ".7z",
    ".bak",
    ".ckpt",
    ".db",
    ".doc",
    ".docx",
    ".engine",
    ".key",
    ".log",
    ".npz",
    ".pdf",
    ".pem",
    ".pickle",
    ".pkl",
    ".plan",
    ".ppt",
    ".pptx",
    ".pyc",
    ".pyo",
    ".rar",
    ".sqlite",
    ".sqlite3",
    ".tar",
    ".tmp",
    ".trt",
    ".zip",
}

TEXT_SUFFIXES = {
    "",
    ".cff",
    ".cfg",
    ".conf",
    ".csv",
    ".env",
    ".example",
    ".gitattributes",
    ".gitignore",
    ".html",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

PRIVATE_PATTERNS = {
    "concrete Windows user path": re.compile(
        r"[A-Za-z]:\\Users\\[^\\\r\n]+\\",
        re.IGNORECASE,
    ),
    "MSU OneDrive path": re.compile(
        r"OneDrive\s*-\s*Mississippi State University",
        re.IGNORECASE,
    ),
    "private HPC path": re.compile(
        r"/home/(?:pa526|ae1028)(?:/|$)",
        re.IGNORECASE,
    ),
    "workstation identifier": re.compile(
        r"\b(?:ae1028|pa526)\b",
        re.IGNORECASE,
    ),
    "historical private model path": re.compile(
        r"F:\\3Spring 2025",
        re.IGNORECASE,
    ),
}

SECRET_PATTERNS = {
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    "OpenAI API key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "private key block": re.compile(r"BEGIN [A-Z ]*PRIVATE KEY"),
}

AUDIT_SOURCE_EXCLUSIONS = {
    "scripts/audit_release.py",
}


class AuditFailure(RuntimeError):
    pass


def run_git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        raise AuditFailure(
            "Git command failed: git "
            + " ".join(args)
            + ("\n" + message if message else "")
        )

    return result.stdout


def candidate_files() -> list[str]:
    output = run_git(
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
    )

    paths = {
        line.strip().replace("\\", "/")
        for line in output.splitlines()
        if line.strip()
    }

    return sorted(paths)


def read_text(relative_path: str) -> str:
    path = ROOT / relative_path

    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8-sig")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditFailure(message)


def print_ok(message: str) -> None:
    print("[OK] " + message)


def validate_required_files(files: set[str]) -> None:
    missing = sorted(REQUIRED_FILES - files)
    require(not missing, "Missing required files: " + ", ".join(missing))

    for path in REQUIRED_FILES:
        require((ROOT / path).is_file(), "Required path is not a file: " + path)

    print_ok(f"Required release files: {len(REQUIRED_FILES)}")


def validate_forbidden_files(files: Iterable[str]) -> None:
    findings: list[str] = []

    for path in files:
        lowered = path.lower()
        suffix = Path(path).suffix.lower()

        if "__pycache__/" in lowered:
            findings.append(path)
            continue

        if "/.pytest_cache/" in "/" + lowered:
            findings.append(path)
            continue

        if lowered.endswith("/.env") or lowered == ".env":
            findings.append(path)
            continue

        if lowered == "configs/device.env":
            findings.append(path)
            continue

        if suffix in FORBIDDEN_SUFFIXES:
            findings.append(path)

    require(
        not findings,
        "Forbidden cache, runtime, secret, archive, manuscript, or binary files: "
        + ", ".join(sorted(findings)),
    )

    print_ok("No forbidden cache, runtime, secret, archive, or manuscript files.")


def validate_ordinary_sizes(files: Iterable[str]) -> None:
    findings: list[str] = []

    for relative_path in files:
        if relative_path in EXPECTED_LFS:
            continue

        path = ROOT / relative_path

        if not path.is_file():
            continue

        size = path.stat().st_size

        if size > 25 * 1024 * 1024:
            findings.append(f"{relative_path} ({size} bytes)")

    require(
        not findings,
        "Ordinary files larger than 25 MB: " + ", ".join(findings),
    )

    print_ok("No ordinary release file exceeds 25 MB.")


def validate_lfs(files: set[str]) -> None:
    attributes = read_text(".gitattributes")

    require(
        "models/artifacts/*.onnx filter=lfs" in attributes,
        "ONNX Git LFS rule is missing.",
    )

    require(
        "models/artifacts/*.h5 filter=lfs" in attributes,
        "HDF5 Git LFS rule is missing.",
    )

    for relative_path, expected in EXPECTED_LFS.items():
        require(relative_path in files, "Missing LFS artifact: " + relative_path)

        committed = run_git("show", "HEAD:" + relative_path)

        lines = [
            line.strip()
            for line in committed.splitlines()
            if line.strip()
        ]

        require(
            lines
            and lines[0] == "version https://git-lfs.github.com/spec/v1",
            "Invalid Git LFS pointer: " + relative_path,
        )

        expected_oid = "oid sha256:" + str(expected["oid"])
        expected_size = "size " + str(expected["size"])

        require(
            expected_oid in lines,
            "Unexpected Git LFS SHA-256: " + relative_path,
        )

        require(
            expected_size in lines,
            "Unexpected Git LFS size: " + relative_path,
        )

    print_ok(f"Git LFS model pointers validated: {len(EXPECTED_LFS)}")


def validate_cad(files: set[str]) -> None:
    missing = sorted(EXPECTED_CAD - files)
    require(not missing, "Missing CAD files: " + ", ".join(missing))

    for relative_path in EXPECTED_CAD:
        require((ROOT / relative_path).is_file(), "CAD file missing: " + relative_path)

    print_ok(f"SolidWorks CAD files validated: {len(EXPECTED_CAD)}")


def validate_license_notice() -> None:
    license_text = read_text("LICENSE")
    notice_text = read_text("NOTICE")

    require(
        "Permission is hereby granted, free of charge" in license_text,
        "MIT permission paragraph is missing.",
    )

    require(
        'THE SOFTWARE IS PROVIDED "AS IS"' in license_text,
        "MIT warranty disclaimer is missing.",
    )

    required_notice_sections = {
        "## Maintained software and documentation",
        "## Trained model artifacts",
        "## Mechanical CAD files",
        "## Datasets and example media",
        "## Publications",
        "## Third-party components",
        "## Research-use boundary",
    }

    for section in required_notice_sections:
        require(section in notice_text, "NOTICE section missing: " + section)

    print_ok("MIT License and NOTICE boundaries validated.")


def validate_citation() -> None:
    cff = read_text("CITATION.cff")
    bib = read_text("CITATION.bib")

    cff_patterns = {
        r"(?m)^cff-version:\s*1\.2\.0\s*$",
        r"(?m)^version:\s*0\.1\.0\s*$",
        r"(?m)^date-released:\s*2026-08-25\s*$",
        r"(?m)^license:\s*MIT\s*$",
        r"https://github\.com/amirhossein-eskorouchi/Wood-Chip-Monitor",
        r"An Edge Artificial Intelligence System for Wood Chip Quality Evaluation",
    }

    for pattern in cff_patterns:
        require(re.search(pattern, cff) is not None, "CFF field missing: " + pattern)

    bib_patterns = {
        r"@software\{eskorouchi_wood_chip_monitor_2026",
        r"@inproceedings\{eskorouchi_edge_ai_wood_chip_2026",
        r"@article\{rahman_moistnet_2025",
        r"@dataset\{eskorouchi_woodchip_detection_2026",
        r"10\.1016/j\.eswa\.2024\.125363",
        r"10\.5281/zenodo\.18392693",
    }

    for pattern in bib_patterns:
        require(re.search(pattern, bib) is not None, "BibTeX field missing: " + pattern)

    print_ok("Version 0.1.0 citation metadata validated.")


def validate_readme() -> None:
    readme = read_text("README.md")

    required_sections = {
        "## Overview",
        "## Quick Start",
        "## Documentation",
        "## Related Research",
        "## Data and Runtime Privacy",
        "## Project Status",
        "## Citation",
        "## License",
        "## Security",
        "## Contributing",
        "## Acknowledgments",
    }

    for section in required_sections:
        require(section in readme, "README section missing: " + section)

    require(
        readme.count("## Acknowledgments") == 1,
        "README must contain one Acknowledgments section.",
    )

    unfinished = {
        "Citation metadata will be added before the public release.",
        "A source-code license will be added before the public release.",
        "The repository will remain under preparation until the public release materials are validated.",
        "will be finalized with the public release",
    }

    for phrase in unfinished:
        require(phrase.lower() not in readme.lower(), "Unfinished README text: " + phrase)

    print_ok("README public-release sections validated.")


def validate_readme_links() -> None:
    text = read_text("README.md")
    targets = re.findall(r"\[[^\]]+\]\(([^)]+)\)", text)
    broken: list[str] = []

    for target in targets:
        target = target.strip()

        if not target or target.startswith(("http://", "https://", "mailto:", "#")):
            continue

        local_target = target.split("#", 1)[0]

        if not local_target:
            continue

        if not (ROOT / local_target).exists():
            broken.append(target)

    require(not broken, "Broken README links: " + ", ".join(sorted(set(broken))))
    print_ok("All README local links resolve.")


def validate_privacy_and_secrets(files: Iterable[str]) -> None:
    findings: list[str] = []

    for relative_path in files:
        if relative_path in AUDIT_SOURCE_EXCLUSIONS:
            continue

        path = ROOT / relative_path

        if not path.is_file():
            continue

        suffix = path.suffix.lower()

        if suffix not in TEXT_SUFFIXES:
            continue

        try:
            text = read_text(relative_path)
        except (UnicodeDecodeError, OSError):
            continue

        for label, pattern in PRIVATE_PATTERNS.items():
            if pattern.search(text):
                findings.append(relative_path + ": " + label)

        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                findings.append(relative_path + ": " + label)

    require(
        not findings,
        "Private path or secret findings: " + "; ".join(sorted(set(findings))),
    )

    example = read_text("configs/device.env.example")

    require(
        re.search(
            r"(?m)^WOODCHIP_ADMIN_PASSWORD=change-this-password\s*$",
            example,
        )
        is not None,
        "Example administrator password must remain an explicit placeholder.",
    )

    print_ok("No targeted private path, workstation identifier, or concrete secret.")
    print_ok("Example administrator password is a documented placeholder.")


def validate_python_syntax(files: Iterable[str]) -> None:
    checked = 0
    failures: list[str] = []

    for relative_path in files:
        if not relative_path.endswith(".py"):
            continue

        path = ROOT / relative_path

        try:
            source = read_text(relative_path)
            compile(source, str(path), "exec")
            checked += 1
        except SyntaxError as error:
            failures.append(
                f"{relative_path}:{error.lineno}:{error.offset}: {error.msg}"
            )

    require(not failures, "Python syntax failures: " + "; ".join(failures))
    print_ok(f"Python syntax checked without imports: {checked} files")


def validate_git_whitespace() -> None:
    result = subprocess.run(
        ["git", "diff", "--check"],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    require(
        result.returncode == 0,
        "Git whitespace validation failed: "
        + (result.stdout.strip() or result.stderr.strip()),
    )

    print_ok("Git whitespace validation passed.")


def main() -> int:
    print("=" * 68)
    print("WOOD-CHIP MONITOR PUBLIC-RELEASE AUDIT")
    print("=" * 68)

    files = candidate_files()
    file_set = set(files)

    print_ok(f"Release-candidate files: {len(files)}")

    validate_required_files(file_set)
    validate_forbidden_files(files)
    validate_ordinary_sizes(files)
    validate_lfs(file_set)
    validate_cad(file_set)
    validate_license_notice()
    validate_citation()
    validate_readme()
    validate_readme_links()
    validate_privacy_and_secrets(files)
    validate_python_syntax(files)
    validate_git_whitespace()

    print("=" * 68)
    print("PUBLIC-RELEASE AUDIT RESULT")
    print("=" * 68)
    print("[OK] Wood-Chip Monitor release boundary passed.")
    print("[OK] Models and CAD remain preserved.")
    print("[OK] Repository is ready for final pre-commit review.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuditFailure as error:
        print("[FAIL] " + str(error))
        raise SystemExit(1)
