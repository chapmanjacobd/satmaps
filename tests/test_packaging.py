import os
import subprocess
import sys
import sysconfig
from pathlib import Path


def test_pip_install_target_includes_local_modules_and_templates(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    install_root = tmp_path / "install"
    install_root.mkdir()

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-deps",
            "--no-build-isolation",
            "--target",
            str(install_root),
            str(repo_root),
        ],
        check=True,
        cwd=repo_root,
    )

    assert (install_root / "common.py").is_file()
    assert (install_root / "satmaps_assets" / "templates" / "index.html").is_file()
    assert not (install_root / "templates" / "index.html").exists()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(install_root)

    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import satmaps; "
                "import tuner_ui; "
                "tuner_ui.app.jinja_env.get_template('index.html')"
            ),
        ],
        check=True,
        cwd=tmp_path,
        env=env,
    )


def test_pip_install_prefix_keeps_templates_in_site_packages(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    prefix_root = tmp_path / "prefix"
    purelib = Path(
        sysconfig.get_path(
            "purelib",
            vars={"base": str(prefix_root), "platbase": str(prefix_root)},
        )
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-deps",
            "--no-build-isolation",
            "--prefix",
            str(prefix_root),
            str(repo_root),
        ],
        check=True,
        cwd=repo_root,
    )

    assert (purelib / "common.py").is_file()
    assert (purelib / "satmaps_assets" / "templates" / "index.html").is_file()
    assert not (prefix_root / "templates" / "index.html").exists()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(purelib)

    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import satmaps; "
                "import tuner_ui; "
                "tuner_ui.app.jinja_env.get_template('index.html')"
            ),
        ],
        check=True,
        cwd=tmp_path,
        env=env,
    )
