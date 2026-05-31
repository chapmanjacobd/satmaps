import os
import subprocess
import sys
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
    assert (install_root / "templates" / "index.html").is_file()

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
