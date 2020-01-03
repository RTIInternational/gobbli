import subprocess
from pathlib import Path

import click

INTERACTIVE_DIR = Path(__file__).parent / "interactive"


def _streamlit_run(app_name: str, *args):
    return subprocess.check_call(
        ["streamlit", "run", str(INTERACTIVE_DIR / f"{app_name}.py"), "--", *args]
    )


@click.group()
def main():
    pass


@main.command("explore", context_settings=dict(ignore_unknown_options=True))
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def main_explore(args):
    _streamlit_run("explore", *args)


@main.command("evaluate")
@click.argument("model_checkpoint", type=str)
@click.argument("data", type=str)
def main_evaluate(model_checkpoint, data):
    _streamlit_run("evaluate", model_checkpoint, data)


if __name__ == "__main__":
    main()
