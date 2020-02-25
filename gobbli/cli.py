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


@main.command(
    # Forward the --help argument to the streamlit apps
    "explore",
    context_settings=dict(ignore_unknown_options=True),
    add_help_option=False,
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def main_explore(args):
    _streamlit_run("explore", *args)


@main.command(
    "evaluate",
    context_settings=dict(ignore_unknown_options=True),
    add_help_option=False,
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def main_evaluate(args):
    _streamlit_run("evaluate", *args)


@main.command(
    "explain", context_settings=dict(ignore_unknown_options=True), add_help_option=False
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def main_explain(args):
    _streamlit_run("explain", *args)


if __name__ == "__main__":
    main()
