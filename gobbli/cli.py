import subprocess
from pathlib import Path

import click

INTERACTIVE_DIR = Path(__file__).parent / "interactive"


@click.group()
def main():
    pass


@main.command("explore")
@click.argument("data", type=str)
def main_explore(data):
    subprocess.check_call(
        ["streamlit", "run", str(INTERACTIVE_DIR / "explore.py"), data]
    )


if __name__ == "__main__":
    main()
