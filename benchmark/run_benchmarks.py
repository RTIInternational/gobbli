import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import click
import yaml

import gobbli
from benchmark_util import BENCHMARK_DATA_DIR
from scenario import (
    ClassImbalanceScenario,
    DataAugmentationScenario,
    DocumentWindowingScenario,
    IMDBClassificationScenario,
    IMDBEmbeddingScenario,
    LowResourceScenario,
    MovieSummaryClassificationScenario,
    NewsgroupsClassificationScenario,
    NewsgroupsEmbeddingScenario,
    load_scenario,
)

LOGGER = logging.getLogger(__name__)

BENCHMARK_DIR = Path(__file__).parent

BENCHMARK_SPECS_FILE = BENCHMARK_DIR / "BENCHMARK_SPECS.yml"
BENCHMARK_OUTPUT_DIR = BENCHMARK_DIR / "benchmark_output"

BENCHMARK_DEBUG_OUTPUT_DIR = BENCHMARK_DIR / "benchmark_output_debug"
BENCHMARK_DEBUG_SPECS_FILE = BENCHMARK_DIR / "BENCHMARK_SPECS_DEBUG.yml"

ALL_SCENARIOS: Dict[str, Any] = {
    "newsgroups": NewsgroupsClassificationScenario,
    "newsgroups_embed": NewsgroupsEmbeddingScenario,
    "imdb": IMDBClassificationScenario,
    "moviesummary": MovieSummaryClassificationScenario,
    "imdb_embed": IMDBEmbeddingScenario,
    "class_imbalance": ClassImbalanceScenario,
    "low_resource": LowResourceScenario,
    "data_augmentation": DataAugmentationScenario,
    "document_windowing": DocumentWindowingScenario,
}


def load_specs(specs_file: Path) -> Dict[str, Any]:
    """
    Read the specifications for benchmark scenarios from the given file.

    Returns:
      A dict where keys are scenario names and values are scenario configuration.
    """
    with open(specs_file, "r") as f:
        specs_list = yaml.safe_load(f)

    specs: Dict[str, Any] = {}
    for spec in specs_list:
        name = spec.pop("scenario")
        specs[name] = spec

    return specs


@click.command("run")
@click.option(
    "--scenario",
    "scenario_names",
    multiple=True,
    help="Which benchmark scenarios to run.  If not passed, runs all of them.",
)
@click.option(
    "--force/--no-force",
    default=False,
    help="If force, rerun all benchmarks regardless of whether they've already been run.",
)
@click.option(
    "--output-dir",
    default=str(BENCHMARK_OUTPUT_DIR),
    help="Directory to save benchmark output in.",
)
@click.option(
    "--log-level",
    default="INFO",
    show_default=True,
    help="Verbosity of logging -- can be any value accepted by logging.setLevel.",
)
@click.option(
    "--debug/--no-debug",
    default=False,
    help="If --debug, use a minimal configuration for each scenario to facilitate debugging "
    "benchmark scenario implementations.  This will also shrink the dataset size to make "
    "scenarios run faster.",
)
@click.option(
    "--raise-exceptions/--catch-exceptions",
    default=False,
    help="Whether to raise exceptions which occur during benchmark runs or catch them and "
    "continue on to the next benchmark run.",
)
def run(
    scenario_names: List[str],
    force: bool,
    output_dir: str,
    log_level: str,
    debug: bool,
    raise_exceptions: bool,
):
    # Make sure all models run outside of experiments create their data under the
    # assigned benchmark directory
    os.environ["GOBBLI_DIR"] = str(BENCHMARK_DATA_DIR)

    logging.basicConfig(
        level=log_level, format="[%(asctime)s] %(levelname)s - %(name)s: %(message)s"
    )

    output_path = Path(output_dir)
    specs_file = BENCHMARK_SPECS_FILE
    dataset_limit = None
    if debug:
        output_path = BENCHMARK_DEBUG_OUTPUT_DIR
        specs_file = BENCHMARK_DEBUG_SPECS_FILE
        dataset_limit = 50

    output_path.mkdir(exist_ok=True, parents=True)
    specs = load_specs(specs_file)

    if len(scenario_names) == 0:
        scenario_names = list(specs.keys())

    LOGGER.info(f"Running scenarios: {scenario_names}")

    for scenario_name in scenario_names:
        LOGGER.info(f"Loading scenario: {scenario_name}")
        if scenario_name not in ALL_SCENARIOS:
            raise ValueError(
                f"Unknown scenario type: '{scenario_name}'. Valid values are: "
                f"{list(ALL_SCENARIOS.keys())}"
            )
        if scenario_name not in specs:
            raise ValueError(
                f"No spec for scenario named: '{scenario_name}'. Valid values are: "
                f"{list(specs.keys())}"
            )

        spec = specs[scenario_name]
        scenario = load_scenario(
            ALL_SCENARIOS[scenario_name],
            output_path / scenario_name,
            spec.get("params", {}),
            spec.get("runs", []),
            force=force,
            dataset_limit=dataset_limit,
        )

        LOGGER.info(f"Running scenario: {scenario_name}")
        scenario.run(raise_exceptions=raise_exceptions)
        LOGGER.info(f"Scenario complete: {scenario_name}")

    # Remove task input/output saved to disk, but keep model weights in case
    # future runs can re-use them
    gobbli.util.cleanup(force=True)


if __name__ == "__main__":
    run()
