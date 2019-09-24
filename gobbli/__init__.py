import warnings

import pkg_resources

import gobbli.augment as augment
import gobbli.dataset as dataset
import gobbli.experiment as experiment
import gobbli.io as io
import gobbli.model as model
from gobbli.util import TokenizeMethod

# Warn the user of potential conflicts using the old third-party typing/dataclasses
# modules
for conflicting_pkg in ("typing", "dataclasses"):
    req = pkg_resources.Requirement.parse(conflicting_pkg)
    if pkg_resources.working_set.find(req) is not None:
        warnings.warn(
            f"You've installed a third-party module named '{conflicting_pkg}' which "
            "conflicts with a standard library module of the same name.  This can cause "
            "errors when unpickling code, e.g. when running experiments using Ray. Consider "
            f"uninstalling the module:\n\npip uninstall {conflicting_pkg}"
        )

__all__ = [
    # Modules
    "augment",
    "dataset",
    "experiment",
    "model",
    "io",
    # Misc top level imports
    "TokenizeMethod",
]
