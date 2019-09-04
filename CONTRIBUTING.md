# Contributing

Thanks for your contribution!  We're starting out with a few simple guidelines:

## Contributor License Agreement

You must [sign a Contributor License Agreement](TODO) (CLA) to contribute to this project.

## Code Style

We use a few linting tools to enforce consistency of style and formatting, which are run in CI.  Make sure your code passes the pre-test checks in `run_ci.sh` before it's pushed.  Additionally:

 - Pretty much any code added under the main gobbli codebase should have type hints.  Code run as part of model Docker containers is exempt from this guideline but should still be formatted.
 - Add docstrings where appropriate (especially public interface functions).  Use Sphinx references to link to other parts of the project where needed.

## Tests

A lot of the functionality in gobbli is difficult to test (large models, long runtimes, complex functions).  We don't test every edge case, but try to make sure there's at least a black box test verifying end-to-end success of any new code you add.  White box testing is appreciated where feasible.

## Code Reviews

All submissions must come in the form of PRs against the master branch and will be reviewed.  If possible, ensure your patch only implements/changes one thing.
