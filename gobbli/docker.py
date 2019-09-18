import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import docker


def format_container_output(output: bytes) -> str:
    """
    Format the output of a Docker container for printing or logging.

    Args:
      output: Raw bytes output by the container.

    Returns:
      Output formatted as a string.
    """
    # Decode bytestring, remove trailing newlines that get inserted.
    return output.decode("utf-8").rstrip("\n")


def maybe_mount(
    volumes: Dict[str, Dict[str, Any]],
    host_path: Optional[Path],
    container_path: Optional[Path],
    mode: str = "rw",
):
    """
    If the given host path is not None, mutate the volumes to add an appropriate
    mapping for the path.  Otherwise, don't modify the volumes.

    Args:
      volumes: Dict of existing Docker volumes.
      host_path: An optional host path to be mounted.
      container_path: The path to be mounted if the host path exists.
      mode: Mount mode for the path.
    """
    if host_path is not None:
        volumes[str(host_path)] = {"bind": str(container_path), "mode": mode}


def run_container(
    client: docker.DockerClient,
    image_tag: str,
    cmd: str,
    logger: logging.Logger,
    **kwargs,
) -> str:
    """
    Run a container, appropriately display/handle output, and remove after
    finishing.  Resolve all host directories to be mounted as volumes in order to
    prevent errors related to docker mounting relative paths.

    Args:
      client: The Docker client to use to run the container.
      image_tag: Tag for the image to create the container from.
      cmd: Command to run in the container.
      logger: Logger which will be used to log debug information from the container.
      **kwargs: Additional arguments to pass to :meth:`docker.models.containers.ContainerCollection.run`.

    Returns:
      A string containing all output from the container.
    """
    # Resolve host paths
    run_kwargs = deepcopy(kwargs)
    volumes = run_kwargs.get("volumes", {})
    run_kwargs["volumes"] = {
        os.path.abspath(host_path): container_path
        for host_path, container_path in volumes.items()
    }

    logger.debug(f"Running container for image '{image_tag}' with command '{cmd}'")
    logger.debug(f"Container volumes: {run_kwargs['volumes']}")

    # Run as current UID to avoid files created by root
    try:
        container = client.containers.run(
            image_tag, cmd, user=os.geteuid(), group_add=[os.getegid()], **run_kwargs
        )
    except docker.errors.ImageNotFound:
        raise RuntimeError(
            "gobbli couldn't find the Docker image for the container it was asked to run. "
            "This probably means you didn't call .build() on a model before using it."
        )

    try:
        for line in container.logs(stream=True):
            logger.debug(f"CONTAINER: {format_container_output(line)}")

        container_logs = format_container_output(container.logs())
        results = container.wait()
    finally:
        container.stop()
        container.remove()

    if results["StatusCode"] != 0:
        tail_logs = "\n".join(container_logs.split("\n")[-20:])
        err_msg = (
            f"Error running container (return code {results['StatusCode']})."
            " Last 20 lines of logs: \n"
            f"{tail_logs}"
        )
        if results["StatusCode"] == 137:
            err_msg += (
                "\n\nError code 137 indicates 'out of memory'.  Your input/model are likely too"
                " large to fit in RAM, although you should read the above error message to"
                " confirm.  Try adjusting parameters that affect memory usage, such as"
                " lowering the batch size or sequence length."
            )

        raise RuntimeError(err_msg)

    return container_logs
