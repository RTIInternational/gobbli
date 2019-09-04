from dataclasses import dataclass
from pathlib import Path

DOCKER_ROOT = Path("/gobbli")


@dataclass
class ContainerTaskContext:
    """
    Encapsulates filesystem organization for tasks which take some
    input and produce some output using directories mapped between
    the host and a container.

    Provide necessary input/output directories but also allow the caller to create
    their own directories as needed and map them to container paths.

    Args:
      task_root_dir: The root directory for the task on the host.  All directories in the context
        will be descendents of this directory.
    """

    task_root_dir: Path

    def host_dir(self, name: str) -> Path:
        """
        Create (if necessary) and return a directory on the host under the task root path.

        Args:
          name: The name of the directory to create (relative to
            :paramref:`ContainerTaskContext.params.task_root_dir`).

        Returns:
          The full path to the created task root directory.
        """
        named_dir = self.task_root_dir / name
        named_dir.mkdir(parents=True, exist_ok=True)
        return named_dir

    def to_container(self, host_dir: Path) -> Path:
        """
        Convert a given directory on the host to a directory under some canonical
        path in the container.  The given host directory must be under our root
        directory.

        Args:
          host_dir: The full path to a descendent directory of the
            :paramref:`ContainerTaskContext.params.task_root_dir`).
        """
        return DOCKER_ROOT / host_dir.resolve().relative_to(
            self.task_root_dir.resolve()
        )

    @property
    def container_root_dir(self) -> Path:
        """
        Returns:
          The container root directory corresponding to the
          :paramref:`ContainerTaskContext.params.task_root_dir` host directory.
        """
        return self.to_container(self.task_root_dir)

    @property
    def host_input_dir(self) -> Path:
        """
        Returns:
          The host directory to be used for task input.
        """
        return self.host_dir("input")

    @property
    def host_output_dir(self) -> Path:
        """
        Returns:
          The host directory to be used for task output.
        """
        return self.host_dir("output")

    @property
    def container_input_dir(self) -> Path:
        """
        Returns:
          The directory to be used for task input, as mapped in the container.
        """
        return self.to_container(self.host_input_dir)

    @property
    def container_output_dir(self) -> Path:
        """
        Returns:
          The directory to be used for task output, as mapped in the container.
        """
        return self.to_container(self.host_output_dir)
