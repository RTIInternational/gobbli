import os
from pathlib import Path

import gobbli

os.environ["GOBBLI_DIR"] = "./benchmark_data"

m = gobbli.augment.MarianMT(
    target_languages=["french"],
    # use_gpu=True, nvidia_visible_devices="0"
)
m.build()

print(m.augment(["This is four words " * 500]))
