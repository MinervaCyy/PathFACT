"""Optional byte-exact regression for the frozen G06 materializer.

Set PATHFACT_PIPELINE_REFERENCE to the canonical pipeline directory.
No historical dataset is required for ordinary public tests.
"""

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from pathfact.datasets.g06 import materialize_g06


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TestFrozenG06Materializer(unittest.TestCase):

    def test_byte_exact_reproduction(self):
        reference = os.environ.get(
            "PATHFACT_PIPELINE_REFERENCE"
        )

        if not reference:
            self.skipTest(
                "PATHFACT_PIPELINE_REFERENCE not configured"
            )

        root = Path(reference)

        audit_path = (
            root
            / "G06_train200_lossval44"
            / "G06_AUDIT_V1.json"
        )

        audit = json.loads(
            audit_path.read_text(encoding="utf-8")
        )

        # Verify the recorded historical source-script identity.
        script = audit["materializer_script"]

        self.assertEqual(
            sha256_file(root / script["path"]),
            script["sha256"],
        )

        # Verify upstream files against their frozen fingerprints.
        inputs = audit["inputs"]

        for name in (
            "g00_cases",
            "g05_joint_valid",
            "g04b_contrastive_valid",
        ):
            item = inputs[name]

            with self.subTest(input=name):
                self.assertEqual(
                    sha256_file(root / item["path"]),
                    item["sha256"],
                )

        task_sources = inputs["task_prompt_sources"]

        for item in task_sources:
            with self.subTest(
                task_source=item["path"]
            ):
                self.assertEqual(
                    sha256_file(root / item["path"]),
                    item["sha256"],
                )

        # Generate into a temporary directory.
        # Original frozen files remain untouched.
        with tempfile.TemporaryDirectory(
            prefix="pathfact_g06_regression_"
        ) as temp:

            outputs = materialize_g06(
                g00_path=(
                    root / inputs["g00_cases"]["path"]
                ),
                g05_path=(
                    root / inputs["g05_joint_valid"]["path"]
                ),
                g04b_path=(
                    root / inputs["g04b_contrastive_valid"]["path"]
                ),
                task_prompt_paths=[
                    root / item["path"]
                    for item in task_sources
                ],
                output_dir=Path(temp) / "generated",
            )

            expected = audit["outputs"]

            self.assertEqual(len(expected), 7)
            self.assertEqual(set(outputs), set(expected))

            for name, item in expected.items():
                with self.subTest(output=name):
                    generated = outputs[name]
                    frozen = root / item["path"]

                    self.assertEqual(
                        sha256_file(frozen),
                        item["sha256"],
                    )

                    self.assertEqual(
                        sha256_file(generated),
                        item["sha256"],
                    )

                    self.assertEqual(
                        generated.read_bytes(),
                        frozen.read_bytes(),
                    )


if __name__ == "__main__":
    unittest.main()
