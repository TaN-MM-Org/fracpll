"""The tutorials in examples/ run, and print what their docstrings and
the README say they print (key numbers checked line by line).  The
numbers are not anchors of the physics (the tests of each function
are); this checks that the documented output is the real output."""
import os
import re
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXAMPLES = os.path.join(ROOT, "examples")
TUTORIALS = ["tutorial_design_and_margins.py", "tutorial_jitter_from_trace.py"]

# the tutorials live in the repository, not in the installed package
pytestmark = pytest.mark.skipif(not os.path.isdir(EXAMPLES),
                                reason="examples/ is not present")


def _documented(path):
    """The indented block after 'Printed output' in the docstring."""
    text = open(path, encoding="utf-8").read()
    block = text.split("Printed output (fracpll 0.6.0):\n\n", 1)[1]
    block = block.split('"""', 1)[0]
    return [line[4:] for line in block.splitlines() if line.strip()]


@pytest.mark.parametrize("name", TUTORIALS)
def test_tutorial_prints_its_documented_output(name):
    path = os.path.join(EXAMPLES, name)
    env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    out = subprocess.run([sys.executable, path], capture_output=True,
                         text=True, check=True, env=env, cwd=ROOT,
                         timeout=120).stdout
    got = out.strip().splitlines()
    want = _documented(path)
    assert got == want
    # and the README shows the same block
    readme = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
    assert "\n".join(want) in readme


def test_key_numbers():
    """The headline numbers, parsed from the documented output."""
    a = "\n".join(_documented(os.path.join(EXAMPLES, TUTORIALS[0])))
    kv = re.search(r"Kvco = ([\d.]+) \+- ([\d.]+) MHz/V", a)
    assert abs(float(kv.group(1)) - 50.0) < 3 * float(kv.group(2))
    pm = re.search(r"phase margin ([\d.]+) \+- ([\d.]+) degrees", a)
    assert abs(float(pm.group(1)) - 55.0) < 1.0
    b = "\n".join(_documented(os.path.join(EXAMPLES, TUTORIALS[1])))
    meas = re.search(r"measured jitter ([\d.]+) \+- ([\d.]+) ps", b)
    model = re.search(r"model jitter +([\d.]+) ps", b)
    assert abs(float(meas.group(1)) - float(model.group(1))) \
        < 3 * float(meas.group(2))
