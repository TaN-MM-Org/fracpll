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
TUTORIALS = ["tutorial_design_and_margins.py", "tutorial_jitter_from_trace.py",
             "tutorial_reference_spurs.py"]

# the tutorials live in the repository, not in the installed package
pytestmark = pytest.mark.skipif(not os.path.isdir(EXAMPLES),
                                reason="examples/ is not present")


def _documented(path):
    """The indented block after 'Printed output' in the docstring."""
    text = open(path, encoding="utf-8").read()
    block = re.split(r"Printed output \(fracpll \d+\.\d+\.\d+\):\n\n",
                     text, maxsplit=1)[1]
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


def test_key_numbers_of_the_spur_tutorial():
    """Checks of the spur tutorial's printed numbers against things the
    script does not compute that way:
    * the pulses do not depend on the filter, so adding the R3-C3
      section changes the 50 MHz spur by exactly the ratio of the two
      transimpedances there (checked on the printed, rounded means:
      0.1 dB);
    * the spur is nearly proportional to the leakage (the locked pulse
      replaces the leaked charge), so its spread in dB is close to
      20 log10(e) u_rel of the leakage, times sqrt((n-1)/(n-3)), the
      standard deviation of a t-distribution with n - 1 degrees of
      freedom over its scale (within 30 %: the mismatch adds a little)."""
    import math

    import numpy as np

    from fracpll import loop_filter_impedance, third_order_impedance
    c = "\n".join(_documented(os.path.join(EXAMPLES, TUTORIALS[2])))
    filt = re.search(r"C1 = ([\d.]+) pF, R2 = ([\d.]+) Ohm, C2 = ([\d.]+) nF",
                     c)
    c1, r2, c2 = (float(filt.group(1)) * 1e-12, float(filt.group(2)),
                  float(filt.group(3)) * 1e-9)
    third = re.search(r"R3 = ([\d.]+) kOhm, C3 = ([\d.]+) pF: spur at "
                      r"50 MHz (-[\d.]+) \+- ([\d.]+) dBc", c)
    r3, c3 = float(third.group(1)) * 1e3, float(third.group(2)) * 1e-12
    s2 = re.search(r"second order: spur at 50 MHz (-[\d.]+) \+- ([\d.]+)",
                   c)
    w = 2 * math.pi * 50e6
    z2 = abs(loop_filter_impedance(w, c1, [(r2, c2)])[0])
    z3 = abs(third_order_impedance(w, c1, r2, c2, r3, c3)[0])
    drop = float(s2.group(1)) - float(third.group(3))
    assert abs(drop - 20 * math.log10(z2 / z3)) <= 0.1
    leak = re.search(r"leakage +([\d.]+) \+- ([\d.]+) uA \((\d+) readings",
                     c)
    n = int(leak.group(3))
    u_rel = float(leak.group(2)) / float(leak.group(1))
    want = 20 / math.log(10) * u_rel * math.sqrt((n - 1) / (n - 3))
    assert abs(float(s2.group(2)) / want - 1) < 0.3
    assert np.isfinite(want)
