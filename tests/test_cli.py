"""The command line, in the two places it decides something that costs money or
time on someone else's machine.

Nothing here starts a simulation. `--gpu-hours` is a number that becomes a Slurm
`--time`, and a Slurm `--time` is the backstop that gives an A100 back when
everything else has failed to -- so the translation is worth pinning, and so is
the fact that the flag actually reaches `salloc` rather than sitting in a parsed
namespace nobody reads.
"""
import os

import pytest

from lammps_live import cli
from lammps_live.playground import registry


@pytest.mark.parametrize("hours, expected", [
    (1, "01:00:00"),
    (2.5, "02:30:00"),
    (0.5, "00:30:00"),
    (12, "12:00:00"),
    (3.25, "03:15:00"),
    # Rounded to the nearest minute, not truncated.
    (1.008, "01:00:00"),
    (1.01, "01:01:00"),
])
def test_hours_become_a_slurm_walltime(hours, expected):
    assert cli._slurm_walltime(hours) == expected


def test_a_tiny_request_is_floored_at_one_minute():
    """`--time=00:00:00` is a request Slurm grants and then kills at once, so a
    fat-fingered `--gpu-hours 0.0001` must not silently mean "no time at all"."""
    assert cli._slurm_walltime(0.0001) == "00:01:00"


def test_gpu_hours_is_rejected_rather_than_reinterpreted():
    """Zero and negative are mistakes, and the failure has to be at the prompt
    rather than an hour later in a queue."""
    parser = cli.build_parser()
    for bad in ("0", "-2"):
        with pytest.raises(SystemExit):
            cli.main(["--playground", "mesomem_remote", "--gpu-hours", bad,
                      "--remote", "127.0.0.1:1"])
    assert parser.parse_args(["--gpu-hours", "2"]).gpu_hours == 2.0


def test_gpu_hours_reaches_the_salloc_line(monkeypatch):
    """The flag is applied through LAMMPS_LIVE_REMOTE_TIME, which is the one door
    every RemoteTarget resolution goes through (the RemoteSystem's and the
    RemoteSession's). This is the test that says so -- a flag that only reached one
    of them would work in the panel and not in the job."""
    monkeypatch.delenv("LAMMPS_LIVE_REMOTE_TIME", raising=False)
    playground = registry.load("mesomem_remote")
    assert playground.remote.resolved().time == "01:00:00", "the declared default"

    monkeypatch.setenv("LAMMPS_LIVE_REMOTE_TIME", cli._slurm_walltime(3))
    target = playground.remote.resolved()
    assert target.time == "03:00:00"
    assert "--time=03:00:00" in target.salloc_args()


def test_the_flag_overrides_an_environment_variable_that_is_already_set(monkeypatch):
    """Because the help says it does, and because the two would otherwise disagree
    silently: the flag writes the same variable it would be overridden by."""
    monkeypatch.setenv("LAMMPS_LIVE_REMOTE_TIME", "00:05:00")
    os.environ["LAMMPS_LIVE_REMOTE_TIME"] = cli._slurm_walltime(4)
    assert registry.load("mesomem_remote").remote.resolved().time == "04:00:00"
