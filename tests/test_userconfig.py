"""The config file, which exists so that this repo is not one person's.

Every test here is a version of the same sentence: a second person, with a second
account on a second cluster, must be able to run the remote demos without editing
a source file. The things that can go wrong with that are all about PRECEDENCE
(who wins between the playground, the file and the environment) and about TYPOS
(a key that silently does nothing is worse than one that complains), so that is
what is pinned.
"""
import os

import pytest

from lammps_live import cli, userconfig
from lammps_live.remote.target import RemoteTarget


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """No test may see the developer's own ~/.config/lammps-live/config.toml.

    Autouse and not opt-in: a test that accidentally reads the real file would
    pass here and fail on someone else's machine, which is precisely the failure
    this whole module is about.
    """
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "home" / ".config"))
    monkeypatch.delenv("LAMMPS_LIVE_CONFIG", raising=False)
    monkeypatch.delenv("LAMMPS_LIVE_SYSTEM", raising=False)
    for name in list(os.environ):
        if name.startswith("LAMMPS_LIVE_REMOTE_"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "home").mkdir()
    return tmp_path


def write(tmp_path, text, name="lammps-live.toml"):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def test_no_config_at_all_changes_nothing():
    """The common case, and the one that must cost nothing: an app with no config
    file resolves exactly what the playground declared."""
    declared = RemoteTarget(host="snellius.surf.nl", user="stefanh")
    assert declared.resolved() == declared
    assert userconfig.load().path == ""


def test_the_username_moves_out_of_the_source_file(tmp_path, monkeypatch):
    """The whole point, in one test: `user = "pietro"` in a file, and the demo a
    third person wrote still logs in as Pietro."""
    monkeypatch.setenv("LAMMPS_LIVE_CONFIG",
                       write(tmp_path, '[remote]\nuser = "pietro"\n'))
    target = RemoteTarget(host="snellius.surf.nl", user="stefanh").resolved()
    assert target.user == "pietro"
    assert target.destination == "pietro@snellius.surf.nl"
    assert target.host == "snellius.surf.nl", "unrelated fields are untouched"


def test_a_named_machine_is_selected_and_the_other_one_is_not(tmp_path, monkeypatch):
    monkeypatch.setenv("LAMMPS_LIVE_CONFIG", write(tmp_path, '''
[remote]
system = "groupbox"
user = "pietro"

[remote.systems.snellius]
host = "snellius.surf.nl"
partition = "gpu_a100"

[remote.systems.groupbox]
host = "cluster.example.org"
partition = "compute"
gpus = 0
profile = "cluster-cpu"
'''))
    target = RemoteTarget().resolved()
    assert (target.host, target.partition, target.gpus) == ("cluster.example.org",
                                                            "compute", 0)
    assert target.profile == "cluster-cpu"
    # The top-level key still applies -- it is the one thing that is usually the
    # same on every machine a person has.
    assert target.user == "pietro"


def test_one_machine_needs_no_choosing(tmp_path, monkeypatch):
    """Requiring `system = "..."` when there is exactly one block would be
    ceremony for the case almost everyone is in."""
    monkeypatch.setenv("LAMMPS_LIVE_CONFIG", write(tmp_path, '''
[remote.systems.mine]
host = "cluster.example.org"
'''))
    assert RemoteTarget().resolved().host == "cluster.example.org"
    name, why = userconfig.load().selected_system()
    assert (name, why) == ("mine", "the only one defined")


def test_hpc_picks_a_machine_for_one_run(tmp_path, monkeypatch):
    """`--hpc` writes LAMMPS_LIVE_SYSTEM rather than threading a value down, so
    that the panel and the session -- which both resolve targets independently --
    cannot disagree about which cluster this run is on."""
    path = write(tmp_path, '''
[remote]
system = "snellius"
[remote.systems.snellius]
host = "snellius.surf.nl"
[remote.systems.groupbox]
host = "cluster.example.org"
''')
    args = cli.build_parser().parse_args(["--config", path, "--hpc", "groupbox"])
    assert cli._apply_config_selection(args) == ""
    assert RemoteTarget().resolved().host == "cluster.example.org"
    assert userconfig.load().selected_system() == ("groupbox", "LAMMPS_LIVE_SYSTEM")


def test_the_environment_still_wins_over_the_file(tmp_path, monkeypatch):
    """The stack is playground < file < environment, and --gpu-hours lives at the
    top of it: a config file that says two hours must not quietly outrank the
    flag someone just typed."""
    monkeypatch.setenv("LAMMPS_LIVE_CONFIG",
                       write(tmp_path, '[remote]\ntime = "02:00:00"\npartition = "gpu_h100"\n'))
    assert RemoteTarget().resolved().time == "02:00:00"

    monkeypatch.setenv("LAMMPS_LIVE_REMOTE_TIME", cli._slurm_walltime(3))
    target = RemoteTarget().resolved()
    assert target.time == "03:00:00"
    assert "--time=03:00:00" in target.salloc_args()
    assert target.partition == "gpu_h100", "the rest of the file still applies"


def test_the_file_wins_over_the_playground_declaration(tmp_path, monkeypatch):
    """A playground declares the SHAPE of the run; the file says whose account and
    which queue. If the declaration won, a second person could not run the demo
    without editing it -- which is the bug this file exists to fix."""
    monkeypatch.setenv("LAMMPS_LIVE_CONFIG",
                       write(tmp_path, '[remote]\npartition = "gpu_h100"\naccount = "prj1234"\n'))
    declared = RemoteTarget(partition="gpu_a100")
    target = declared.resolved()
    assert target.partition == "gpu_h100"
    assert "--account=prj1234" in target.salloc_args()


def test_typed_values_survive_the_trip(tmp_path, monkeypatch):
    """TOML has real types; the target's fields have real types; nothing in
    between may turn a float into a string (an fps of "30.0" reaches the wire
    format and fails much later than it should)."""
    monkeypatch.setenv("LAMMPS_LIVE_CONFIG", write(tmp_path, '''
[remote]
gpus = 2
cpus_per_task = 32
fps = 30.0
queue_wait = 120
free_run = true
extra_salloc = ["--constraint=a100", "--exclusive"]
'''))
    target = RemoteTarget().resolved()
    assert target.gpus == 2 and isinstance(target.gpus, int)
    assert target.fps == 30.0 and isinstance(target.fps, float)
    assert target.queue_wait == 120.0 and isinstance(target.queue_wait, float)
    assert target.free_run is True
    assert target.extra_salloc == ("--constraint=a100", "--exclusive")
    assert "--constraint=a100" in target.salloc_args()


def test_a_bad_value_costs_that_key_and_nothing_else(tmp_path, monkeypatch):
    """One hand-edited line should not stop a demo, and must not pass silently."""
    monkeypatch.setenv("LAMMPS_LIVE_CONFIG", write(tmp_path, '''
[remote]
gpus = "one"
partition = "gpu_h100"
'''))
    target = RemoteTarget(gpus=1).resolved()
    assert target.gpus == 1, "the bad value was skipped, not applied"
    assert target.partition == "gpu_h100", "the good ones still landed"
    assert any("gpus" in p for p in userconfig.load().problems)


def test_a_misspelled_key_is_reported_rather_than_ignored(tmp_path, monkeypatch, capsys):
    """`usr = "pietro"` doing nothing is a twenty-second SSH prompt for the wrong
    account and no explanation. It has to reach a human at startup."""
    monkeypatch.setenv("LAMMPS_LIVE_CONFIG", write(tmp_path, '''
[remote]
usr = "pietro"

[buld]
compiler = "g++"
'''))
    cfg = userconfig.load()
    assert any("usr" in p and "user" in p for p in cfg.problems), \
        "a near-miss key names the field it was probably meant to be"
    assert any("[buld]" in p for p in cfg.problems), "an unknown SECTION is a typo"
    assert userconfig.report_problems() > 0
    assert "buld" in capsys.readouterr().err


def test_a_system_name_that_matches_nothing_says_so(tmp_path, monkeypatch):
    monkeypatch.setenv("LAMMPS_LIVE_CONFIG", write(tmp_path, '''
[remote]
system = "snelius"
[remote.systems.snellius]
host = "snellius.surf.nl"
'''))
    problems = userconfig.load().problems
    assert any("snelius" in p and "snellius" in p for p in problems)


def test_broken_toml_does_not_stop_the_app(tmp_path, monkeypatch):
    """A stray bracket in a file about SSH usernames must not be the reason a
    local, offline, no-cluster demo refuses to start."""
    monkeypatch.setenv("LAMMPS_LIVE_CONFIG", write(tmp_path, "[remote\nuser = nope"))
    cfg = userconfig.load()
    assert cfg.problems and cfg.data == {}
    declared = RemoteTarget(user="stefanh")
    assert declared.resolved() == declared


def test_an_edit_takes_effect_without_a_restart(tmp_path, monkeypatch):
    """The parse is cached (resolved() is called from the panel), and the cache key
    includes the mtime -- so fixing a username and pressing Connect again works,
    which is how anyone actually debugs one of these."""
    path = write(tmp_path, '[remote]\nuser = "typo"\n')
    monkeypatch.setenv("LAMMPS_LIVE_CONFIG", path)
    assert RemoteTarget().resolved().user == "typo"
    os.utime(path, (0, 0))
    with open(path, "w") as handle:
        handle.write('[remote]\nuser = "pietro"\n')
    assert RemoteTarget().resolved().user == "pietro"


def test_the_starter_config_is_valid_and_complains_about_nothing(tmp_path, monkeypatch):
    """`--write-config` writes an example, and an example that generates warnings
    on its own is worse than no example. This also pins that every key in it is a
    key the loader knows -- the template and the parser drifting apart is the
    obvious way this rots."""
    path = userconfig.write_template(str(tmp_path / "sub" / "config.toml"))
    assert os.path.isfile(path)
    monkeypatch.setenv("LAMMPS_LIVE_CONFIG", path)
    cfg = userconfig.load()
    assert cfg.problems == (), cfg.problems
    assert cfg.selected_system() == ("snellius", "config [remote] system")
    target = RemoteTarget().resolved()
    assert target.host == "snellius.surf.nl"
    assert target.user == "your-login", "the placeholder is meant to be replaced"

    with pytest.raises(FileExistsError):
        userconfig.write_template(path)


def test_the_search_order_prefers_the_checkout_to_the_home_directory(tmp_path, monkeypatch):
    """A per-project file beats the personal one, so a group can check a
    lammps-live.toml into a shared repo and still have one person override it."""
    home = tmp_path / "home" / ".config" / "lammps-live"
    home.mkdir(parents=True)
    (home / "config.toml").write_text('[remote]\nuser = "personal"\n')
    assert RemoteTarget().resolved().user == "personal"

    write(tmp_path, '[remote]\nuser = "project"\n')
    assert userconfig.find_path() == str(tmp_path / "lammps-live.toml")
    assert RemoteTarget().resolved().user == "project"

    monkeypatch.setenv("LAMMPS_LIVE_CONFIG",
                       write(tmp_path, '[remote]\nuser = "explicit"\n', name="other.toml"))
    assert RemoteTarget().resolved().user == "explicit"


def test_the_app_table_only_fills_in_what_was_not_typed(tmp_path, monkeypatch):
    """A preference file is for the things you always want; the flag you just
    typed is for this run. The flag wins."""
    monkeypatch.setenv("LAMMPS_LIVE_CONFIG", write(tmp_path, '''
[app]
input = "joystick"
ui_scale = 1.5
playground = "mesomem_sheet"
'''))
    parser = cli.build_parser()
    settings = userconfig.load().app_settings()

    args = parser.parse_args([])
    cli._apply_app_defaults(args, settings, parser)
    assert (args.input, args.ui_scale, args.target) == ("joystick", 1.5, "mesomem_sheet")

    args = parser.parse_args(["--input", "mouse", "--playground", "lj_argon"])
    cli._apply_app_defaults(args, settings, parser)
    assert args.target == "lj_argon"
    # --input mouse is indistinguishable from the default, and that is a real
    # limit of doing it this way: someone with `input = "joystick"` in a file who
    # wants the mouse for one run says so with the file, not the flag. Pinned so
    # the limitation is a decision rather than a surprise.
    assert args.input == "joystick"


def test_a_missing_explicit_config_is_an_error_at_the_prompt(tmp_path):
    """`--config typo.toml` silently falling back to the search path would run
    the demo on the wrong cluster. Fail where the typo was made."""
    args = cli.build_parser().parse_args(["--config", str(tmp_path / "nope.toml")])
    assert "no config file" in cli._apply_config_selection(args)


# --- what a different site's Slurm wants -------------------------------------
def test_a_site_that_spells_it_gres_gets_gres(tmp_path, monkeypatch):
    """`--gpus=N` is Slurm 20.11 and later. Plenty of clusters are older, or
    configure GPUs as a generic resource, and there `--gres=gpu:a100:1` is the only
    spelling that works. It cannot be guessed from here, so it is a config key --
    and it must replace the flag it stands in for rather than join it."""
    monkeypatch.setenv("LAMMPS_LIVE_CONFIG",
                       write(tmp_path, '[remote]\ngres = "gpu:a100:1"\n'))
    args = RemoteTarget(gpus=1).resolved().salloc_args()
    assert "--gres=gpu:a100:1" in args
    assert not any(a.startswith("--gpus") for a in args)


def test_a_cpu_only_target_asks_for_no_gpu_at_all(tmp_path, monkeypatch):
    """`--gpus=0` is not "never mind": some Slurm versions reject it and others
    grant it while binding nothing. A CPU cluster is a supported target (profile
    "cluster-cpu"), so the flag has to be absent."""
    monkeypatch.setenv("LAMMPS_LIVE_CONFIG", write(tmp_path, '''
[remote]
gpus = 0
profile = "cluster-cpu"
partition = "compute"
'''))
    target = RemoteTarget().resolved()
    args = target.salloc_args()
    assert not any("gpu" in a for a in args), args
    assert "--partition=compute" in args
    assert target.gpu_request() == []


def test_every_srun_asks_for_the_same_thing_as_the_salloc(tmp_path, monkeypatch):
    """Three places make this request -- salloc, the probe's srun, the server's
    srun -- and they all go through one helper, because two of three being fixed is
    the failure mode of having three copies."""
    monkeypatch.setenv("LAMMPS_LIVE_CONFIG",
                       write(tmp_path, '[remote]\ngres = "gpu:1"\n'))
    target = RemoteTarget().resolved()
    assert target.gpu_request() == ["--gres=gpu:1"]
    assert all(flag in target.salloc_args() for flag in target.gpu_request())
