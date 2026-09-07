#!/usr/bin/env python3
"""Interactive real-time MD built on LAMMPS: explore a force field by feel.

A PLAYGROUND names a force field, a scenario and a mode; every live force-field
parameter becomes a slider you can drag while the simulation runs. Add a new one
by writing a ~30-line file in lammps_live/playgrounds/ (or anywhere, and pass its
path).

    lammps-live --list                          # everything runnable
    lammps-live --playground mesomem_sheet      # drag one bead out of a membrane
    lammps-live --playground mesomem_assembly   # watch 1500 beads self-assemble
    lammps-live --playground mesomem_remote      # 10k beads on a cluster GPU
    lammps-live --playground mesomem_remote --gpu-hours 3   # ... for three hours
    lammps-live --playground mesomem_assembly --mode game   # ... then poke it
    lammps-live --playground mesomem_sheet --preset buckled --input joystick
    lammps-live --verify                        # check the force fields' energy
    lammps-live --playground ./my_idea.py       # your own file, anywhere
    lammps-live --doctor                        # what this machine resolved to
    lammps-live --write-config                  # a config file of your own
    lammps-live --hpc groupbox --playground mesomem_remote   # your cluster

See README.md for setup, controls, and how to write a playground; docs/install.md
for per-platform build options and docs/cluster-setup.md for the remote side.
"""
import argparse
import sys


def build_parser():
    parser = argparse.ArgumentParser(prog="lammps-live", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", choices=["mouse", "keyboard", "joystick"],
                        default="mouse",
                        help="control source: mouse (pointer position + L/R "
                             "buttons), keyboard (WASD + Q/E), or joystick "
                             "(default: mouse)")
    parser.add_argument("--playground", "--system", dest="target", default=None,
                        metavar="KEY_OR_PATH",
                        help="what to run: a bundled playground key, or a path "
                             "to your own playground .py "
                             "(default: the first playground; see --list)")
    parser.add_argument("--mode", choices=["game", "sim"], default=None,
                        help="override a playground's mode: game (drive a "
                             "particle with the input device) or sim (Play / "
                             "Pause / Reset). Both work on any playground")
    parser.add_argument("--preset", default=None, metavar="NAME",
                        help="named parameter set from the playground's presets "
                             "(see --list-presets)")
    parser.add_argument("--fullscreen", action="store_true",
                        help="start in fullscreen (toggle any time with F11)")
    parser.add_argument("--ui-scale", type=float, default=None, metavar="FACTOR",
                        help="how big to draw the 2D UI (text, panel, plots): 1 = "
                             "the classic pixel sizes, 2 = double, which is what a "
                             "4K screen at its NATIVE resolution wants -- the fonts "
                             "are rasterized at that size, so the UI is sharp "
                             "rather than magnified. Default: 2 on a screen taller "
                             "than 1800 px, else 1")
    parser.add_argument("--debug", action="store_true",
                        help="show a per-frame timing breakdown (sim vs. analysis "
                             "vs. render vs. device I/O) in the GUI header")
    parser.add_argument("--list", "--list-systems", "--list-playgrounds",
                        dest="list_all", action="store_true",
                        help="print everything runnable and exit")
    parser.add_argument("--list-presets", action="store_true",
                        help="print each playground's named presets and exit")
    parser.add_argument("--verify", action="store_true",
                        help="check every force field's Python energy expression "
                             "against the potential energy LAMMPS computes from "
                             "the compiled pair style, then exit")
    parser.add_argument("--calibrate", action="store_true",
                        help="print live joystick state for a few seconds and exit")
    parser.add_argument("--remote", default=None, metavar="HOST:PORT",
                        help="for a remote playground: connect straight to an "
                             "already-running server instead of allocating one. "
                             "This is the loopback path -- run the server yourself "
                             "(python -m lammps_live.remote.server) and skip the "
                             "SSH and Slurm machinery entirely")
    parser.add_argument("--token", default="", metavar="SECRET",
                        help="shared secret for --remote (the server's --token)")
    parser.add_argument("--config", default=None, metavar="PATH",
                        help="config file to use, instead of searching "
                             "./lammps-live.toml and "
                             "~/.config/lammps-live/config.toml (see --write-config)")
    parser.add_argument("--hpc", "--system-name", dest="hpc", default=None,
                        metavar="NAME",
                        help="which [remote.systems.NAME] block of the config "
                             "file to run remote playgrounds on, for this run "
                             "only. Overrides the file's own `system = ...`")
    parser.add_argument("--doctor", action="store_true",
                        help="print what this machine resolved to -- compiler, "
                             "architecture flags, MPI headers, config file, and "
                             "the resolved remote target of every remote "
                             "playground -- then exit. The thing to paste into a "
                             "bug report")
    parser.add_argument("--write-config", nargs="?", const="", default=None,
                        metavar="PATH",
                        help="write a commented starter config file (default: "
                             "~/.config/lammps-live/config.toml) and exit. This "
                             "is where your cluster login, your account and your "
                             "compiler flags go, so nothing in the repo has to be "
                             "edited to run it as someone else")
    parser.add_argument("--build-plugin", action="store_true",
                        help="compile the MesoMem pair style now, with the flags "
                             "this machine resolves to, and exit. It is otherwise "
                             "built on demand the first time a 3D scene opens")
    parser.add_argument("--rebuild-plugin", action="store_true",
                        help="like --build-plugin, but compile even if the cached "
                             "library looks current")
    parser.add_argument("--gpu-hours", type=float, default=None, metavar="HOURS",
                        help="how long to ask Slurm for the GPU, in hours (e.g. "
                             "2, or 0.5 for half an hour). This is the WALL CLOCK "
                             "on the allocation and the backstop that gives the "
                             "GPU back if everything else fails to -- a crashed "
                             "app, a lost network, a closed lid -- so it is not a "
                             "budget to pad. A longer request may also queue "
                             "longer. Applies to every remote playground this "
                             "session, and overrides both the playground's own "
                             "declared time (1 hour) and LAMMPS_LIVE_REMOTE_TIME")
    return parser


def _slurm_walltime(hours):
    """Hours as a float -> Slurm's HH:MM:SS.

    Rounded to the nearest minute and floored at one, because `--time=00:00:00`
    is a request Slurm grants and then immediately kills -- and a typo that
    silently means "no time at all" is the wrong failure for a flag whose whole
    job is to keep the session alive. Hours are not capped: which partitions
    allow what is the cluster's business, and it says so plainly when it refuses.
    """
    minutes = max(1, int(round(hours * 60.0)))
    return f"{minutes // 60:02d}:{minutes % 60:02d}:00"


def _apply_app_defaults(args, settings, parser):
    """Fill in what the config file's [app] table says, for anything not passed.

    Only defaults: a flag on the command line always wins, because the flag is
    the thing you typed thirty seconds ago and the file is the thing you wrote
    last month.
    """
    if not settings:
        return
    if args.target is None and settings.get("playground"):
        args.target = str(settings["playground"])
    if args.input == "mouse" and settings.get("input"):
        choice = str(settings["input"])
        if choice not in ("mouse", "keyboard", "joystick"):
            parser.error(f'config [app] input = "{choice}" is not mouse, '
                         "keyboard or joystick")
        args.input = choice
    if args.ui_scale is None and settings.get("ui_scale") is not None:
        args.ui_scale = float(settings["ui_scale"])
    if not args.fullscreen and settings.get("fullscreen"):
        args.fullscreen = True


def _print_listing():
    from .playground import registry
    entries = registry.list_playgrounds()
    width = max((len(k) for k, _s in entries), default=12)
    print("\nPlaygrounds:")
    for key, spec in entries:
        mode = "sim " if spec.playback_controls else "game"
        print(f"  {key:<{width}}  [{mode}]  {spec.name} -- {spec.description}")
    print()
    return 0


def _print_presets():
    from .playground import registry
    for key, pg in registry.all_playgrounds():
        print(f"\n{key}  ({pg.name})")
        if not pg.presets:
            print("  (no presets defined)")
            continue
        for name, overrides in pg.presets.items():
            detail = ", ".join(f"{k}={v}" for k, v in overrides.items()) or "declared defaults"
            print(f"  {name:<18} {detail}")
    print()
    return 0


def _run_verify(target):
    """Cross-check the Python energy expressions against LAMMPS. This is the
    force-field regression check -- see playground/verify.py."""
    from .playground import registry
    from .playground.verify import verify_all
    # Every playground in the package, including the ones the demo does not offer
    # (see registry._SHELVED): this is the force-field regression check, and a
    # force field does not stop needing checking because a scene was shelved.
    refs = [target] if target else sorted(registry.package_keys())
    ok, results = verify_all(refs)
    for label, res in results:
        if isinstance(res, Exception):
            print(f"[verify] ERROR {label}: {res}")
            continue
        print(res.report(label))
    print()
    print("[verify] all force fields agree with LAMMPS" if ok
          else "[verify] MISMATCH -- the Python expression and the pair style disagree")
    return 0 if ok else 1


def _apply_config_selection(args):
    """Put --config and --hpc into the environment, before anything reads them.

    Through the environment on purpose: userconfig and RemoteTarget.resolved()
    are the one door every consumer already goes through (the panel, the session,
    the toolchain), and a second plumbing route would be a second answer to the
    same question -- the same reasoning as --gpu-hours below.
    """
    import os
    if args.config:
        path = os.path.expanduser(args.config)
        if not os.path.isfile(path):
            return f"no config file at {path}"
        os.environ["LAMMPS_LIVE_CONFIG"] = path
    if args.hpc:
        os.environ["LAMMPS_LIVE_SYSTEM"] = args.hpc
    return ""


def _write_config(path):
    from . import userconfig
    try:
        written = userconfig.write_template(path or None)
    except FileExistsError as exc:
        print(f"{exc} already exists -- edit it, or pass a path to write "
              "somewhere else.")
        return 1
    print(f"Wrote {written}\n\n"
          "Set your login and cluster under [remote], your compiler flags under\n"
          "[build], then check it with:  lammps-live --doctor")
    return 0


def _build_plugin(force):
    """Compile the pair style now and say what it did."""
    from .forcefields.mesomem import MESOMEM_PLUGIN
    from .playground import plugin, toolchain
    try:
        chain = toolchain.detect(stub_dir_for=MESOMEM_PLUGIN.directory)
    except toolchain.ToolchainError as exc:
        print(f"[build] {exc}")
        return 1
    for line in chain.describe():
        print(f"[build] {line}")
    try:
        path, what = plugin.build(MESOMEM_PLUGIN, chain=chain, force=force,
                                  log=lambda line: print(f"[build] {line}"))
    except toolchain.ToolchainError as exc:
        print(f"[build] {exc}")
        return 1
    print(f"[build] {path}  ({what})")
    return 0


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    problem = _apply_config_selection(args)
    if problem:
        parser.error(problem)

    from . import userconfig
    userconfig.report_problems()

    if args.write_config is not None:
        return _write_config(args.write_config)
    if args.doctor:
        from . import doctor
        return doctor.report()
    if args.build_plugin or args.rebuild_plugin:
        return _build_plugin(force=args.rebuild_plugin)

    _apply_app_defaults(args, userconfig.load().app_settings(), parser)

    if args.list_all:
        return _print_listing()
    if args.list_presets:
        return _print_presets()
    if args.verify:
        return _run_verify(args.target)

    if args.calibrate:
        from .input import JoystickInput
        # Synchronous (no I/O worker): calibrate reads the device directly on this
        # thread, so a background reader would race it for reports.
        js = JoystickInput(background=False)
        try:
            js.calibrate()
        finally:
            js.close()
        return 0

    from .playground import registry
    if args.target:
        initial_key = registry.resolve(args.target)
    else:
        keys = registry.bundled_keys()
        if not keys:
            parser.error("no playgrounds found")
        initial_key = keys[0]

    # THE GPU WALL CLOCK, applied through the environment rather than threaded
    # down to the panel. Not laziness: RemoteTarget already resolves every field
    # from LAMMPS_LIVE_REMOTE_* precisely so the same playground file works for a
    # second person with different limits (see remote/target.py), and both places a
    # target is resolved -- the RemoteSystem and the RemoteSession -- go through
    # that one door. A parallel plumbing route would have to reach both and would
    # be a second answer to the same question.
    if args.gpu_hours is not None:
        if args.gpu_hours <= 0:
            parser.error("--gpu-hours wants a positive number of hours")
        import os
        os.environ["LAMMPS_LIVE_REMOTE_TIME"] = _slurm_walltime(args.gpu_hours)

    remote_address = None
    if args.remote:
        host, _, port = args.remote.rpartition(":")
        if not host or not port.isdigit():
            parser.error("--remote wants HOST:PORT, e.g. 127.0.0.1:5723")
        remote_address = (host, int(port))

    from .app import App
    app = App(input_mode=args.input, initial_system_key=initial_key,
              fullscreen=args.fullscreen, debug=args.debug,
              mode=args.mode, preset=args.preset,
              remote_address=remote_address, remote_token=args.token,
              ui_scale=args.ui_scale)
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
