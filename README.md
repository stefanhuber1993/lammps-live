# Impostor Viewer

A standalone WebGL2 viewer for LAMMPS dump / data / xyz files, drawing every
bead as a ray-cast sphere impostor. Live at
https://stefanhuber1993.github.io/lammps-live/

**This branch is the source of truth for the viewer.** It is not built from
`main`: edit `impostor-viewer.html` here, commit, push, and GitHub Pages
serves the new version a minute later. `index.html` only redirects to it, so
the short link on the poster's QR code keeps working.

The whole viewer is one self-contained file, with no build step and no
dependencies; it also runs straight from disk.
