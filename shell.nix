# shell.nix
{ pkgs ? import <nixpkgs> {} }:
with pkgs;
let
  my-python = python3;
  python-with-my-packages = my-python.withPackages (p: with p; [
    pip
    virtualenv
    setuptools
    wheel
    xvfbwrapper
    ipython
    pyglet
    imageio
    matplotlib
    torch
    gymnasium
    pandas
  ]);
  
in
pkgs.mkShell {
  name = "pipzone";
  buildInputs = [
    gcc
    pkg-config
    #cairo.dev
    #xorg.libxcb.dev
    #xorg.libX11.dev
    #xorg.libXext.dev
    ffmpeg
    xvfb-run
    freeglut.dev
    zlib
    my-python
    python-with-my-packages
  ];
  LD_LIBRARY_PATH = lib.makeLibraryPath [
    stdenv.cc.cc
    my-python
    freeglut.dev
    glib
    glibc
    zlib
    python-with-my-packages
  ];
  PATH = lib.strings.makeBinPath [ "/run/current-system/sw" "./penv" ];
  shellHook = ''
    export PIP_PREFIX=$(pwd)/_build/pip_packages
    export PYTHONPATH="$PIP_PREFIX/${pkgs.python3.sitePackages}:$PYTHONPATH:$PWD/telepresence"
    export PATH="$PIP_PREFIX/bin:$PATH"
    unset SOURCE_DATE_EPOCH
    #python3 -m penv
    #. ./penv/bin/activate
    pip3 --cache /tmp/pip_cache install -r requirements.txt
    #zsh
  '';
}
