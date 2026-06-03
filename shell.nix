# shell.nix
{ pkgs ? import <nixpkgs> {} }:
with pkgs;
let
  my-python = python312;
  python-with-my-packages = my-python.withPackages (p: with p; [
    pip
    virtualenv
    setuptools
    wheel
    ipython
    pandas
    chromadb
    sentence-transformers
  ]);
  
in
pkgs.mkShell {
  name = "pipzone";
  buildInputs = [
    gcc
    pkg-config
    zlib
    my-python
    python-with-my-packages
  ];
  LD_LIBRARY_PATH = lib.makeLibraryPath [
    stdenv.cc.cc
    my-python
    glib
    glibc
    zlib
    python-with-my-packages
  ];
  PATH = lib.strings.makeBinPath [ "/run/current-system/sw" "./_build" ];
  shellHook = ''
    export PIP_PREFIX=$(pwd)/_build/pip_packages
    export PYTHONPATH="$PIP_PREFIX/${python312.sitePackages}:$PYTHONPATH:$PWD/"
    export PATH="$PIP_PREFIX/bin:$PATH"
    unset SOURCE_DATE_EPOCH
    pip3 --cache /tmp/pip_cache install -r requirements.txt
    pip3 --cache /tmp/pip_cache install -e .
  '';
}
