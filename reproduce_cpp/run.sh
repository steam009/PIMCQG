#!/bin/bash

set -e

echo "======== Cleaning previous builds ========"
make -f Makefile clean >/dev/null 2>&1 || true
make -f Makefile.mixed clean >/dev/null 2>&1 || true

echo "======== Building project ========"
./build.sh
./bin/host_code