#!/usr/bin/env python3
import argparse
import filecmp
import shutil
import sys

from msevb_tests.cases import discover, find_case


def main() -> int:
    parser = argparse.ArgumentParser(description="Copy example inputs into cases/, or report drift.")
    parser.add_argument("cases", nargs="*", help="case names or ids (default: all)")
    parser.add_argument("--check", action="store_true", help="report differences, copy nothing")
    args = parser.parse_args()

    cases = [find_case(c) for c in args.cases] if args.cases else discover()
    drift = 0
    for case in cases:
        src_dir = case.source_dir
        for name in case.files:
            src, dst = src_dir / name, case.path / name
            if not src.exists():
                print(f"{case.id}: missing source {src}")
                drift += 1
                continue
            same = dst.exists() and filecmp.cmp(src, dst, shallow=False)
            if args.check:
                if not same:
                    print(f"{case.id}: {name} differs from {src}")
                    drift += 1
            elif not same:
                shutil.copy2(src, dst)
                print(f"{case.id}: copied {name}")
    if args.check:
        print("no drift" if drift == 0 else f"{drift} file(s) drifted")
    return 1 if drift else 0


if __name__ == "__main__":
    sys.exit(main())
