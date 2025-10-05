import argparse
import sys
import platform
import importlib.metadata

def main():
    parser = argparse.ArgumentParser(
        prog="pymatcha",
        description="🍵 PyMatcha — A lightweight deep learning library inspired by PyTorch."
    )

    parser.add_argument(
        "command",
        nargs="?",
        default="welcome",
        help="Command to run: welcome | info | test | help"
    )

    args = parser.parse_args()

    if args.command == "welcome":
        print("""
======================================
   🍵 Welcome to PyMatcha (v0.1.0)
   Lightweight Deep Learning Library
   Author: karinadegou
   Repository: https://github.com/karinadegou/PyMatcha
======================================
""")
    elif args.command == "info":
        print("Environment Info:")
        print(f"Python version: {platform.python_version()}")
        print(f"Platform: {platform.system()} {platform.release()}")
        try:
            numpy_version = importlib.metadata.version("numpy")
            print(f"Numpy version: {numpy_version}")
        except importlib.metadata.PackageNotFoundError:
            print("Numpy not found.")
    elif args.command == "test":
        print("Running PyMatcha self-test...")
        try:
            import numpy as np
            from pymatcha import tensor
            x = np.array([1, 2, 3])
            print(f"Tensor demo: {x}")
            print("Test passed.")
        except Exception as e:
            print("Test failed:", e)
    elif args.command == "help":
        parser.print_help()
    else:
        print(f"Unknown command: {args.command}")
        print("Use 'pymatcha help' to see available commands.")


if __name__ == "__main__":
    main()
