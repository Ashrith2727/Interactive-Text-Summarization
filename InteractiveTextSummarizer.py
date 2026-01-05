"""
Lightweight launcher that exposes the project as a set of modules.

Use `run.py` for training or interactive summarization. This file keeps
the original entrypoint for backward compatibility.
"""

from run import main


if __name__ == '__main__':
    main()
