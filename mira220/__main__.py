"""Allow `python -m mira220` to run the command-line interface.

Python looks for this file when a package is executed with `-m`. The real
command logic lives in cli.py; this file simply forwards control there.
"""

from .cli import main


if __name__ == "__main__":
    # Only run the CLI when this file is used as the program entry point.
    main()
