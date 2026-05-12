# Copyright (c) 2024 Dan Wells
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CLI utilities for Fire-based scripts."""

import inspect
import sys


def validate_fire_args(cls):
    """Check for unrecognized flags before Fire runs the command.

    Fire defers unknown-argument errors until after the method completes,
    which means a typo'd flag silently runs the command with defaults
    and only reports the error afterwards. This pre-validates by
    inspecting the target method's signature against the CLI flags.

    Call this before fire.Fire(cls) in the script's main():

        from wespeaker.utils.cli import validate_fire_args
        validate_fire_args(MeetingSimulator)
        fire.Fire(MeetingSimulator)

    Args:
      cls: The class passed to fire.Fire().
    """
    args = sys.argv[1:]
    if not args:
        return

    # Identify the subcommand (first non-flag argument)
    subcommand = None
    for arg in args:
        if not arg.startswith('-'):
            subcommand = arg
            break
    if subcommand is None:
        return

    method = getattr(cls, subcommand, None)
    if method is None:
        return  # Let Fire handle unknown subcommands

    # Get valid parameter names from the method signature
    sig = inspect.signature(method)
    valid_params = set(sig.parameters.keys()) - {'self'}

    # Check each flag-style argument
    for arg in args:
        if arg.startswith('--'):
            # Strip leading dashes and any =value suffix
            flag = arg.lstrip('-').split('=')[0]
            # Normalize: fire accepts both underscores and hyphens
            flag_normalized = flag.replace('-', '_')
            if flag_normalized not in valid_params:
                print(
                    f"ERROR: Unknown argument '--{flag}' for "
                    f"'{subcommand}' subcommand.\n"
                    f"Valid options: {sorted(valid_params)}",
                    file=sys.stderr)
                sys.exit(1)
