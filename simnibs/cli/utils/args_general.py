import textwrap

from simnibs import __version__
from simnibs.cli.utils.helpers import CommandLineArgument
from simnibs.cli.utils import actions

subid = CommandLineArgument(
    ["subid"],
    dict(
        type=str,
        help="Subject ID or /path/to/{subid} or /path/to/m2m_{subid}. The former will resolve to m2m_{subid} in the current working directory. The latter cases both resolve to /path/to/m2m_{subid}",
        action=actions.ResolveSubjectID,
    ),
)

out_dir = CommandLineArgument(
    ["-o", "--out_dir"],
    dict(default=".", help="Directory in which to save output file(s)."),
)

version = CommandLineArgument(
    ["-v", "--version"], dict(action="version", version=__version__)
)

debug = CommandLineArgument(
    ["--debug"],
    dict(
        action="store_true",
        default=False,
        help="""Write results from intermediate steps to disk.""",
    ),
)

fsaverage = CommandLineArgument(
    ["--fsaverage"],
    dict(
        type=int,
        choices=[5, 6, 7],
        default=7,
        help=textwrap.dedent("""
        The fsaverage template to morph to. The number denotes the subdivision
        factor. The number of vertices per hemisphere for each subdivision is
            5  ->  10,242
            6  ->  40,962
            7  -> 163,842 (full resolution)
        By default, use fsaverage 7.
        """),
    ),
)
