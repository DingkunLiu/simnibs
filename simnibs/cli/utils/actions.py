import argparse
from pathlib import Path


class ResolveSubjectID(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        m2m_dir = Path(values).resolve()
        m2m_dir = (
            m2m_dir
            if m2m_dir.name.startswith("m2m_")
            else m2m_dir.with_name(f"m2m_{m2m_dir.name}")
        )
        setattr(namespace, self.dest, m2m_dir)
