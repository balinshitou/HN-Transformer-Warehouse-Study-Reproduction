"""Reproduce manuscript Tables 2–4 and the warehouse trajectory comparison."""
import argparse
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from warehouse.plot_comparison import FIGURE_NAME, plot_comparison
from warehouse.protocol import ROOT, new_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'results',
                        help='New output directory (default: results/ in the project root).')
    args = parser.parse_args()
    output = args.out.resolve()
    if output.exists():
        parser.error(f'Output already exists. Choose a new directory with --out: {output}')

    try:
        with new_output(output) as results, TemporaryDirectory(dir=results) as folder:
            work = Path(folder)
            tracks = work / 'trajectories'
            lengths = work / 'lengths'
            scores = work / 'scores'
            stages = (
                ('reconstruct.py', '--raw-root', str(ROOT / 'data/pdr_raw'), '--out', str(tracks)),
                ('summarize_lengths.py', '--tracks', str(tracks), '--paper-only', '--out', str(lengths)),
                ('export_kpis.py', '--paper-only', '--out', str(scores)),
            )
            for i, stage in enumerate(stages, 1):
                print(f'[{i}/{len(stages)}] Running {stage[0]}', flush=True)
                subprocess.run([sys.executable, '-B', *stage], cwd=ROOT, check=True)
            print('[Figure] Rendering the four-panel trajectory comparison', flush=True)
            plot_comparison(tracks, results / FIGURE_NAME)
            for source in (lengths / 'table2.csv', scores / 'table3.csv', scores / 'table4.csv'):
                source.rename(results / source.name)
    except subprocess.CalledProcessError as error:
        parser.exit(error.returncode or 1, f'Reproduction failed in {error.cmd[2]}.\n')

    print(f'Manuscript Tables 2–4 and the comparison figure written to {output}')


if __name__ == '__main__':
    main()
