"""Rebuild the app's toy workbook from its registered numerical models."""
from __future__ import annotations

import argparse
from itertools import zip_longest
from pathlib import Path
import sys

import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Font

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models import models


# Each entry defines two independent response profiles, not a fitted IVIVC.
# Grids are (start, stop, count); explicit seeds keep examples independent of
# their order. The first five retain the original workbook's smooth curves.
EXAMPLES = [
    dict(model='exponential', seed=101,
         vitro=dict(a=1., b=-.09), vivo=dict(a=1., b=-.07)),
    dict(model='stretched_exponential', seed=102,
         vitro=dict(a=1., b=-.01, c=2.), vivo=dict(a=1., b=-.02, c=1.5)),
    dict(model='double_exponential', seed=103,
         vitro=dict(a=.6, b=-.3, c=.4, d=-.035),
         vivo=dict(a=.45, b=-.2, c=.55, d=-.025)),
    dict(model='mass_loss_one_phase', seed=104,
         vitro=dict(y0=1., A=.9, t50=13., s=2.),
         vivo=dict(y0=1., A=.85, t50=24., s=4.)),
    dict(model='mass_loss_two_phase', seed=105,
         vitro=dict(y0=1., B=.2, kb=.3, D=.68, t50=14., s=1.5),
         vivo=dict(y0=1., B=.15, kb=.18, D=.72, t50=18., s=2.)),
    dict(model='linear_power_decay', seed=106,
         vitro=dict(y0=1., k=.015, tf=25., n=3.),
         vivo=dict(y0=1., k=.01, tf=32., n=3.)),
    dict(model='logistic_logistic', seed=107,
         vitro=dict(A=1., w=.5, t1=5., t2=15., k1=1., k2=2.),
         vivo=dict(A=1., w=.5, t1=7.5, t2=22.5, k1=1./1.5, k2=2./1.5),
         vitro_grid=(0., 20., 41), vivo_grid=(0., 30., 41)),
]
DEFAULT_GRIDS = {'vitro': (0., 30., 31), 'vivo': (0., 30., 16)}
NOISE_SD = 0.03


def example_profiles(example, noisy=False):
    """Evaluate the backend; optionally add reproducible illustrative noise."""
    function = models[example['model']]['model_function']
    profiles = []
    for index, environment in enumerate(('vitro', 'vivo')):
        time = np.linspace(*example.get(environment + '_grid', DEFAULT_GRIDS[environment]))
        value = np.asarray(function(time, **example[environment]), dtype=float)
        if value.shape != time.shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid example values for {example['model']} ({environment}).")
        if noisy:
            rng = np.random.default_rng(np.random.SeedSequence([example['seed'], index]))
            noise = rng.normal(0., NOISE_SD, size=time.size)
            noise[time == 0.] = 0.
            value = np.clip(value + noise, 0., 1.)
        profiles.append((time, value))
    return profiles


def generate_workbook(output):
    """Write smooth sheets first, then their noisy counterparts."""
    workbook = Workbook()
    workbook.remove(workbook.active)
    names = set()
    for noisy in (False, True):
        for example in EXAMPLES:
            name = models[example['model']]['display_name'] + (' (noisy)' if noisy else ' (smooth)')
            if len(name) > 31 or any(c in name for c in '[]:*?/\\') or name.casefold() in names:
                raise ValueError(f'Invalid or duplicate worksheet name: {name!r}')
            names.add(name.casefold())
            (time1, value1), (time2, value2) = example_profiles(example, noisy)
            sheet = workbook.create_sheet(name)
            sheet.append(['time1', 'value1', 'time2', 'value2'])
            for first, second in zip_longest(zip(time1, value1), zip(time2, value2),
                                             fillvalue=(None, None)):
                sheet.append([None if v is None else float(v) for v in (*first, *second)])
            for row in sheet:
                for cell in row:
                    cell.font = Font(name='Aptos Narrow', size=12)
            for column in 'ABCD':
                sheet.column_dimensions[column].width = 14
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output)
    workbook.close()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'toy-data.xlsx',
                        help='Output workbook (default: project toy-data.xlsx; overwrites it).')
    args = parser.parse_args()
    output = generate_workbook(args.output)
    print(f'Wrote {output} ({2 * len(EXAMPLES)} worksheets).')


if __name__ == '__main__':
    main()
