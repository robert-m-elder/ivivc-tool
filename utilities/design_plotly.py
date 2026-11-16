"""Small Plotly figure specifications using original numerical coordinates.

This module does not import Plotly or calculate uncertainty. It converts already
prepared lines, pointwise ranges, and observations to finite JSON-compatible
values; the browser owns common interaction and download settings.
"""
from __future__ import annotations

from html import escape
import math


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _pairs(values):
    x, y = [], []
    for raw_x, raw_y in values:
        point_x, point_y = _finite(raw_x), _finite(raw_y)
        valid = point_x is not None and point_y is not None
        x.append(point_x if valid else None)
        y.append(point_y if valid else None)
    return x, y


def make_figure(series, *, band=None, observations=None, title, x_label,
                y_label, identifier, band_label='Pointwise 95% reference range',
                observation_label='Observed response'):
    """Return a ``data``/``layout`` figure, or None if no point is available.

    Missing line coordinates remain gaps. A missing band coordinate splits its
    filled region rather than joining the available endpoints across the gap.
    Labels are escaped because Plotly supports limited HTML in displayed text.
    """
    data = []
    text_x, text_y = escape(str(x_label)), escape(str(y_label))
    hover = text_x + ': %{x:.6g}<br>' + text_y + ': %{y:.6g}<extra>%{fullData.name}</extra>'
    if band is not None:
        segments, segment = [], []
        for raw_x, raw_lo, raw_hi in band:
            x, lo, hi = (_finite(value) for value in (raw_x, raw_lo, raw_hi))
            if x is not None and lo is not None and hi is not None and lo <= hi:
                segment.append((x, lo, hi))
            elif segment:
                segments.append(segment)
                segment = []
        if segment:
            segments.append(segment)
        for index, segment in enumerate(segments):
            common = dict(type='scatter', mode='lines', x=[point[0] for point in segment],
                          name=escape(str(band_label)), legendgroup='range',
                          line=dict(width=0), connectgaps=False)
            data.append(dict(common, y=[point[1] for point in segment],
                             showlegend=False, hoverinfo='skip'))
            data.append(dict(common, y=[point[2] for point in segment],
                             fill='tonexty', fillcolor='rgba(36,83,92,0.16)',
                             showlegend=index == 0,
                             customdata=[[point[1], point[2]] for point in segment],
                             hovertemplate=(text_x + ': %{x:.6g}<br>' +
                                            text_y + ': %{customdata[0]:.6g}–%{customdata[1]:.6g}'
                                            '<extra>%{fullData.name}</extra>')))
    for index, line in enumerate(series):
        x, y = _pairs(line['values'])
        if not any(value is not None for value in x):
            continue
        dash = str(line.get('dash', ''))
        dash = 'dash' if dash == '6 4' else dash or 'solid'
        data.append(dict(type='scatter', mode='lines' if line.get('join', True) else 'markers',
                         name=escape(str(line['label'])), legendgroup=f'series-{index}',
                         x=x, y=y, connectgaps=False,
                         line=dict(color=line['color'], width=2, dash=dash),
                         marker=dict(color=line['color'], size=7), hovertemplate=hover))
    if observations is not None:
        x, y = _pairs(observations)
        valid = [(point_x, point_y) for point_x, point_y in zip(x, y) if point_x is not None]
        if valid:
            data.append(dict(type='scatter', mode='markers', name=escape(str(observation_label)),
                             legendgroup='observations', x=[point[0] for point in valid],
                             y=[point[1] for point in valid],
                             marker=dict(color='#20313d', size=8,
                                         line=dict(color='white', width=1)), hovertemplate=hover))
    if not data:
        return None
    return dict(data=data, layout=dict(
        title=dict(text=escape(str(title)), x=0.),
        xaxis=dict(title=dict(text=text_x), showgrid=False, zeroline=False),
        yaxis=dict(title=dict(text=text_y), gridcolor='#e5ebef', zeroline=False),
        font=dict(family='Arial, sans-serif', color='#20313d'),
        paper_bgcolor='white', plot_bgcolor='white',
        margin=dict(l=70, r=25, t=65, b=80),
        legend=dict(orientation='h', y=-.2, x=0., groupclick='togglegroup'),
        hovermode='closest', uirevision=str(identifier),
    ))
