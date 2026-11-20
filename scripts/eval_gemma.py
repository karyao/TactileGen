"""Run and resume a Gemma baseline on a PlotQA image split."""
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

from src.baseline.gemma import DEFAULT_MODEL_ID, PROMPT_VERSION, GemmaChartExtractor, GemmaExtractionError
from src.contracts import VerticalBarChart


def norm(value):
    return ' '.join(str(value).casefold().split())


def unique_matches(truth, predicted):
    a, b = list(map(norm, truth)), list(map(norm, predicted))
    return {i: b.index(x) for i, x in enumerate(a) if x and a.count(x) == 1 and b.count(x) == 1}


def score(record, chart):
    models = record['models']
    axis = record['general_figure_info']['x_axis']
    labels, ticks = axis['major_labels']['values'], axis['major_ticks']['values']
    lookup = {}
    for tick, label in zip(ticks, labels):
        lookup.setdefault(str(tick), str(label))
    categories = [lookup.get(str(x), str(x)) for x in models[0]['x']]
    names = [m.get('label') or m.get('name') or '' for m in models]
    values = [v for m in models for v in m['y'] if v is not None]
    out = dict(category_total=len(categories), series_total=len(names), value_total=len(values),
               category_correct=0, series_correct=0, value_aligned=0, value_correct=0, bar_count_correct=0)
    if not isinstance(chart, VerticalBarChart):
        return out
    cm = unique_matches(categories, chart.categories)
    sm = unique_matches(names, [s.name or '' for s in chart.series])
    out['category_correct'], out['series_correct'] = len(cm), len(sm)
    if len(models) == len(chart.series) == 1:
        sm = {0: 0}
    out['bar_count_correct'] = int(sum(len(s.values) for s in chart.series if s.has_bars) == len(values))
    scale = max((abs(v) for v in values), default=0)
    for i, j in sm.items():
        series = chart.series[j]
        if not series.has_bars:
            continue
        for k, l in cm.items():
            if k >= len(models[i]['y']) or models[i]['y'][k] is None:
                continue
            out['value_aligned'] += 1
            out['value_correct'] += int(abs(series.values[l] - models[i]['y'][k]) <= 0.05 * scale)
    return out


def write_json(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images', type=Path, default=Path.home() / 'Desktop/TactileGen/dataset_plotqa_bars_eval/images/test')
    p.add_argument('--annotations', type=Path, default=Path('data/plotqa/test/annotations.json'))
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--limit', type=int)
    p.add_argument('--model-id', default=DEFAULT_MODEL_ID)
    p.add_argument('--revision')
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    p.add_argument('--dtype', default='auto', choices=['auto', 'float32', 'float16', 'bfloat16'])
    p.add_argument('--max-new-tokens', type=int, default=4096)
    a = p.parse_args()
    if a.limit is not None and a.limit < 1:
        p.error('--limit must be positive')
    images = sorted(a.images.glob('*.png'))
    if not images:
        p.error('No PNG images found')
    images = images[:a.limit]
    ids = [x.stem for x in images]
    if len(ids) != len(set(ids)):
        p.error('Duplicate image IDs')
    wanted = set(ids)
    records = {}
    for r in json.loads(a.annotations.read_text()):
        key = str(r['image_index'])
        if key in wanted:
            if key in records:
                p.error(f'Duplicate annotation: {key}')
            records[key] = r
    if wanted - records.keys():
        p.error(f'Missing annotations: {sorted(wanted - records.keys())[:10]}')
    digest = hashlib.sha256(a.annotations.read_bytes()).hexdigest()
    config = dict(images=str(a.images.resolve()), ids=ids, annotations=str(a.annotations.resolve()), annotation_sha256=digest,
                  model_id=a.model_id, revision=a.revision, device=a.device, dtype=a.dtype,
                  max_new_tokens=a.max_new_tokens, prompt_version=PROMPT_VERSION)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = a.output_dir / 'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != config:
            p.error('Existing run settings differ; choose a new output directory')
    else:
        if any(a.output_dir.iterdir()):
            p.error('Output directory must be empty for a new run')
        write_json(manifest, config)
    rows_dir = a.output_dir / 'results'
    rows_dir.mkdir(exist_ok=True)
    extractor = GemmaChartExtractor(model_id=a.model_id, revision=a.revision, device=a.device, dtype=a.dtype, max_new_tokens=a.max_new_tokens)
    for index, image in enumerate(images, 1):
        path = rows_dir / (image.stem + '.json')
        if path.exists():
            continue
        row = {'image_id': image.stem, 'status': 'failed'}
        try:
            result = extractor.extract_result(image)
            row.update(status='complete', chart=dataclasses.asdict(result.chart), raw_response=result.raw_response,
                       metadata=result.metadata, scores=score(records[image.stem], result.chart))
        except GemmaExtractionError as exc:
            row.update(error=str(exc), error_stage=exc.stage, raw_response=exc.raw_response, metadata=exc.metadata,
                       scores=score(records[image.stem], None))
            if exc.stage in ('dependencies', 'device', 'loading'):
                raise SystemExit(f'{exc.stage}: {exc}') from exc
        write_json(path, row)
        print(f'[{index}/{len(images)}] {image.stem}: {row["status"]}', flush=True)
    rows = [json.loads((rows_dir / (i + '.json')).read_text()) for i in ids]
    totals = {key: sum(r['scores'][key] for r in rows) for key in rows[0]['scores']}
    def ratio(n, d):
        return n / d if d else None
    metrics = dict(images=len(rows), successes=sum(r['status'] == 'complete' for r in rows), totals=totals,
                   bar_count_accuracy=ratio(totals['bar_count_correct'], len(rows)),
                   category_accuracy=ratio(totals['category_correct'], totals['category_total']),
                   series_label_accuracy=ratio(totals['series_correct'], totals['series_total']),
                   value_alignment_coverage=ratio(totals['value_aligned'], totals['value_total']),
                   value_accuracy_aligned=ratio(totals['value_correct'], totals['value_aligned']),
                   value_accuracy_all=ratio(totals['value_correct'], totals['value_total']))
    write_json(a.output_dir / 'metrics.json', metrics)
    (a.output_dir / 'report.md').write_text('# Gemma baseline\n\n' + '\n'.join(f'- {k}: {v}' for k, v in metrics.items() if k != 'totals') + '\n')


if __name__ == '__main__':
    main()
