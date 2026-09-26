"""Summaries and paper-style plots consume actual result files only."""
import argparse
import csv
import statistics
from pathlib import Path
from .common import read_json, write_json


def summarize(a):
    groups = {}
    for path in Path(a.root).rglob('result.json'):
        result = read_json(path)
        if 'final_top1' not in result:
            continue
        # Keep method/dataset parent and architecture to avoid mixing experiments.
        key = (str(path.parent.parent.relative_to(a.root)), result.get('arch', ''), result.get('ipc', ''))
        groups.setdefault(key, []).append((result['final_top1'], result.get('seed'), str(path)))
    rows = []
    for (experiment, arch, ipc), values in sorted(groups.items()):
        data = [v[0] for v in values]
        rows.append(dict(experiment=experiment, arch=arch, ipc=ipc, n=len(data),
                         mean=statistics.mean(data), std=statistics.stdev(data) if len(data)>1 else None,
                         seeds=[v[1] for v in values], files=[v[2] for v in values]))
    write_json(a.out, rows)
    with Path(a.out).with_suffix('.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['experiment','arch','ipc','n','mean','std','seeds','files'])
        writer.writeheader()
        writer.writerows(rows)


def plot(a):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    scales = [1,2.5,5,7.5,10,12.5,15,17.5,20]
    times = [408,510,612,693,795,856,917,958,999]
    data = np.full((len(times),len(scales)), np.nan)
    for i,t in enumerate(times):
        for j,s in enumerate(scales):
            results = list(Path(a.root).glob('lambda%s_t%d/seed*/result.json'%(s,t)))
            if results:
                data[i,j] = np.mean([read_json(p)['final_top1'] for p in results])
    if not np.isfinite(data).any():
        raise ValueError('No completed sensitivity experiments; refusing fabricated heatmap')
    fig, ax = plt.subplots(figsize=(9,6))
    artist=ax.imshow(np.ma.masked_invalid(data), cmap='coolwarm', aspect='auto')
    ax.set_xticks(range(len(scales)), labels=scales)
    ax.set_yticks(range(len(times)), labels=times)
    ax.set_xlabel('Entropy guidance strength λ'); ax.set_ylabel('Original cutoff timestep')
    for i,j in np.argwhere(np.isfinite(data)):
        ax.text(j,i,'%.1f'%data[i,j],ha='center',va='center',fontsize=8)
    fig.colorbar(artist, ax=ax, label='Top-1 accuracy (%)')
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=300)


def grid(a):
    from PIL import Image, ImageDraw
    base, guided = Path(a.base), Path(a.guided)
    classes = read_json(base/'config.json')['config']['classes']
    canvas = Image.new('RGB',(120+512*a.count,256*len(classes)), 'white')
    draw = ImageDraw.Draw(canvas)
    for i,c in enumerate(classes):
        draw.text((5,i*256+5),c,fill='black')
        for j in range(a.count):
            for k,folder in enumerate([base,guided]):
                with Image.open(folder/c/('%05d.png'%j)) as image:
                    canvas.paste(image.convert('RGB').resize((256,256)),(120+j*512+k*256,i*256))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(a.out)


def main():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest='action',required=True)
    for name, fn in [('summary',summarize),('heatmap',plot),('grid',grid)]:
        q=sub.add_parser(name);q.set_defaults(func=fn)
        q.add_argument('--out',required=True)
        if name in ['summary','heatmap']:
            q.add_argument('--root',required=True)
        else:
            q.add_argument('--base',required=True);q.add_argument('--guided',required=True)
        if name=='grid': q.add_argument('--count',type=int,default=2)
    a=p.parse_args();a.func(a)


if __name__=='__main__':
    main()
