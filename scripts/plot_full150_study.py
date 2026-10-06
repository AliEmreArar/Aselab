"""Export full-dataset model/caption comparisons as standalone research figures."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / 'runs/full150'
PROFILES = ['identity', 'llm_brief', 'llm_balanced', 'llm_detailed',
            'llm_human_description', 'llm_attribute_list', 'llm_distinctive_first']
LABELS = ['Orijinal', 'Kısa', 'Dengeli', 'Ayrıntılı', 'Günlük dil', 'Özellik\nlistesi', 'Ayırt edici\nönce']

for kind, title in [('image', 'Metin → görsel'), ('text', 'Metin → orijinal açıklama')]:
    frame = pd.read_csv(ROOT / f'{kind}_summary.csv')
    profiles = PROFILES if kind == 'image' else PROFILES[1:]
    labels = LABELS if kind == 'image' else LABELS[1:]
    models = sorted(frame.model.unique())
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5), constrained_layout=True)
    for ax, domain, label in zip(axes, ['face','person','vehicle'], ['Yüz','Kişi','Araç']):
        table = frame[frame.domain == domain].pivot(index='model', columns='variant_type', values='R1')
        values = table.reindex(index=models, columns=profiles).to_numpy()
        im = ax.imshow(values, vmin=0, vmax=1, cmap='viridis', aspect='auto')
        ax.set_title(label + ' — 50 sorgu / 50 aday')
        ax.set_xticks(range(len(profiles)), labels, rotation=45, ha='right', fontsize=9)
        ax.set_yticks(range(len(models)), models, fontsize=9)
        for y, x in np.ndindex(values.shape):
            ax.text(x, y, f'{values[y,x]:.0%}', ha='center', va='center',
                    color='black' if values[y,x] > .65 else 'white', fontsize=9)
    fig.colorbar(im, ax=axes, label='Recall@1', shrink=.85)
    fig.suptitle(title + ' — 150 örneğin tamamında betimsel karşılaştırma', fontsize=15)
    fig.savefig(ROOT / f'{kind}_recall_at_1.png', dpi=200)
    plt.close(fig)
    print(ROOT / f'{kind}_recall_at_1.png')
