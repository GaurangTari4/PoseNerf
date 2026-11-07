"""Reproducible evidence figures from persisted PoseNeRF reports only."""
from __future__ import annotations
import argparse,json,shutil
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

METHODS={'D0':'Measured training depth','D1_E3':'E3 NeRF depth','D2_E6':'E6 NeRF depth'}
def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,default=Path('workspace'));p.add_argument('--output',type=Path,required=True);a=p.parse_args();w=a.workspace.resolve();o=a.output.resolve()
 if o.exists():raise FileExistsError(o)
 o.mkdir(parents=True);report_path=w/'outputs'/'m11_official_test_20261105'/'m11_report.json';r=json.loads(report_path.read_text());frames=r['frames']
 values={k:np.array([x['methods'][k]['translation_error_m'] for x in frames])*100 for k in METHODS};rots={k:np.array([x['methods'][k]['rotation_error_deg'] for x in frames]) for k in METHODS}
 for name,items,xlabel in [('translation_error_distribution',values,'Translation error (cm)'),('rotation_error_distribution',rots,'Rotation error (degrees)')]:
  fig,ax=plt.subplots(figsize=(8,4));[ax.hist(v,bins=60,density=True,histtype='step',linewidth=1.6,label=METHODS[k]) for k,v in items.items()];ax.set(xlabel=xlabel,ylabel='Density',title='Official Chess test: 2,000 frozen queries');ax.legend();fig.tight_layout();fig.savefig(o/f'{name}.png',dpi=160);plt.close(fig)
 fig,ax=plt.subplots(figsize=(8,4));
 for k,v in values.items():
  x=np.linspace(0,100,250);ax.plot(x,[np.mean(v<=t)*100 for t in x],label=METHODS[k])
 ax.set(xlabel='Translation threshold (cm)',ylabel='Queries localized (%)',title='Official Chess cumulative translation success');ax.legend();fig.tight_layout();fig.savefig(o/'cumulative_translation_success.png',dpi=160);plt.close(fig)
 fig,ax=plt.subplots(figsize=(8,4));labels=list(METHODS);x=np.arange(3);ax.bar(x-.18,[r[k]['metrics']['translation_m']['median']*100 for k in labels],.36,label='Median translation (cm)');ax.bar(x+.18,[r[k]['metrics']['rotation_deg']['median'] for k in labels],.36,label='Median rotation (degrees)');ax.set_xticks(x,[METHODS[k] for k in labels]);ax.set_title('Official Chess test benchmark');ax.legend();fig.tight_layout();fig.savefig(o/'benchmark_comparison.png',dpi=160);plt.close(fig)
 src=w/'outputs'/'m9_geometry_e3_20261105'/'smoke_matches.png'
 if src.exists():shutil.copy2(src,o/'m9_pnp_inlier_matches.png')
 (o/'provenance.json').write_text(json.dumps({'source_report':str(report_path.relative_to(w)),'query_count':len(frames),'methods':METHODS},indent=2))
if __name__=='__main__':main()
