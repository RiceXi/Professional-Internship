"""Render publication/exportable figures from the saved experiment CSV."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,default=Path('experiments/task1'))
    args=parser.parse_args()
    rows=list(csv.DictReader((args.input/'summary.csv').open()))
    raw=json.loads((args.input/'raw.json').read_text())
    modes=['contiguous','paged','swap']
    labels=['Contiguous reservation','Demand paging','Paging + host swap']
    colors=['#8a96a8','#2771b6','#159b86']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4.3),layout='constrained')
    for mode,label,color in zip(modes,labels,colors):
        selected=[r for r in rows if r['scenario']=='mixed' and r['block_size']=='4' and r['mode']==mode]
        means=[];errors=[];frags=[]
        for frames in [8,16,32]:
            group=[r for r in selected if int(r['frames'])==frames]
            values=[int(r['completed']) for r in group]
            means.append(statistics.mean(values));errors.append(statistics.stdev(values))
            frags.append(statistics.mean(float(r['weighted_internal_fragmentation'])*100 for r in group))
        axes[0].errorbar([8,16,32],means,yerr=errors,marker='o',label=label,color=color,capsize=4)
        axes[1].plot([8,16,32],frags,marker='o',color=color,label=label)
    axes[0].set(ylabel='Completed requests / 32',xlabel='Physical pages (4 tokens/page)',ylim=(0,34),title='Capacity outcomes — mean ± sample SD')
    axes[1].set(ylabel='Time-weighted internal fragmentation (%)',xlabel='Physical pages (4 tokens/page)',ylim=(0,100),title='Allocated-slot waste')
    for ax in axes: ax.grid(alpha=.15);ax.set_xticks([8,16,32])
    axes[0].legend(fontsize=8)
    fig.suptitle('CPU simulation · seeds 42, 43, 44 · not GPU performance',fontsize=13)
    for ext in ['png','svg']:
        fig.savefig(args.input/f'capacity_fragmentation.{ext}',dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    branches=raw['branches']
    before=[r['before_write']['allocated_slots'] for r in branches]
    after=[r['after_write']['allocated_slots'] for r in branches]
    for x,b,a in zip([0,1],before,after):
        ax.bar(x-.17,b,.34,color='#8a96a8');ax.bar(x+.17,a,.34,color='#2771b6')
        ax.text(x-.17,b+1,str(b),ha='center');ax.text(x+.17,a+1,str(a),ha='center')
    ax.set(xticks=[0,1],xticklabels=['Independent copies','Shared prefix + COW'],ylabel='Allocated physical token slots',ylim=(0,max(after)*1.2),title='CPU simulation · 9-token prefix, 8 branches, 4-token pages')
    ax.legend(['Before branch append','After one append per branch'])
    for ext in ['png','svg']:
        fig.savefig(args.input/f'cow_sharing.{ext}',dpi=180)
    plt.close(fig)
    print('Rendered capacity/fragmentation and COW figures')


if __name__=='__main__':
    main()
