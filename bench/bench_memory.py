"""Reproducible CPU experiments for task one; no GPU/model dependencies."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.memory import ContiguousMemory, PagedMemory, AllocationError
from src.memory.simulation import Request, simulate, workload


def summarize(run, **labels):
    trace = run['trace']
    allocated = sum(t['allocated_slots'] for t in trace)
    useful = sum(t['useful_slots'] for t in trace)
    total = len(run['workload'])
    return {**labels, 'mode':run['mode'], 'frames':run['frames'], 'block_size':run['block_size'],
        'submitted':total, 'completed':run['completed'], 'capacity_rejected':run['capacity_rejected'],
        'rejection_rate':run['capacity_rejected']/total if total else 0,
        'weighted_internal_fragmentation':1-useful/allocated if allocated else 0,
        'peak_allocated_slots':max((t['allocated_slots'] for t in trace),default=0),
        'swap_in_pages':run['final'].get('swap_in_pages',0),
        'swap_out_pages':run['final'].get('swap_out_pages',0)}


def branch_case(shared: bool):
    memory = PagedMemory(40, 4)
    parent = memory.new_sequence()
    prefix = list(range(9))
    for value in prefix:
        memory.append(parent, value)
    children = []
    for _ in range(8):
        if shared:
            child = memory.fork(parent)
        else:
            child = memory.new_sequence()
            for value in prefix:
                memory.append(child, value)
        children.append(child)
    before = memory.metrics()
    for i, child in enumerate(children):
        memory.append(child, 100+i)
    after = memory.metrics()
    assert memory.read_sequence(parent) == prefix
    for i, child in enumerate(children):
        assert memory.read_sequence(child) == prefix+[100+i]
    memory.check_invariants()
    for sid in [parent]+children:
        memory.free(sid)
    memory.check_invariants()
    return {'shared':shared, 'prefix_tokens':9, 'branches':8,
            'before_write':before, 'after_write':after, 'final':memory.metrics()}


def external_fragmentation_case():
    memory = ContiguousMemory(12)
    a,b,c = [memory.new_sequence(4) for _ in range(3)]
    memory.free(a)
    memory.free(c)
    metrics = memory.metrics()
    rejected = False
    try:
        memory.new_sequence(5)
    except AllocationError:
        rejected = True
    memory.free(b)
    memory.check_invariants()
    return {'layout':'[free 4][used 4][free 4]', 'requested_tokens':5,
            'rejected':rejected, 'fragmented':metrics, 'after_coalesce':memory.metrics()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('experiments/task1'))
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    runs, rows = [], []
    for block in [2,4,8]:
        for frames in [8,16,32]:
            for seed in [42,43,44]:
                for mode in ['contiguous','paged','swap']:
                    run = simulate(workload(seed),mode,frames=frames,block_size=block,host_pages=32)
                    run.update(scenario='mixed',seed=seed)
                    runs.append(run)
                    rows.append(summarize(run,scenario='mixed',seed=seed,concurrency=32,tokens=0))
    for count in [4,8,16,32]:
        for length in [8,16,32,64]:
            jobs=[Request(i,0,length,64) for i in range(count)]
            for mode in ['contiguous','paged','swap']:
                run=simulate(jobs,mode,frames=8,block_size=4,host_pages=64)
                run.update(scenario='capacity_sweep',seed=0)
                runs.append(run)
                rows.append(summarize(run,scenario='capacity_sweep',seed=0,concurrency=count,tokens=length))
    sources = [*sorted((ROOT/'src/memory').glob('*.py')), Path(__file__)]
    report = {'kind':'cpu_simulation', 'created_utc':datetime.now(timezone.utc).isoformat(),
        'python':sys.version,'platform':platform.platform(),
        'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'working_tree_dirty':bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()),
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'clock':'unit iteration ticks; no hardware time or bandwidth model',
        'admission':'arrivals ordered by arrival/rid; capacity failures terminate request without retry',
        'host_capacity':'32 pages for mixed runs; 64 pages for capacity sweep; one full working set must fit GPU frames',
        'runs':runs, 'branches':[branch_case(False),branch_case(True)],
        'external_fragmentation':external_fragmentation_case()}
    (args.output/'raw.json').write_text(json.dumps(report,ensure_ascii=False,separators=(',',':'))+'\n')
    with (args.output/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'{len(runs)} deterministic runs -> {args.output}')


if __name__=='__main__':
    main()
