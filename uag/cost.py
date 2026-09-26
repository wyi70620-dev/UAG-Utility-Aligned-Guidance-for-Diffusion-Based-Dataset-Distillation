"""Table 10 timing harness. No timing numbers are created in plan-only mode."""
import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path
from .common import ROOT, load_config, write_json, read_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default=str(ROOT/'configs/imagewoof.json'))
    p.add_argument('--out',required=True)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args();cfg=load_config(a.config);out=Path(a.out).resolve()
    commands={}
    for method,scale in [('dit',0),('uag',10)]:
        commands[method]=[sys.executable,'-m','uag.sample','--config',str(Path(a.config).resolve()),
                          '--out',str(out/method),'--ipc','10','--seed','0','--scale',str(scale)]
    for method in ['minimax','mgd3','ims3']:
        checkpoint=cfg['dit_checkpoint'] if method=='mgd3' else cfg[method+'_checkpoint']
        args=['--ckpt',checkpoint,'--save-dir',str(out/method),'--spec','woof','--seed','0']
        if method=='ims3':
            args+=['--ipc','10','--groups','5','--real-train-dir',cfg['data_root']+'/train',
                   '--w-real','.4','--w-sep','.9','--sel-eps','0','--sample-batch','10']
        else:
            args+=['--num-samples','10']
            if method=='mgd3':
                args+=['--guidance','--stop_t','25','--imagenet_dir',cfg['data_root'],'--num-datasets','1']
        commands[method]=[sys.executable,'-m','uag.upstream','--method',method,'--config',str(Path(a.config).resolve()),'--']+args
    for name,command in commands.items(): print(name,shlex.join(command))
    if not a.execute:
        print('Plan only; no timing measurements or model execution.')
        return
    out.mkdir(parents=True,exist_ok=True)
    if any((out/name).exists() for name in commands):
        raise FileExistsError('Timing requires fresh output directories, not resumed/cached generations')
    rows=[]
    for name,command in commands.items():
        tick=time.perf_counter()
        with (out/(name+'.log')).open('w') as f:
            subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        seconds=time.perf_counter()-tick
        finetune=0. if name in ['dit','uag','mgd3'] else None
        if name in ['minimax','ims3']:
            provenance=Path(cfg[name+'_checkpoint']).with_suffix('.provenance.json')
            if provenance.exists(): finetune=read_json(provenance).get('finetune_seconds')
        rows.append(dict(method=name,sampling_seconds=seconds,finetune_seconds=finetune,
                         total_seconds=seconds+finetune if finetune is not None else None))
        write_json(out/'timing.json',dict(rows=rows,includes_process_startup_and_model_load=True,
                   hardware='Record nvidia-smi separately; no assumed hardware timing'))


if __name__=='__main__': main()
