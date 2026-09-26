"""Official base-method fine-tuning recipes; prints commands unless --execute."""
import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path
from .common import ROOT, load_config, write_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--method',choices=['minimax','ims3'],required=True)
    p.add_argument('--epochs',type=int,default=8)
    p.add_argument('--checkpoint-every',type=int,default=1000)
    p.add_argument('--step',type=int,help='Require an exact saved training step rather than latest')
    p.add_argument('--execute',action='store_true')
    a=p.parse_args();config=Path(a.config).resolve();cfg=load_config(config)
    spec={'imagewoof':'woof','imagenette':'nette','imagenet100':'100'}
    if cfg['dataset'] not in spec:
        p.error('Pinned upstream fine-tuning recipes cover Woof/Nette/ImageNet100; standalone UAG supports all five')
    target=Path(cfg[a.method+'_checkpoint'])
    output=target.parent/(a.method+'_runs')
    flags={'minimax':('-', 'data-path','global-batch-size','results-dir','ckpt-every','log-every','finetune-ipc'),
           'ims3':('_','data_path','global_batch_size','results_dir','ckpt_every','log_every','finetune_ipc')}[a.method]
    _,data,batch,results,save,log,ipc=flags
    extra=['--'+data,cfg['data_root']+'/train','--ckpt',cfg['dit_checkpoint'],
           '--'+batch,'8','--'+results,str(output),'--'+save,str(a.checkpoint_every),'--'+log,'200',
           '--epochs',str(a.epochs),'--tag',a.method,'--condense','--'+ipc,'-1',
           '--spec',spec[cfg['dataset']],'--nclass',str(len(cfg['classes']))]
    if a.method=='ims3': extra+=['--lambda_match','.002']
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=1',
             '--module','uag.upstream','--method',a.method,'--stage','finetune','--config',str(config),'--']+extra
    print(shlex.join(command))
    if not a.execute:
        print('Plan only. No fine-tuning started. Checkpoint alias:',target)
        return
    if target.exists() or target.is_symlink():
        raise FileExistsError('Existing checkpoint alias preserved: '+str(target))
    tick=time.perf_counter()
    subprocess.run(command,cwd=ROOT,check=True)
    seconds=time.perf_counter()-tick
    checkpoints=list(output.glob('*/checkpoints/*.pt'))
    if a.step is not None: checkpoints=[p for p in checkpoints if int(p.stem)==a.step]
    if not checkpoints:
        raise RuntimeError('No saved checkpoint. Choose a smaller checkpoint interval or inspect the logs.')
    chosen=max(checkpoints,key=lambda p:(p.stat().st_mtime,int(p.stem)))
    target.parent.mkdir(parents=True,exist_ok=True)
    target.symlink_to(chosen.resolve())
    write_json(target.with_suffix('.provenance.json'),dict(command=command,checkpoint=str(chosen),
               chosen_step=int(chosen.stem),finetune_seconds=seconds,
               note='Checkpoint selection is explicit engineering policy, not an author-supplied checkpoint'))


if __name__=='__main__': main()
