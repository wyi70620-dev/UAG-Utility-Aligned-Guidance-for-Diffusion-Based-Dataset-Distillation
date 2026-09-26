"""Generate complete experiment commands by default; --execute is opt-in."""
import argparse
import subprocess
import sys
from pathlib import Path
from .common import ROOT, load_config, write_json

ARCHS = ['convnet6', 'resnetap10', 'resnet18']
MATRIX = {
    'imagewoof': ([10, 20, 50, 70, 100], ARCHS),
    'imagenette': ([10, 20, 50], ['resnetap10']),
    'imageidc': ([1, 10, 20, 50], ['resnet18']),
    'imagenet100': ([10, 20], ARCHS),
    'imagenet1k': ([10, 50], ['resnet18']),
}


class Plan:
    def __init__(self, root):
        self.root = root
        self.jobs = []
        self.names = set()

    def add(self, name, module, *args):
        if name in self.names:
            return
        self.names.add(name)
        self.jobs.append(dict(name=name, cwd=str(ROOT), command=[sys.executable, '-m', module]+[str(v) for v in args]))

    def evaluate(self, dataset, config, syn, output, ipc, seed, arch):
        if dataset == 'imagenet1k':
            regions = syn / 'soft_regions'
            self.add('regions:'+str(regions), 'uag.soft_labels', '--config', config,
                     '--syn', syn, '--ipc', ipc, '--out', regions)
            self.add(str(output), 'uag.train', '--config', config, '--out', output, '--role', 'soft',
                     '--arch', arch, '--ipc', ipc, '--seed', seed, '--regions', regions/'manifest.json')
        else:
            self.add(str(output), 'uag.train', '--config', config, '--out', output,
                     '--syn', syn, '--arch', arch, '--ipc', ipc, '--seed', seed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--suite', choices=['main', 'ablation', 'references', 'plugins', 'diagnostics', 'redundancy', 'baselines'], default='main')
    p.add_argument('--datasets', nargs='+', choices=list(MATRIX), default=list(MATRIX))
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--root', default=str(ROOT/'artifacts'))
    p.add_argument('--plan', default=str(ROOT/'plans/commands.json'))
    p.add_argument('--execute', action='store_true')
    p.add_argument('--main-checkpoint', choices=['dit', 'minimax'], default='dit',
                   help='Manuscript does not identify every main-table checkpoint; choose explicitly')
    a = p.parse_args()
    root = Path(a.root).resolve()
    plan = Plan(root)

    def generate(dataset, config, ipc, seed, tag, scale=10, cutoff=795, ref=None):
        cfg = load_config(config)
        syn = root/'generated'/dataset/tag/('ipc%d_seed%d' % (ipc, seed))
        checkpoint = cfg['dit_checkpoint' if a.main_checkpoint == 'dit' else 'minimax_checkpoint']
        args = ['--config', config, '--out', syn, '--ipc', ipc, '--seed', seed,
                '--scale', scale, '--cutoff', cutoff, '--checkpoint', checkpoint]
        if ref:
            args += ['--reference', ref]
        plan.add('generate:'+str(syn), 'uag.sample', *args)
        return syn

    if a.suite == 'main':
        for dataset in a.datasets:
            config = ROOT/'configs'/(dataset+'.json')
            cfg = load_config(config)
            if dataset != 'imagenet1k' and not Path(cfg['reference']).is_file():
                plan.add('reference:'+dataset, 'uag.train', '--config', config, '--role', 'reference',
                         '--out', Path(cfg['reference']).parent)
            budgets, architectures = MATRIX[dataset]
            for ipc in budgets:
                for seed in a.seeds:
                    for method, scale in [('dit', 0), ('uag', 10)]:
                        tag = a.main_checkpoint+'_'+method
                        syn = generate(dataset, config, ipc, seed, tag, scale)
                        for arch in architectures:
                            out = root/'evaluation'/dataset/tag/('ipc%d_seed%d_%s' % (ipc, seed, arch))
                            plan.evaluate(dataset, config, syn, out, ipc, seed, arch)
    elif a.suite == 'ablation':
        config = ROOT/'configs/imagewoof.json'
        for scale in [1, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20]:
            for cutoff in [408, 510, 612, 693, 795, 856, 917, 958, 999]:
                tag = 'lambda%s_t%d' % (scale, cutoff)
                for seed in a.seeds:
                    syn = generate('imagewoof', config, 20, seed, tag, scale, cutoff)
                    plan.evaluate('imagewoof', config, syn, root/'ablation'/tag/('seed%d'%seed), 20, seed, 'resnetap10')
    elif a.suite == 'references':
        config = ROOT/'configs/imagewoof.json'
        for ref_arch in ARCHS + ['wrn28_10', 'vit_tiny16']:
            if ref_arch == 'resnet18':
                ref = Path(load_config(config)['reference'])
            else:
                ref = root/'reference/imagewoof'/ref_arch/'final.pt'
                plan.add('ref:'+ref_arch, 'uag.train', '--config', config, '--role', 'reference',
                         '--arch', ref_arch, '--out', ref.parent)
            for ipc in [10, 20, 50]:
                for seed in a.seeds:
                    syn = generate('imagewoof', config, ipc, seed, 'ref_'+ref_arch, ref=ref)
                    plan.evaluate('imagewoof', config, syn,
                        root/'reference_ablation'/ref_arch/('ipc%d_seed%d'%(ipc, seed)), ipc, seed, 'resnetap10')
    elif a.suite == 'plugins':
        config = ROOT/'configs/imagewoof.json'
        cfg = load_config(config)
        for method in ['minimax', 'mgd3', 'ims3']:
            checkpoint = cfg['dit_checkpoint'] if method == 'mgd3' else cfg[method+'_checkpoint']
            for ipc in [10, 20, 50, 70, 100]:
                for seed in a.seeds:
                    for scale in [0, 10]:
                        tag = method+('_uag' if scale else '')
                        out = root/'plugins'/tag/('ipc%d_seed%d'%(ipc, seed))
                        args = ['--ckpt', checkpoint, '--save-dir', str(out), '--spec', 'woof', '--seed', str(seed)]
                        if method == 'mgd3':
                            args += ['--num-samples', str(ipc), '--guidance', '--stop_t', '25',
                                     '--imagenet_dir', cfg['data_root'], '--num-datasets', '1']
                            syn = out/'dataset_0'
                        elif method == 'ims3':
                            args += ['--ipc', str(ipc), '--groups', '5', '--real-train-dir', cfg['data_root']+'/train',
                                     '--w-real', '.4', '--w-sep', '.9', '--sel-eps', '0', '--sample-batch', '10']
                            syn = out/'final_distilled/train'
                        else:
                            args += ['--num-samples', str(ipc)]
                            syn = out
                        plan.add('plugin:'+str(out), 'uag.upstream', '--method', method, '--config', config,
                                 '--scale', scale, '--', *args)
                        # Preserve each base method's evaluator and default schedule.
                        for arch in ARCHS:
                            net, depth = {'convnet6': ('convnet', 6), 'resnetap10': ('resnet_ap', 10), 'resnet18': ('resnet', 18)}[arch]
                            eval_out = root/'plugin_evaluation'/tag/('ipc%d_seed%d_%s'%(ipc, seed, arch))
                            args = ['-d', 'imagenet', '--imagenet_dir', str(syn), cfg['data_root'],
                                    '-n', net, '--depth', str(depth), '--nclass', '10', '--norm_type', 'instance',
                                    '--ipc', str(ipc), '--tag', tag, '--slct_type', 'random', '--spec', 'woof',
                                    '--repeat', '1', '--seed', str(seed), '--save-dir', str(eval_out)]
                            if method == 'ims3':
                                args += ['--lr', '.1', '--randaug', 'true', '--randaug_n', '1', '--randaug_m', '6']
                            plan.add('plugin_eval:'+str(eval_out), 'uag.upstream', '--method', method,
                                     '--stage', 'evaluate', '--config', config, '--', *args)
    elif a.suite in ('diagnostics', 'redundancy'):
        config = ROOT/'configs/imagewoof.json'
        cfg = load_config(config)
        for seed in a.seeds:
            directory = root/a.suite/('seed%d'%seed)
            if a.suite == 'diagnostics':
                plan.add('real_features:'+str(directory), 'uag.diagnostics', 'features', '--config', config,
                         '--images', cfg['data_root']+'/train', '--out', directory/'real.npz')
            for method, scale in [('base', 0), ('uag', 10)]:
                syn = generate('imagewoof', config, 50, seed, 'diagnostic_'+method, scale)
                if a.suite == 'redundancy':
                    plan.add('redundancy:'+str(directory/method), 'uag.redundancy', '--config', config,
                             '--syn', syn, '--out', directory/method, '--seed', seed, '--execute')
                else:
                    student = directory/(method+'_student')
                    plan.evaluate('imagewoof', config, syn, student, 50, seed, 'resnet18')
                    plan.add('features:'+str(directory/method), 'uag.diagnostics', 'features', '--config', config,
                             '--images', syn, '--out', directory/(method+'.npz'), '--ipc', 50)
                    plan.add('utility:'+str(directory/method), 'uag.diagnostics', 'utility', '--config', config,
                             '--images', syn, '--out', directory/(method+'_utility'), '--ipc', 50, '--seed', seed,
                             '--checkpoints', *[student/('checkpoint_'+f+'.pt') for f in ['25', '50', '75']])
            if a.suite == 'diagnostics':
                plan.add('regions:'+str(directory), 'uag.diagnostics', 'regions', '--real', directory/'real.npz',
                         '--base', directory/'base.npz', '--guided', directory/'uag.npz',
                         '--base-utility', directory/'base_utility/utility.npz',
                         '--guided-utility', directory/'uag_utility/utility.npz', '--out', directory/'regions', '--seed', seed)
    elif a.suite == 'baselines':
        for dataset in a.datasets:
            config = ROOT/'configs'/(dataset+'.json')
            cfg = load_config(config)
            feats = root/'features'/(dataset+'.npz')
            plan.add('features:'+dataset, 'uag.diagnostics', 'features', '--config', config,
                     '--images', cfg['data_root']+'/train', '--out', feats)
            for ipc in MATRIX[dataset][0]:
                for seed in a.seeds:
                    for method in ['random', 'kcenter', 'herding']:
                        syn = root/'baselines'/dataset/method/('ipc%d_seed%d'%(ipc, seed))
                        plan.add('select:'+str(syn), 'uag.select', '--config', config, '--out', syn,
                                 '--method', method, '--ipc', ipc, '--seed', seed, '--features', feats)
                        for arch in MATRIX[dataset][1]:
                            plan.evaluate(dataset, config, syn, root/'baseline_evaluation'/dataset/method/
                                          ('ipc%d_seed%d_%s'%(ipc, seed, arch)), ipc, seed, arch)
    write_json(a.plan, dict(suite=a.suite, assumptions='docs/REPRODUCTION_NOTES.md', jobs=plan.jobs))
    print('Prepared %d commands. Plan: %s' % (len(plan.jobs), a.plan))
    if a.execute:
        logs = root/'logs'/a.suite
        logs.mkdir(parents=True, exist_ok=True)
        for i, job in enumerate(plan.jobs):
            print(job['name'], flush=True)
            with (logs/('%04d.log'%i)).open('a') as f:
                subprocess.run(job['command'], cwd=job['cwd'], stdout=f, stderr=subprocess.STDOUT, check=True)
    else:
        print('No generation, training or evaluation was executed.')


if __name__ == '__main__':
    main()
