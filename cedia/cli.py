"""Entrada única: python -m cedia.cli --help, desde la raíz del proyecto."""
import argparse
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description='Correcciones experimentales de glioma, terminal/CEDIA')
    sub = p.add_subparsers(dest='command',required=True)
    def dataset(q):
        q.add_argument('--data-root',required=True,type=Path)
        q.add_argument('--manifest',required=True,type=Path)
    def fold(q):
        dataset(q); q.add_argument('--fold',required=True,type=int)
        q.add_argument('--output',required=True,type=Path)
        q.add_argument('--resume',action='store_true')
    def network(q):
        q.add_argument('--arch',choices=['unet','segresnet'],default='unet')
        q.add_argument('--roi',nargs=3,type=int,default=[128,128,128])
        q.add_argument('--device',default='cuda')
    def eval_options(q):
        q.add_argument('--tolerance-mm',type=float,default=1.)
        q.add_argument('--figures-per-fold',type=int,default=3)
    q = sub.add_parser('prepare',help='Inventario + splits persistidos; NO entrena')
    q.add_argument('--data-root',required=True,type=Path); q.add_argument('--output',required=True,type=Path)
    q.add_argument('--folds',type=int,default=5); q.add_argument('--seed',type=int,default=0)
    q.add_argument('--inner-fraction',type=float,default=.15)
    q.add_argument('--original-csv',type=Path); q.add_argument('--patient-map',type=Path)
    q = sub.add_parser('preflight',help='Valida geometría, archivos y entorno')
    dataset(q); q.add_argument('--output',type=Path); q.add_argument('--deep',action='store_true')
    q.add_argument('--require-cuda',action='store_true')
    q = sub.add_parser('train',help='Entrena un fold; nunca evalúa outer holdout')
    fold(q); network(q)
    q.add_argument('--epochs',type=int,default=20); q.add_argument('--batch-size',type=int,default=4)
    q.add_argument('--lr',type=float,default=1e-4); q.add_argument('--workers',type=int,default=4)
    q.add_argument('--scheduler',choices=['none','plateau'],default='none')
    q.add_argument('--patience',type=int,default=0,help='Early stopping: 0 deshabilita')
    q = sub.add_parser('evaluate',help='Checkpoint de UNet/SegResNet sobre su propio outer fold')
    fold(q); network(q); eval_options(q)
    q.add_argument('--checkpoint',required=True,type=Path); q.add_argument('--name')
    q.add_argument('--save-predictions',action='store_true')
    q.add_argument('--allow-unverified-original-splits',action='store_true')
    q = sub.add_parser('analyze',help='Bootstrap, tamaño, outliers, figuras y comparación')
    q.add_argument('--inputs',nargs='+',required=True,type=Path); q.add_argument('--output',required=True,type=Path)
    q.add_argument('--manifest',type=Path,help='Verifica todos los casos/folds/pacientes del análisis final; sin esta opción es una vista preliminar')
    q.add_argument('--bootstrap',type=int,default=2000); q.add_argument('--seed',type=int,default=20260923)
    q.add_argument('--size-cutoffs-ml',nargs=2,type=float,default=[10.,50.])
    q.add_argument('--include-unverified',action='store_true')
    q = sub.add_parser('nnunet-export',help='Crea dataset independiente por outer fold')
    dataset(q); q.add_argument('--output',required=True,type=Path)
    q.add_argument('--dataset-base',type=int,default=700); q.add_argument('--link-mode',choices=['symlink','copy'],default='symlink')
    q = sub.add_parser('nnunet-install-split',help='Instala inner split tras preprocessing')
    q.add_argument('--dataset-dir',required=True,type=Path); q.add_argument('--preprocessed-dir',required=True,type=Path)
    q = sub.add_parser('nnunet-predict',help='Predice heldout con checkpoint_best de inner fold0')
    fold(q); q.add_argument('--trained-model',required=True,type=Path)
    q.add_argument('--dataset-dir',required=True,type=Path); q.add_argument('--device',default='cuda')
    q = sub.add_parser('nnunet-train',help='Verifica inner split y registra procedencia antes de entrenar')
    dataset(q); q.add_argument('--fold',required=True,type=int)
    q.add_argument('--dataset-dir',required=True,type=Path)
    q.add_argument('--trainer',default='nnUNetTrainer_100epochs')
    q.add_argument('--device',default='cuda',choices=['cuda','cpu','mps'])
    q.add_argument('--resume',action='store_true')
    q = sub.add_parser('nnunet-evaluate',help='Evalúa predicción nnU-Net en la misma rejilla de referencia')
    fold(q); eval_options(q); q.add_argument('--prediction-root',required=True,type=Path)
    return p


def main(argv=None):
    p = parser(); args = p.parse_args(argv)
    if hasattr(args,'roi') and (min(args.roi)<16 or any(n%16 for n in args.roi)):
        p.error('--roi must contain multiples of 16, minimum 16')
    for option in ('epochs','batch_size'):
        if hasattr(args,option) and getattr(args,option)<1:
            p.error(f'--{option.replace("_","-")} must be positive')
    if hasattr(args,'fold') and args.fold < 1:
        p.error('--fold must be positive')
    if hasattr(args,'figures_per_fold') and args.figures_per_fold < 0:
        p.error('--figures-per-fold must be nonnegative')
    if hasattr(args,'workers') and args.workers < 0:
        p.error('--workers must be nonnegative')
    if args.command=='prepare':
        from .manifest import prepare
        m=prepare(args.data_root,args.output,args.folds,args.seed,args.inner_fraction,args.original_csv,args.patient_map)
        print(f'{len(m["cases"])} cases, {len(m["folds"])} folds, provenance={m["provenance"]}')
    elif args.command in ('preflight','train','evaluate'):
        from . import runner
        getattr(runner,args.command)(args)
    elif args.command=='analyze':
        from .analysis import analyze
        analyze(args)
    else:
        from . import nnunet
        getattr(nnunet,{'nnunet-export':'export','nnunet-install-split':'install_split',
                        'nnunet-train':'train','nnunet-predict':'predict','nnunet-evaluate':'evaluate'}[args.command])(args)


if __name__=='__main__':
    main()
