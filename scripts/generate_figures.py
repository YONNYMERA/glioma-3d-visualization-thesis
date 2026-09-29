from pathlib import Path
import json
import re
import hashlib
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image

import argparse
parser = argparse.ArgumentParser(description="Regenerate the seven statistical thesis figures")
parser.add_argument('--results', type=Path, default=Path(__file__).resolve().parents[1]/'results')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
SRC = args.results.resolve()
OUT = args.output.resolve()
OUT.mkdir(parents=True, exist_ok=True)
DATA = pd.read_csv(SRC / 'comparacion_tres_modelos' / 'all_cases.csv')
SUMMARY = pd.read_csv(SRC / 'comparacion_tres_modelos' / 'summary_ci_and_size.csv')
VOLUME = pd.read_csv(SRC / 'comparacion_tres_modelos' / 'volumetric_agreement.csv').set_index('Model')
MODELS = ['unet_baseline', 'segresnet_baseline', 'nnunet']
NAMES = {'unet_baseline': '3D U-Net', 'segresnet_baseline': 'SegResNet', 'nnunet': 'nnU-Net'}
COLORS = {'unet_baseline': '#0072B2', 'segresnet_baseline': '#D55E00', 'nnunet': '#009E73'}
FOLD_COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#7C6A00']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 11,
                     'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
                     'legend.fontsize': 9, 'figure.facecolor': 'white', 'savefig.facecolor': 'white',
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.spines.top': False,
                     'axes.spines.right': False})
RECORDS = []

def save(fig, stem, caption):
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'{stem}.{ext}', dpi=300, bbox_inches='tight', pad_inches=.08)
    plt.close(fig)
    RECORDS.append({'stem': stem, 'caption_en': caption})

def grid(ax):
    ax.grid(axis='y', color='#E1E5E8', lw=.6)
    ax.set_axisbelow(True)

def patient_metric(frame, metric):
    vals = pd.to_numeric(frame[metric])
    finite = np.isfinite(vals)
    return frame.loc[finite].groupby('Patient_ID')[metric].mean().to_numpy()

def training_curves():
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.7), sharex=True, layout='constrained')
    for row, model in enumerate(MODELS[:2]):
        for fold, color in zip(range(1, 6), FOLD_COLORS):
            d = pd.read_csv(SRC / 'train' / model / f'fold_{fold:02d}' / 'training_log.csv')
            assert d.epoch.tolist() == list(range(1, 21)), (model, fold)
            axes[row, 0].plot(d.epoch, d.train_loss, color=color, lw=1.45, label=f'Fold {fold}')
            axes[row, 1].plot(d.epoch, d.inner_val_dice, color=color, lw=1.45)
        axes[row, 0].set_title(f'{NAMES[model]}: training loss')
        axes[row, 0].set_ylabel('Dice + cross-entropy loss')
        axes[row, 1].set_title(f'{NAMES[model]}: inner validation')
        axes[row, 1].set_ylabel('Mean volume Dice')
        axes[row, 1].set_ylim(0, 1)
        for ax in axes[row]:
            ax.set_xticks([1, 5, 10, 15, 20]); ax.set_xlim(1, 20); grid(ax)
    axes[0, 0].legend(ncol=3, loc='upper right')
    for ax in axes[1]: ax.set_xlabel('Completed epoch')
    save(fig, 'training_unet_segresnet',
         'Training loss and inner-validation whole-volume Dice for the five patient-grouped outer folds of 3D U-Net and SegResNet (20 epochs each). Each curve is one fold; the inner-validation scores select checkpoints and are not held-out evaluation scores. Loss is the implemented Dice plus cross-entropy objective; identical axis scales within the validation column allow comparison of the recorded trajectories.')

def nnunet_curves():
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.7), sharex=True, layout='constrained')
    parsed = []
    for fold, color in zip(range(1, 6), FOLD_COLORS):
        paths = sorted((SRC / 'nnunet' / 'nnUNet_results' / f'Dataset{700+fold}_GliomaOuter{fold}').rglob('training_log_*.txt'))
        epochs = {}
        for path in paths:
            epoch = None
            for line in path.read_text(encoding='utf-8').splitlines():
                match = re.search(r': Epoch (\d+)\s*$', line)
                if match:
                    epoch = int(match.group(1)); epochs.setdefault(epoch, {'epoch': epoch+1, 'fold': fold})
                if epoch is None: continue
                for key, pattern in [('train_loss', r': train_loss ([-+0-9.eE]+)'),
                                     ('val_loss', r': val_loss ([-+0-9.eE]+)'),
                                     ('pseudo_dice', r': Pseudo dice \[(?:np\.float32\()?([-+0-9.eE]+)'),
                                     ('new_best_ema', r'New best EMA pseudo Dice: ([-+0-9.eE]+)')]:
                    found = re.search(pattern, line)
                    if found: epochs[epoch][key] = float(found.group(1))
        d = pd.DataFrame([epochs[k] for k in sorted(epochs)])
        assert d.epoch.tolist() == list(range(1, 101)), (fold, d.epoch.tolist())
        assert d[['train_loss','val_loss','pseudo_dice']].notna().all().all()
        d['best_ema_so_far'] = d.new_best_ema.ffill()
        parsed.append(d)
        for ax, key in zip(axes.flat, ['train_loss', 'val_loss', 'pseudo_dice', 'best_ema_so_far']):
            if key == 'best_ema_so_far':
                ax.step(d.epoch, d[key], where='post', color=color, lw=1.3, label=f'Fold {fold}')
            else: ax.plot(d.epoch, d[key], color=color, lw=1.15, label=f'Fold {fold}')
    titles = ['Training loss', 'Inner-validation loss', 'Inner-validation pseudo Dice', 'Best EMA pseudo Dice so far']
    for i, (ax, title) in enumerate(zip(axes.flat, titles)):
        ax.set_title(title); ax.set_xlim(1,100); ax.set_xticks([1,20,40,60,80,100]); grid(ax)
        ax.set_ylabel('Logged loss' if i<2 else 'Pseudo Dice')
        if i>=2: ax.set_ylim(.75, 1.0); ax.set_xlabel('Completed epoch (native index + 1)')
    axes[0,0].legend(ncol=3, loc='upper right')
    pd.concat(parsed, ignore_index=True).to_csv(OUT/'nnunet_training_parsed.csv', index=False)
    save(fig, 'training_nnunet',
         'nnU-Net trajectories for the five outer folds (100 epochs). Native epoch indices 0--99 are displayed as completed epochs 1--100. Training/validation losses and patch-aggregated pseudo Dice come from the text logs; they are not the held-out whole-volume Dice used in the main comparison. Bottom right shows the best EMA pseudo Dice reached so far, obtained only from exact logged checkpoint-improvement values and held constant between improvements; it is not the complete EMA trajectory. Pseudo Dice values in the text logs are rounded to four decimals. Negative nnU-Net losses are expected from its loss formulation and must not be numerically compared with the MONAI loss.')

def metric_comparison():
    fig, axes = plt.subplots(2, 2, figsize=(9.1, 6.5), layout='constrained')
    specs = [('Dice', 'Whole-tumor Dice (higher is better)', (.875,.945)),
             ('SurfaceDice', 'Surface Dice at 1 mm (higher is better)', None),
             ('HD95_mm', 'HD95, finite values (mm; lower is better)', None),
             ('ASSD_mm', 'ASSD, finite values (mm; lower is better)', None)]
    for ax, (metric, title, lim) in zip(axes.flat, specs):
        for pos, model in enumerate(MODELS):
            row = SUMMARY[(SUMMARY.Model==model)&(SUMMARY.Size_group=='all')&(SUMMARY.Metric==metric)].iloc[0]
            mean, lo, hi = row[['mean_finite_patient','ci95_low','ci95_high']]
            ax.errorbar(pos, mean, yerr=[[mean-lo],[hi-mean]], color=COLORS[model], fmt='o', capsize=5, ms=7, lw=1.7)
            ax.annotate(f'{mean:.3f}' if metric in ('Dice','SurfaceDice') else f'{mean:.2f}',
                        (pos, hi), xytext=(0,6), textcoords='offset points', ha='center', fontsize=9)
        ax.set_title(title, fontsize=10); ax.set_xticks(range(3), [NAMES[m] for m in MODELS]); ax.set_xlim(-.45,2.45)
        if lim: ax.set_ylim(*lim)
        else: ax.margins(y=.35)
        ax.set_ylabel('Patient-level mean with 95% CI'); grid(ax)
    save(fig, 'metric_comparison',
         'Patient-level mean and 95% confidence interval for Dice, surface Dice at 1 mm, HD95 and ASSD. Repeated scans are averaged within each patient before 2,000 bootstrap resamples within the fixed outer folds. The intervals are conditional on the trained checkpoints, not retraining uncertainty. Distance means use finite scans: one empty U-Net prediction has infinite HD95/ASSD and is excluded from those two means, but remains a Dice=0 failure. All models cover the same 1,250 scans from 1,132 patients. The vertical axes are zoomed and do not imply a zero baseline.')

def confusion_matrices():
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), layout='constrained')
    records=[]
    for ax, model in zip(axes, MODELS):
        counts=DATA[DATA.Model==model][['TP','FP','TN','FN']].sum().astype('int64')
        a=np.array([[counts.TN,counts.FP],[counts.FN,counts.TP]], dtype=np.int64)
        norm=a/a.sum(axis=1,keepdims=True)*100
        im=ax.imshow(norm,vmin=0,vmax=100,cmap='Blues')
        for row in range(2):
            for col in range(2):
                ax.text(col,row,f'{norm[row,col]:.3f}%\n{a[row,col]:,}', ha='center',va='center',
                        fontsize=10,color='white' if norm[row,col]>60 else '#16324F')
                records.append({'Model':model,'Reference':['Background','Whole tumor'][row],
                                'Prediction':['Background','Whole tumor'][col],
                                'Voxel_count':int(a[row,col]),'Row_percent':float(norm[row,col])})
        ax.set_title(NAMES[model]); ax.set_xticks([0,1],['Background','Whole tumor'])
        ax.set_yticks([0,1],['Background','Whole tumor']); ax.set_xlabel('Predicted label')
        if ax is axes[0]: ax.set_ylabel('Reference label')
        for s in ax.spines.values(): s.set_visible(False)
    fig.colorbar(im,ax=axes,location='bottom',fraction=.08,shrink=.55,pad=.12,label='Percentage within each reference row')
    pd.DataFrame(records).to_csv(OUT/'confusion_counts.csv',index=False)
    save(fig, 'confusion_matrices',
         'Pooled voxel-level binary confusion matrices across all 1,250 held-out scans. Cell colors and percentages are normalized within each reference row; the second line gives the raw voxel count. Background includes the common RAS 1-mm foreground-cropped and divisibility-padded evaluation grid, not only brain parenchyma. These are segmentation counts, not patient-classification matrices; pooling weights scans by their number of voxels and differs from the patient-level macro averages in the results table.')

def distributions():
    fig,axes=plt.subplots(1,2,figsize=(10,4.1),layout='constrained')
    for ax,metric,title in zip(axes,['Dice','HD95_mm'],['Patient-level Dice','Finite patient-level HD95']):
        vals=[patient_metric(DATA[DATA.Model==m],metric) for m in MODELS]
        parts=ax.violinplot(vals, positions=range(3),showextrema=False,widths=.7)
        for body,model in zip(parts['bodies'],MODELS):
            body.set_facecolor(COLORS[model]); body.set_alpha(.22)
        box=ax.boxplot(vals,positions=range(3),widths=.22,patch_artist=True,
                       showfliers=True,flierprops={'marker':'.','markersize':2,'alpha':.3},
                       medianprops={'color':'black','linewidth':1.4})
        for patch,model in zip(box['boxes'],MODELS): patch.set_facecolor(COLORS[model]); patch.set_alpha(.7)
        ax.set_xticks(range(3),[NAMES[m] for m in MODELS]);ax.set_title(title);grid(ax)
    axes[0].set_ylim(-.03,1.02);axes[0].set_ylabel('Dice')
    axes[1].set_yscale('log');axes[1].set_ylabel('HD95 (mm; logarithmic scale)')
    save(fig, 'metric_distributions',
         'Distributions of patient-level Dice and finite HD95 after averaging repeated scans within each patient. Boxes show the median and interquartile range, whiskers extend to 1.5 times the interquartile range, and dots retain more extreme values; violin outlines are descriptive kernel-density estimates. HD95 uses a logarithmic axis and excludes the one infinite U-Net distance caused by an empty prediction. Dice includes that failure.')

def volumetry():
    fig,axes=plt.subplots(2,3,figsize=(11,7.1),layout='constrained')
    differences=[]
    for col,model in enumerate(MODELS):
        d=DATA[DATA.Model==model].groupby('Patient_ID')[['Vol_GT_mL','Vol_Pred_mL']].mean()
        ref=d.Vol_GT_mL.to_numpy();pred=d.Vol_Pred_mL.to_numpy();r=VOLUME.loc[model]
        color=COLORS[model]
        ax=axes[0,col];ax.scatter(ref,pred,s=8,alpha=.38,color=color,edgecolors='none',rasterized=True)
        ax.plot([0,375],[0,375],color='#555555',ls='--',lw=1)
        ax.set(xlim=(0,375),ylim=(0,375),xlabel='Reference volume (mL)',ylabel='Predicted volume (mL)',title=NAMES[model])
        ax.set_aspect('equal',adjustable='box')
        ax.text(.04,.96,f"Pearson r² = {r.Pearson_r_squared:.3f}\nPredictive R² = {r.predictive_R2:.3f}",
                transform=ax.transAxes,va='top',fontsize=9,bbox=dict(fc='white',ec='none',alpha=.9))
        x=(ref+pred)/2;delta=pred-ref;differences.extend(delta)
        ax=axes[1,col];ax.scatter(x,delta,s=8,alpha=.38,color=color,edgecolors='none',rasterized=True)
        ax.axhline(0,color='#CCCCCC',lw=.7)
        ax.axhline(r.BA_bias_mL,color='#333333',lw=1.2)
        for v in [r.BA_LoA_lower_mL,r.BA_LoA_upper_mL]:ax.axhline(v,color='#555555',ls='--',lw=1)
        ax.set(xlim=(0,375),xlabel='Mean of reference and prediction (mL)',ylabel='Prediction - reference (mL)')
        ax.text(.04,.97,f"Bias {r.BA_bias_mL:.2f} mL\nLoA [{r.BA_LoA_lower_mL:.2f}, {r.BA_LoA_upper_mL:.2f}] mL",
                transform=ax.transAxes,va='top',fontsize=9,bbox=dict(fc='white',ec='none',alpha=.9))
    ymin,ymax=min(differences),max(differences);pad=(ymax-ymin)*.16
    for ax in axes[1]:ax.set_ylim(ymin-pad,ymax+pad)
    save(fig,'volumetric_agreement',
         'Patient-level volumetric agreement for the three models (1,132 patients; repeated scans averaged first). Top: reference versus predicted whole-tumor volume with the identity line; squared Pearson correlation and predictive R² are distinct statistics. Bottom: Bland-Altman differences (prediction minus reference), mean bias (solid) and bias ±1.96 sample standard deviations (dashed; ddof=1). These limits describe observed agreement and are not clinical acceptance limits. All patients are shown, with common axes across models.')

def size_analysis():
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    groups=['small','medium','large']
    for ax,metric,title in zip(axes,['Dice','HD95_mm'],['Dice by reference tumor volume','Finite HD95 by reference tumor volume']):
        for j,model in enumerate(MODELS):
            rows=SUMMARY[(SUMMARY.Model==model)&(SUMMARY.Metric==metric)].set_index('Size_group').loc[groups]
            y=rows.mean_finite_patient.to_numpy();lo=rows.ci95_low.to_numpy();hi=rows.ci95_high.to_numpy()
            ax.errorbar(np.arange(3)+(j-1)*.16,y,yerr=[y-lo,hi-y],fmt='o',capsize=4,color=COLORS[model],label=NAMES[model],ms=5)
        ax.set_xticks(range(3),['<10 mL\n24 scans','10 to <50 mL\n297 scans','≥50 mL\n929 scans']);grid(ax)
        ax.set_title(title);ax.set_xlim(-.5,2.5);ax.set_xlabel('Exploratory size strata (not clinical categories)')
    axes[0].set_ylabel('Patient-level mean Dice with 95% CI');axes[0].set_ylim(.43,1.0);axes[0].legend(loc='lower right')
    axes[1].set_ylabel('Patient-level mean HD95 (mm) with 95% CI');axes[1].margins(y=.18)
    save(fig,'size_strata',
         'Patient-level means and bootstrap 95% confidence intervals within prespecified exploratory reference-volume strata: <10 mL (24 scans), 10 to <50 mL (297), and ≥50 mL (929). Scan counts are common to the three models. Repeated scans are averaged within each patient and stratum; a patient may contribute to more than one stratum. The small-volume stratum is sparse. Infinite distance values are omitted from the finite HD95 means, including one U-Net failure in the small stratum. These cutoffs are not clinical tumor categories.')

def main():
    assert not DATA.duplicated(['Model','Case_ID']).any()
    for model in MODELS:
        d=DATA[DATA.Model==model];assert len(d)==1250 and d.Patient_ID.nunique()==1132
    assert all(DATA.groupby('Case_ID')[c].nunique().max()==1 for c in ['Patient_ID','Fold','Vol_GT_mL'])
    training_curves();nnunet_curves();metric_comparison();confusion_matrices()
    distributions();volumetry();size_analysis()
    (OUT/'figure_manifest.json').write_text(json.dumps(RECORDS,indent=2),encoding='utf-8')
    print(json.dumps({'figures':len(RECORDS),'output':str(OUT),'names':[r['stem'] for r in RECORDS]},indent=2))

if __name__=='__main__':main()
