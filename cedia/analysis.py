"""Resúmenes auditables, bootstrap por paciente, estratos y comparaciones pareadas."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .manifest import write_json, load as load_manifest

METRICS = ['Dice', 'IoU', 'Sensitivity', 'Precision', 'Specificity', 'NPV', 'HD95_mm',
           'ASSD_mm', 'SurfaceDice', 'RVE_percent', 'Abs_RVE_percent', 'AVD_mL',
           'Vol_Error_mL', 'Inference_seconds', 'GPU_peak_allocated_MiB']


def patient_bootstrap(frame, metric, n_bootstrap, seed):
    # Average repeat scans within each patient. Only finite cases enter conditional estimates.
    if n_bootstrap < 1:
        raise ValueError('At least one bootstrap replicate required')
    if frame[['Patient_ID', 'Fold']].isna().any().any():
        raise ValueError('Patient and fold identifiers must be present')
    if (frame.groupby('Patient_ID')['Fold'].nunique() > 1).any():
        raise ValueError('A patient appears in multiple held-out folds')
    values = pd.to_numeric(frame[metric], errors='coerce')
    finite = np.isfinite(values)
    patients = frame.loc[finite, ['Patient_ID', 'Fold']].copy()
    patients[metric] = values[finite]
    patients = patients.groupby(['Patient_ID', 'Fold'], as_index=False)[metric].mean()
    if patients.empty:
        return {'n_cases': len(frame), 'n_finite_cases': 0, 'n_patients_finite': 0,
                'n_infinite': int(np.isinf(values).sum()), 'n_undefined': int(values.isna().sum()),
                'mean_finite_patient': np.nan, 'sd_finite_patient': np.nan,
                'median_finite_patient': np.nan, 'ci95_low': np.nan, 'ci95_high': np.nan}
    rng = np.random.default_rng(seed)
    groups = [g[metric].to_numpy() for _, g in patients.groupby('Fold')]
    means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        means[i] = np.concatenate([rng.choice(x, len(x), replace=True) for x in groups]).mean()
    vals = patients[metric].to_numpy()
    low, high = np.percentile(means, [2.5, 97.5]) if len(vals) > 1 else (np.nan, np.nan)
    return {'n_cases': len(frame), 'n_finite_cases': int(finite.sum()), 'n_patients_finite': len(vals),
            'n_infinite': int(np.isinf(values).sum()), 'n_undefined': int(values.isna().sum()),
            'mean_finite_patient': vals.mean(), 'sd_finite_patient': vals.std(ddof=1) if len(vals) > 1 else np.nan,
            'median_finite_patient': np.median(vals), 'ci95_low': low, 'ci95_high': high}


def read_results(inputs):
    paths = []
    for item in inputs:
        path = Path(item)
        paths.extend(sorted(path.glob('fold_*/metrics.csv')) if path.is_dir() else [path])
    if not paths:
        raise ValueError('No fold_*/metrics.csv found')
    # Strings preserve identifiers such as '001'; do not silently merge them with '1'.
    df = pd.concat([pd.read_csv(p, dtype={'Case_ID': str, 'Patient_ID': str, 'Model': str,
                                         'Manifest_SHA256': str}) for p in paths], ignore_index=True)
    required = ['Model', 'Case_ID', 'Patient_ID', 'Fold', 'Manifest_SHA256', 'Evidence',
                'Vol_GT_mL', 'Vol_Pred_mL', 'Dice', 'HD95_mm', 'Timing_scope']
    missing = sorted(set(required) - set(df.columns))
    if missing or df.empty:
        raise ValueError(f'Nonempty result tables with required columns expected; missing: {missing}')
    if df[required[:6] + ['Timing_scope']].isna().any().any():
        raise ValueError('Result identifiers, evidence and timing scope must be present')
    if not df['Model'].str.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*').all():
        raise ValueError('Model names must use letters, digits, underscore, period or hyphen')
    folds = pd.to_numeric(df['Fold'], errors='coerce')
    if not (np.isfinite(folds) & (folds >= 1) & (folds == np.floor(folds))).all():
        raise ValueError('Fold identifiers must be positive integers')
    df['Fold'] = folds.astype(int)
    if df.duplicated(['Model', 'Case_ID']).any():
        raise ValueError('Duplicate model/case results: each case must be evaluated exactly once per model')
    for col in METRICS:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in ['Vol_GT_mL', 'Vol_Pred_mL']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if not (np.isfinite(df[col]) & (df[col] >= 0)).all():
            raise ValueError('Reference and predicted volumes must be finite and nonnegative')
    if (df.groupby(['Model', 'Patient_ID'])['Fold'].nunique() > 1).any():
        raise ValueError('A patient appears in multiple held-out folds within one model')
    if (df.groupby('Model')['Timing_scope'].nunique() > 1).any():
        raise ValueError('Mixed timing protocols within one model cannot be summarized together')
    if 'SurfaceDice' in df:
        if 'surface_tolerance_mm' not in df:
            raise ValueError('SurfaceDice requires surface_tolerance_mm provenance')
        tolerance = pd.to_numeric(df['surface_tolerance_mm'], errors='coerce')
        if not (np.isfinite(tolerance) & (tolerance >= 0)).all() or tolerance.nunique() != 1:
            raise ValueError('SurfaceDice comparisons require one common finite surface tolerance')
        df['surface_tolerance_mm'] = tolerance
    return df


def validate_pairs(df):
    """Fail before producing outputs if pairing or the evaluation cohort differs."""
    names = sorted(df.Model.unique())
    if len(names) < 2:
        return
    left = df[df.Model == names[0]].set_index('Case_ID')
    for name in names[1:]:
        right = df[df.Model == name].set_index('Case_ID')
        if set(left.index) != set(right.index):
            raise ValueError('Architecture comparison requires exactly the same evaluated cases; finish missing folds')
        right = right.loc[left.index]
        if not left[['Patient_ID', 'Fold']].equals(right[['Patient_ID', 'Fold']]):
            raise ValueError('Architecture comparison requires identical patient and fold assignments')
        if not np.allclose(left.Vol_GT_mL, right.Vol_GT_mL, rtol=0, atol=1e-6):
            raise ValueError('Architecture comparison has inconsistent reference geometry')
    for metric in ['Dice', 'HD95_mm', 'ASSD_mm', 'SurfaceDice', 'AVD_mL']:
        if metric not in df:
            raise ValueError(f'Architecture comparison requires metric: {metric}')


def validate_complete_cohort(df, manifest_path):
    """Verify the complete reserved cohort and its assignments for every model."""
    manifest = load_manifest(manifest_path)
    expected_sha = manifest.get('manifest_sha256')
    if not expected_sha or not df['Manifest_SHA256'].eq(expected_sha).all():
        raise ValueError('Results do not match the supplied manifest SHA256')
    patients = {str(c['case_id']): str(c['patient_id']) for c in manifest['cases']}
    folds = {str(cid): int(f['fold']) for f in manifest['folds'] for cid in f['heldout']}
    expected_ids = set(patients)
    for model, rows in df.groupby('Model'):
        observed = set(rows['Case_ID'])
        if observed != expected_ids or len(rows) != len(expected_ids):
            missing, extra = expected_ids - observed, observed - expected_ids
            raise ValueError(f'Complete manifest cohort required for {model}: '
                             f'{len(missing)} missing cases, {len(extra)} unexpected cases; finish all folds')
        expected_patients = rows['Case_ID'].map(patients)
        expected_folds = rows['Case_ID'].map(folds)
        if not rows['Patient_ID'].eq(expected_patients).all() or not rows['Fold'].eq(expected_folds).all():
            raise ValueError(f'Patient or held-out fold assignment differs from the supplied manifest: {model}')
    return {'complete_cohort_verified': True, 'expected_cases_per_model': len(expected_ids),
            'expected_heldout_folds': sorted(set(folds.values()))}


def analyze(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    df = read_results(args.inputs)
    if df['Evidence'].str.contains('UNVERIFIED').any() and not args.include_unverified:
        raise ValueError('Unverified original splits present. --include-unverified creates diagnostic-only summaries.')
    if len(df['Manifest_SHA256'].unique()) != 1:
        raise ValueError('Cannot compare runs from different manifests')
    if args.bootstrap < 100:
        raise ValueError('Use at least 100 bootstrap replicates (recommended 2000)')
    if (len(args.size_cutoffs_ml) != 2 or not np.isfinite(args.size_cutoffs_ml).all() or
            not 0 < args.size_cutoffs_ml[0] < args.size_cutoffs_ml[1]):
        raise ValueError('Two increasing positive size thresholds required')
    validate_pairs(df)
    cohort = (validate_complete_cohort(df, args.manifest) if getattr(args, 'manifest', None)
              else {'complete_cohort_verified': False, 'expected_cases_per_model': None,
                    'expected_heldout_folds': None})
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    df['Size_group'] = pd.cut(df['Vol_GT_mL'], [-np.inf, *args.size_cutoffs_ml, np.inf],
                              labels=['small', 'medium', 'large'], right=False)
    df.to_csv(out / 'all_cases.csv', index=False)
    rows, outliers, volume = [], [], []
    for name, model_df in df.groupby('Model'):
        for group, sub in [('all', model_df)] + [(str(k), v) for k, v in model_df.groupby('Size_group', observed=False)]:
            for metric in METRICS:
                if metric in sub:
                    rows.append({'Model': name, 'Size_group': group, 'Metric': metric,
                                 **patient_bootstrap(sub, metric, args.bootstrap, args.seed)})
        q1, q3 = model_df['Dice'].quantile([.25, .75])
        finite_hd = model_df.loc[np.isfinite(model_df['HD95_mm']), 'HD95_mm']
        hq1, hq3 = finite_hd.quantile([.25, .75])
        difficult = model_df[~np.isfinite(model_df['HD95_mm']) |
                             (model_df['Dice'] < q1 - 1.5 * (q3-q1)) |
                             (model_df['HD95_mm'] > hq3 + 1.5 * (hq3-hq1))].copy()
        difficult['Review_reason'] = 'Nonfinite HD95 and/or Dice lower Tukey fence and/or HD95 upper Tukey fence; morphology requires image review'
        outliers.append(difficult)
        model_df.nsmallest(min(20, len(model_df)), 'Dice').to_csv(out / f'{name}_lowest_dice.csv', index=False)
        patient = model_df.groupby('Patient_ID')[['Vol_GT_mL', 'Vol_Pred_mL']].mean()
        x, y = patient['Vol_GT_mL'].to_numpy(), patient['Vol_Pred_mL'].to_numpy()
        delta, avg = y-x, (x+y)/2
        bias, sd = delta.mean(), delta.std(ddof=1) if len(delta) > 1 else np.nan
        corr2 = float(np.corrcoef(x, y)[0, 1]**2) if len(x)>1 and np.std(x)>0 and np.std(y)>0 else np.nan
        predictive_r2 = 1 - np.sum((x-y)**2)/np.sum((x-x.mean())**2) if len(x)>1 and np.std(x)>0 else np.nan
        volume.append({'Model': name, 'n_patients': len(patient), 'Pearson_r_squared': corr2,
                       'predictive_R2': predictive_r2, 'BA_bias_mL': bias,
                       'BA_LoA_lower_mL': bias-1.96*sd, 'BA_LoA_upper_mL': bias+1.96*sd,
                       'SD_definition': 'sample ddof=1; notebook used ddof=0'})
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        axes[0].scatter(x,y,s=12,alpha=.6)
        lim = max(float(x.max()),float(y.max()),1.)
        axes[0].plot([0,lim],[0,lim],'k--')
        axes[0].set(xlabel='Reference volume (mL)', ylabel='Predicted volume (mL)', title=f'{name}: r²={corr2:.3f}')
        axes[1].scatter(avg,delta,s=12,alpha=.6)
        for val in (bias, bias-1.96*sd, bias+1.96*sd):
            axes[1].axhline(val, color='black', linestyle='--')
        axes[1].set(xlabel='Mean volume (mL)', ylabel='Prediction - reference (mL)', title='Bland–Altman: descriptive limits')
        fig.tight_layout()
        fig.savefig(out / f'{name}_volumetry.png', dpi=240)
        plt.close(fig)
        fig, axes = plt.subplots(1, 3, figsize=(12,4))
        for ax, metric in zip(axes, ['Dice','HD95_mm','AVD_mL']):
            groups = [model_df.loc[model_df.Size_group == g,metric].replace([np.inf,-np.inf],np.nan).dropna() for g in ['small','medium','large']]
            ax.boxplot(groups, tick_labels=['small','medium','large'])
            ax.set_title(metric + ' (finite cases)')
        fig.tight_layout(); fig.savefig(out / f'{name}_size_strata.png',dpi=240); plt.close(fig)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / 'summary_ci_and_size.csv',index=False)
    pd.concat(outliers,ignore_index=True).to_csv(out / 'outliers_for_review.csv',index=False)
    pd.DataFrame(volume).to_csv(out / 'volumetric_agreement.csv',index=False)
    paired = []
    names = sorted(df.Model.unique())
    for ia, a in enumerate(names):
        for b in names[ia+1:]:
            left = df[df.Model==a].set_index('Case_ID')
            right = df[df.Model==b].set_index('Case_ID').loc[left.index]
            for metric in ['Dice','HD95_mm','ASSD_mm','SurfaceDice','AVD_mL']:
                diff = left[['Patient_ID','Fold']].copy()
                finite_left, finite_right = np.isfinite(left[metric]), np.isfinite(right[metric])
                both = finite_left & finite_right
                diff[metric] = np.nan
                diff.loc[both, metric] = left.loc[both, metric] - right.loc[both, metric]
                paired.append({'Comparison': f'{a} minus {b}', 'Metric':metric,
                               'n_nonfinite_left_cases': int((~finite_left).sum()),
                               'n_nonfinite_right_cases': int((~finite_right).sum()),
                               'n_both_finite_cases': int(both.sum()),
                               **patient_bootstrap(diff,metric,args.bootstrap,args.seed)})
    if paired:
        pd.DataFrame(paired).to_csv(out / 'paired_patient_differences.csv',index=False)
    overall = summary[summary.Size_group=='all']
    fig, axes = plt.subplots(1,4,figsize=(15,4))
    scopes = df.groupby('Model')['Timing_scope'].first()
    comparable_timing = len(scopes.unique()) == 1
    for ax, metric in zip(axes,['Dice','HD95_mm','Inference_seconds','GPU_peak_allocated_MiB']):
        if metric in ['Inference_seconds','GPU_peak_allocated_MiB'] and not comparable_timing:
            ax.text(.5,.5,'Timing protocols differ\nSee run.json and timing CSV',ha='center',va='center')
            ax.axis('off'); continue
        sub = overall[overall.Metric==metric].set_index('Model').reindex(names)
        ax.bar(names,sub.mean_finite_patient)
        ax.set_title(metric + '\nfinite-patient mean')
        ax.tick_params(axis='x', rotation=25)
    fig.tight_layout()
    for ext in ('png','pdf'):
        fig.savefig(out / f'architecture_comparison.{ext}',dpi=240)
    plt.close(fig)
    write_json(out / 'analysis_protocol.json', {**cohort, 'bootstrap_replicates':args.bootstrap,'seed':args.seed,
               'bootstrap_unit':'patient; scan means first; resample patients within fixed outer folds',
               'ci_interpretation':'conditional on trained checkpoints; does not include retraining variance',
               'nonfinite_policy':'exclude nonfinite scans per metric, then average available finite scans per patient; explicit case failure/undefined counts accompany each metric; paired differences require both scan values finite',
               'surface_metrics':'HD95=max of directed voxel-center percentiles; ASSD=pooled directed voxel-center distances; SurfaceDice=MONAI area-weighted subvoxel surfaces at saved tolerance',
               'size_cutoffs_ml':args.size_cutoffs_ml,'size_cutoffs_interpretation':'prespecified exploratory thresholds, not clinical categories',
               'diagnostic_unverified':bool(df.Evidence.str.contains('UNVERIFIED').any()),
               'n_cases_by_model':{str(k):int(v) for k,v in df.groupby('Model').size().items()},
               'timing_comparable':comparable_timing,'manifest_sha256':str(df.Manifest_SHA256.iloc[0])})
    print(f'Analysis saved: {out}')
