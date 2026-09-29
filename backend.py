import os
import subprocess
import glob
import time
import re
import sys
from pathlib import Path

import torch
import numpy as np
import monai
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    NormalizeIntensityd, ToTensord
)
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference
from monai.data import Dataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATTERN = "best_metric_model_fold_*.pth"
MODEL_DIR_NAME = "MODELS"

# =========================
# HELPERS: BRA TS PATHS
# =========================
def _get_patient_prefix(patient_folder: str) -> str:
    files = os.listdir(patient_folder)
    pref = None
    for f in files:
        if f.endswith("-t1n.nii.gz"):
            pref = f.replace("-t1n.nii.gz", "")
            break
    if pref is None:
        raise FileNotFoundError("No BraTS files (-t1n.nii.gz) were found in the folder.")
    return pref


def get_patient_paths(patient_folder: str) -> dict:
    pref = _get_patient_prefix(patient_folder)
    paths = {
        "t1n": os.path.join(patient_folder, f"{pref}-t1n.nii.gz"),
        "t1c": os.path.join(patient_folder, f"{pref}-t1c.nii.gz"),
        "t2w": os.path.join(patient_folder, f"{pref}-t2w.nii.gz"),
        "t2f": os.path.join(patient_folder, f"{pref}-t2f.nii.gz"),
    }
    for k, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing {k} file: {p}")
    return paths


# =========================
# MONAI TRANSFORMS
# =========================
def get_transforms_image_4ch():
    return monai.transforms.Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        Orientationd(keys=["image"], axcodes="RAS"),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        ToTensord(keys=["image"]),
    ])


def get_transforms_mask_like_image():
    # Load a BET image or mask and resample it to 1 mm RAS.
    return monai.transforms.Compose([
        LoadImaged(keys=["vol"]),
        EnsureChannelFirstd(keys=["vol"]),
        Spacingd(keys=["vol"], pixdim=(1.0, 1.0, 1.0), mode="nearest"),
        Orientationd(keys=["vol"], axcodes="RAS"),
        ToTensord(keys=["vol"]),
    ])


def _load_nii_1mm_RAS(path_nii: str) -> np.ndarray:
    ds = Dataset(data=[{"vol": path_nii}], transform=get_transforms_mask_like_image())
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    for batch in loader:
        return batch["vol"].cpu().numpy()[0, 0]  # (X,Y,Z)
    raise RuntimeError(f"Could not load NIfTI file: {path_nii}")


# =========================
# MODEL
# =========================
def get_runtime_root() -> Path:
    """Return the folder that should contain external app resources."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def get_default_models_dir() -> str:
    """Default MODELS folder, next to the script/exe when possible."""
    candidates = [
        get_runtime_root() / MODEL_DIR_NAME,
        Path.cwd() / MODEL_DIR_NAME,
        Path(__file__).resolve().parent / MODEL_DIR_NAME,
    ]

    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / MODEL_DIR_NAME)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return str(get_runtime_root() / MODEL_DIR_NAME)


def _fold_sort_key(path: str):
    match = re.search(r"fold[_-]?(\d+)", os.path.basename(path), flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return os.path.basename(path).lower()


def find_model_paths(model_source: str | None = None) -> list[str]:
    """Find one or more checkpoints. Directories are scanned for fold models."""
    if model_source is None:
        model_source = get_default_models_dir()

    model_source = os.path.abspath(model_source)

    if os.path.isfile(model_source):
        return [model_source]

    if not os.path.isdir(model_source):
        raise FileNotFoundError(
            f"Model folder does not exist: {model_source}\n"
            f"Create a MODELS folder and place the {MODEL_PATTERN} files there."
        )

    model_paths = glob.glob(os.path.join(model_source, MODEL_PATTERN))
    model_paths = sorted(model_paths, key=_fold_sort_key)

    if not model_paths:
        raise FileNotFoundError(
            f"No models were found in: {model_source}\n"
            f"Expected filename pattern: {MODEL_PATTERN}"
        )

    return model_paths


class ProbabilityUNet(torch.nn.Module):
    """Single UNet wrapper that returns probabilities instead of logits."""

    def __init__(self, model: torch.nn.Module, model_paths: list[str]):
        super().__init__()
        self.model = model
        self.model_paths = model_paths
        self.n_models = len(model_paths)

    def forward(self, inputs):
        return torch.sigmoid(self.model(inputs))


class EnsembleProbabilityUNet(torch.nn.Module):
    """Average probabilities from all fold checkpoints."""

    def __init__(self, models: list[torch.nn.Module], model_paths: list[str]):
        super().__init__()
        self.models = torch.nn.ModuleList(models)
        self.model_paths = model_paths
        self.n_models = len(model_paths)

    def forward(self, inputs):
        prob_sum = None

        for model in self.models:
            probs = torch.sigmoid(model(inputs))
            prob_sum = probs if prob_sum is None else prob_sum + probs

        return prob_sum / float(len(self.models))


def _create_unet():
    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="batch",
    ).to(DEVICE)
    return model


def _load_state_dict(model_path: str):
    try:
        return torch.load(model_path, map_location=DEVICE, weights_only=True)
    except TypeError:
        return torch.load(model_path, map_location=DEVICE)


def load_model(model_source=None):
    """Load all fold models found in MODELS and return an ensemble predictor."""
    model_paths = find_model_paths(model_source)
    models = []

    try:
        for model_path in model_paths:
            model = _create_unet()
            model.load_state_dict(_load_state_dict(model_path))
            model.eval()
            models.append(model)

        if len(models) == 1:
            predictor = ProbabilityUNet(models[0], model_paths).to(DEVICE)
        else:
            predictor = EnsembleProbabilityUNet(models, model_paths).to(DEVICE)

        predictor.eval()
        print(f"Loaded models ({len(model_paths)}):")
        for model_path in model_paths:
            print(f"  - {model_path}")
        return predictor

    except Exception as e:
        raise RuntimeError(f"Error loading models: {e}")


# =========================
# HD-BET wrapper
# =========================
def _run_hdbet_single_file(
    input_nii: str,
    output_file_nii_gz: str,
    hdbet_exe: str,
    device: str = "cpu",
):
    """
    This HD-BET command expects:
      -i file.nii.gz
      -o file.nii.gz  (not a directory)
    """
    out_parent = os.path.dirname(output_file_nii_gz)
    os.makedirs(out_parent, exist_ok=True)

    cmd = [
        hdbet_exe,
        "-i", input_nii,
        "-o", output_file_nii_gz,
        "-device", device,
        "--disable_tta",
        "--save_bet_mask",
        # Keep the BET image output because this workflow uses *_bet.nii.gz.
    ]

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"HD-BET failed.\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDERR:\n{r.stderr}\n"
            f"STDOUT:\n{r.stdout}"
        )


def _list_cache_files(cache_dir: str) -> list[str]:
    try:
        return sorted(os.listdir(cache_dir))
    except Exception:
        return []


def _pick_latest(pattern: str) -> str | None:
    hits = glob.glob(pattern)
    if not hits:
        return None
    hits.sort(key=os.path.getmtime, reverse=True)
    return hits[0]


def extract_brain_mask_external(
    patient_folder: str,
    tool: str = "hdbet",
    input_modality: str = "t1n",
    hdbet_exe: str = "hd-bet",
    hdbet_device: str = "cpu",
    cache_dir: str | None = None
) -> np.ndarray:
    paths = get_patient_paths(patient_folder)
    input_nii = paths[input_modality]

    if cache_dir is None:
        cache_dir = os.path.join(patient_folder, "_brainmask_cache")
    os.makedirs(cache_dir, exist_ok=True)

    pref = _get_patient_prefix(patient_folder)

    if tool.lower() not in ("hdbet", "hd-bet", "hd_bet"):
        raise ValueError("This backend version implements tool='hdbet'.")

    # Base output path. HD-BET then generates *_bet.nii.gz.
    out_file = os.path.join(cache_dir, f"{pref}_hdbet_{input_modality}.nii.gz")

    # Reuse previous BET output when it already exists.
    existing_bet = _pick_latest(os.path.join(cache_dir, f"{pref}*_bet.nii.gz"))
    existing_mask = _pick_latest(os.path.join(cache_dir, f"{pref}*bet_mask*.nii.gz"))

    if existing_mask is None and existing_bet is None:
        _run_hdbet_single_file(
            input_nii=input_nii,
            output_file_nii_gz=out_file,
            hdbet_exe=hdbet_exe,
            device=hdbet_device,
        )
        time.sleep(0.2)

    # 1) Use an explicit mask if one exists.
    mask_path = _pick_latest(os.path.join(cache_dir, f"{pref}*bet_mask*.nii.gz"))
    if mask_path and os.path.exists(mask_path):
        m = _load_nii_1mm_RAS(mask_path)
        return (m > 0.5).astype(np.uint8)

    # 2) If no mask exists, use the BET image: mask = bet > 0.
    bet_path = _pick_latest(os.path.join(cache_dir, f"{pref}*_bet.nii.gz"))
    if bet_path and os.path.exists(bet_path):
        bet = _load_nii_1mm_RAS(bet_path)
        # Robust mask: outside-brain voxels are usually 0.
        return (bet > 0).astype(np.uint8)

    # 3) General fallback: any BET file in cache.
    any_bet = _pick_latest(os.path.join(cache_dir, "*_bet.nii.gz"))
    if any_bet and os.path.exists(any_bet):
        bet = _load_nii_1mm_RAS(any_bet)
        return (bet > 0).astype(np.uint8)

    # 4) Final diagnostic: list actual cache files.
    files = _list_cache_files(cache_dir)
    raise FileNotFoundError(
        "HD-BET finished but neither '*bet_mask*.nii.gz' nor '*_bet.nii.gz' was found in the cache.\n"
        f"Cache: {cache_dir}\n"
        f"Cache files: {files}"
    )


# =========================
# TUMOR SEGMENTATION
# =========================
def segment_patient(
    model,
    patient_folder,
    brain_tool: str = "none",
    brain_input_modality: str = "t1n",
    hdbet_exe: str = "hd-bet",
    hdbet_device: str = "cpu",
):
    paths = get_patient_paths(patient_folder)

    data_dict = [{"image": [paths["t1n"], paths["t1c"], paths["t2w"], paths["t2f"]]}]
    ds = Dataset(data=data_dict, transform=get_transforms_image_4ch())
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    with torch.no_grad():
        for batch in loader:
            inputs = batch["image"].to(DEVICE)

            outputs = sliding_window_inference(
                inputs,
                roi_size=(128, 128, 128),
                sw_batch_size=1,
                predictor=model,
                overlap=0.5
            )

            outputs = (outputs > 0.5).float()

            img_numpy = inputs.cpu().numpy()[0, 3, :, :, :]   # FLAIR
            seg_numpy = outputs.cpu().numpy()[0, 0, :, :, :]  # tumor

            brain_mask_numpy = None
            if brain_tool and brain_tool.lower() != "none":
                brain_mask_numpy = extract_brain_mask_external(
                    patient_folder=patient_folder,
                    tool="hdbet",
                    input_modality=brain_input_modality,
                    hdbet_exe=hdbet_exe,
                    hdbet_device=hdbet_device,
                )

                if brain_mask_numpy.shape != img_numpy.shape:
                    raise RuntimeError(
                        f"Brain mask shape {brain_mask_numpy.shape} != img shape {img_numpy.shape}. "
                        "Check transforms/spacing."
                    )

            if brain_mask_numpy is None:
                return img_numpy, seg_numpy
            return img_numpy, seg_numpy, brain_mask_numpy

    raise RuntimeError("Patient inference could not be completed.")


def calculate_volume(segmentation_mask, voxel_size=(1.0, 1.0, 1.0)):
    voxel_vol_mm3 = voxel_size[0] * voxel_size[1] * voxel_size[2]
    total_voxels = float(np.sum(segmentation_mask))
    volume_mm3 = total_voxels * voxel_vol_mm3
    return volume_mm3 / 1000.0


def tumor_stats(seg_mask: np.ndarray):
    pts = np.argwhere(seg_mask > 0)
    if pts.size == 0:
        return None

    centroid = pts.mean(axis=0)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)

    return {
        "centroid_vox": centroid.astype(float),
        "bbox_min_vox": mins.astype(int),
        "bbox_max_vox": maxs.astype(int),
        "n_voxels": int(pts.shape[0]),
    }
