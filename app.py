import streamlit as st
import os
import numpy as np
import plotly.graph_objects as go

from skimage import measure
from skimage.measure import label
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

import io
import zipfile
from datetime import datetime

from backend import calculate_volume, find_model_paths, get_default_models_dir, load_model, segment_patient, tumor_stats

# === CONFIGURATION ===
MODEL_DIR = get_default_models_dir()
VOXEL_SIZE_MM = (1.0, 1.0, 1.0)  # 1 mm RAS in the backend

TRANSLATIONS = {
    "en": {
        "app_title": "Glioma 3D Anatomical Visualization",
        "safety_warning": "Research prototype for exploratory glioma visualization. Not clinically validated. The binary whole-tumor mask includes edema and does not define tissue to resect or preserve.",
        "cortical_warning": "The smoothed brain surface is an envelope for visual context only. It is not a precise cortical surgical reference. Functional areas, tracts, arteries and venous sinuses are not represented.",
        "patient_section": "Patient",
        "patient_path": "Patient folder:",
        "ensemble_section": "Ensemble model",
        "models_detected": "model(s) detected",
        "view_checkpoints": "View checkpoints",
        "brain_section": "FULL brain (HD-BET)",
        "use_hdbet": "Use HD-BET (recommended)",
        "brain_modality": "Modality for brain extraction",
        "hdbet_command": "HD-BET command",
        "hdbet_device": "HD-BET device",
        "cuts_section": "Virtual display cuts",
        "sagittal_cut": "Sagittal Cut (X)",
        "coronal_cut": "Coronal Cut (Y)",
        "axial_cut": "Axial Cut (Z)",
        "apply_cuts_tumor": "Apply cuts to tumor",
        "tissue_options": "Tissue Options",
        "show_brain": "Show Brain",
        "cavity_mode": "Cavity Mode",
        "show_tumor": "Show Tumor",
        "brain_opacity": "Brain Opacity",
        "validation_2d": "2D visual inspection",
        "show_2d": "Show 2D view",
        "axis": "Axis",
        "export_section": "Export Settings",
        "format": "Format",
        "include_brain": "Include Brain",
        "include_tumor": "Include Tumor",
        "postprocess": "Post-processing",
        "repair": "Repair (watertight/cleanup)",
        "advanced_options": "Advanced options (Trimesh):",
        "smooth_mesh": "Smooth Mesh (iters)",
        "spinner_analyzing": "Analyzing MRI...",
        "inference_error": "Error during inference/brain extraction:",
        "analysis_done": "Analysis completed",
        "brain_source_hdbet": "Brain: HD-BET",
        "brain_source_fallback": "Brain: simple fallback",
        "tumor_volume": "Whole-tumor label volume (includes edema)",
        "position_title": "Position and general info",
        "no_tumor": "No positive voxels in this prediction. This does not exclude a lesion.",
        "laterality": "Laterality",
        "position": "Position",
        "centroid": "Centroid",
        "left": "Left",
        "right": "Right",
        "posterior": "Posterior",
        "anterior": "Anterior",
        "inferior": "Inferior",
        "superior": "Superior",
        "brain_mesh": "Brain",
        "tumor_mesh": "Tumor",
        "empty_3d": "Enable 'Show Brain' or 'Show Tumor' to display the 3D model.",
        "view_2d": "2D View",
        "slice_index": "Slice Index",
        "generate_3d_files": "Generate 3D Files",
        "current_config": "Current settings",
        "generate_export": "Generate export file",
        "spinner_mesh": "Processing meshes (Padding + Blur + Repair)...",
        "empty_meshes": "Error: empty meshes.",
        "postprocess_ok": "Post-processing OK",
        "no_trimesh": "No trimesh available (exporting what is possible).",
        "download_zip": "Download ZIP (OBJ+MTL)",
        "download_3mf": "Download 3MF",
        "cannot_export_3mf": "Could not export 3MF",
        "install_trimesh": "install trimesh",
        "enter_patient": "Enter the patient folder.",
        "plot_overlay_title": "Blue=brain, Red=tumor",
        "trimesh_missing": "trimesh not installed",
        "cleanup_ok": "Cleanup OK",
        "cleanup_error": "Cleanup error",
        "mesh_smooth_failed": "Mesh smoothing failed",
        "repair_ok": "Repair OK",
        "repair_error": "Repair error",
        "brain_info": "Brain",
        "tumor_info": "Tumor",
    },
}


def t(key):
    return TRANSLATIONS["en"].get(key, key)


st.set_page_config(page_title=t("app_title"), layout="wide", page_icon="🧠")

st.markdown("""
    <style>
        .stApp {background-color: #0E1117;}
        h1, h2, h3 {color: #4A90E2;}
        .stSlider {padding-top: 20px;}
        .main-svg {border-radius: 10px; border: 1px solid #333;}
        .block-container {padding-top: 1.2rem;}
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_ai_model():
    return load_model(MODEL_DIR)

# =========================
# ROBUST 3D VISUALIZATION + ORIENTATION FIX
# =========================
def create_mesh_data(volume, level, color, opacity, name):
    """
    - Convert to float.
    - Add padding for marching_cubes.
    - Fix axes: skimage returns verts (z,y,x), plotly expects (x,y,z).
    """
    vol = volume.astype(np.float32)
    if np.count_nonzero(vol) == 0:
        return None

    vol_padded = np.pad(vol, pad_width=1, mode="constant", constant_values=0)
    try:
        verts_zyx, faces, _, _ = measure.marching_cubes(vol_padded, level=float(level))
    except ValueError:
        return None

    verts_zyx -= 1.0
    verts_xyz = verts_zyx[:, [2, 1, 0]]  # (x,y,z)

    return {
        "x": verts_xyz[:, 0],
        "y": verts_xyz[:, 1],
        "z": verts_xyz[:, 2],
        "i": faces[:, 0],
        "j": faces[:, 1],
        "k": faces[:, 2],
        "color": color,
        "opacity": opacity,
        "name": name,
    }

# Fixed front-facing camera.
FRONT_CAMERA = dict(
    up=dict(x=0, y=0, z=1),        # z points upward
    center=dict(x=0, y=0, z=0),
    eye=dict(x=0.0, y=2.6, z=0.25) # front-facing view toward the origin
)

# =========================
# CUT HELPERS
# =========================
def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    lab = label(mask.astype(bool))
    if lab.max() == 0:
        return mask.astype(bool)
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    largest = counts.argmax()
    return (lab == largest)

# =========================
# OPTIONAL 2D DEBUG VIEW
# =========================
def plot_slice_overlays(img, brain_mask, tumor_mask, axis="Z", idx=None):
    img = img.astype(np.float32)
    brain_mask = brain_mask.astype(bool)
    tumor_mask = tumor_mask.astype(bool)

    if axis == "X":
        if idx is None: idx = img.shape[0] // 2
        sl_img = img[idx, :, :]
        sl_brain = brain_mask[idx, :, :]
        sl_tumor = tumor_mask[idx, :, :]
    elif axis == "Y":
        if idx is None: idx = img.shape[1] // 2
        sl_img = img[:, idx, :]
        sl_brain = brain_mask[:, idx, :]
        sl_tumor = tumor_mask[:, idx, :]
    else:
        if idx is None: idx = img.shape[2] // 2
        sl_img = img[:, :, idx]
        sl_brain = brain_mask[:, :, idx]
        sl_tumor = tumor_mask[:, :, idx]

    v = sl_img[sl_img != 0]
    if v.size > 0:
        vmin, vmax = np.percentile(v, 1), np.percentile(v, 99)
    else:
        vmin, vmax = float(sl_img.min()), float(sl_img.max())

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(sl_img.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)

    overlay_brain = np.zeros((*sl_brain.T.shape, 4), dtype=np.float32)
    overlay_brain[..., 2] = 1.0
    overlay_brain[..., 3] = sl_brain.T.astype(np.float32) * 0.25
    ax.imshow(overlay_brain, origin="lower")

    overlay_t = np.zeros((*sl_tumor.T.shape, 4), dtype=np.float32)
    overlay_t[..., 0] = 1.0
    overlay_t[..., 3] = sl_tumor.T.astype(np.float32) * 0.35
    ax.imshow(overlay_t, origin="lower")

    ax.set_title(f"Slice {axis}={idx} | {t('plot_overlay_title')}")
    ax.axis("off")
    st.pyplot(fig)

# =========================
# EXPORT HELPERS (3MF / OBJ)
# =========================
def marching_cubes_mesh_xyz(volume: np.ndarray, level: float = 0.5, voxel_size_mm=(1.0, 1.0, 1.0)):
    vol_padded = np.pad(volume, pad_width=1, mode='constant', constant_values=0)
    vol = vol_padded.astype(np.float32)

    try:
        verts_zyx, faces, _, _ = measure.marching_cubes(vol, level=float(level))
    except ValueError:
        return None, None

    verts_zyx -= 1.0
    verts_xyz = verts_zyx[:, [2, 1, 0]].astype(np.float32)

    scale = np.array(voxel_size_mm, dtype=np.float32)
    verts_xyz_mm = verts_xyz * scale[None, :]
    return verts_xyz_mm, faces.astype(np.int32)

def try_repair_and_smooth_trimesh(verts, faces, smooth_iters=0, smooth_lambda=0.5, do_repair=True):
    try:
        import trimesh
        from trimesh import repair
    except Exception:
        return verts, faces, False, t("trimesh_missing")

    m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    info = []

    try:
        m.merge_vertices()
        m.remove_duplicate_faces()
        m.remove_degenerate_faces()
        info.append(t("cleanup_ok"))
    except Exception as e:
        info.append(f"{t('cleanup_error')}: {e}")

    if smooth_iters > 0:
        try:
            if hasattr(trimesh, "smoothing") and hasattr(trimesh.smoothing, "filter_laplacian"):
                trimesh.smoothing.filter_laplacian(m, lamb=float(smooth_lambda), iterations=int(smooth_iters))
                info.append(f"Smooth mesh: {smooth_iters} iters")
        except Exception as e:
            info.append(f"{t('mesh_smooth_failed')}: {e}")

    if do_repair:
        try:
            repair.fix_inversion(m)
            repair.fix_normals(m)
            m.fill_holes()
            info.append(t("repair_ok"))
        except Exception as e:
            info.append(f"{t('repair_error')}: {e}")

    verts2 = m.vertices.view(np.ndarray).astype(np.float32)
    faces2 = m.faces.view(np.ndarray).astype(np.int32)
    return verts2, faces2, True, "; ".join(info)

def export_obj_with_mtl(brain_verts, brain_faces, tumor_verts, tumor_faces):
    obj_name = "NeuroPlan_export.obj"
    mtl_name = "NeuroPlan_export.mtl"

    mtl = "\n".join([
        "newmtl brain_mat",
        "Ka 0.2 0.2 0.2",
        "Kd 0.7 0.7 0.7",
        "Ks 0.1 0.1 0.1",
        "d 0.25",
        "illum 2",
        "",
        "newmtl tumor_mat",
        "Ka 0.2 0.0 0.0",
        "Kd 1.0 0.0 0.0",
        "Ks 0.1 0.1 0.1",
        "d 1.0",
        "illum 2",
        ""
    ]).encode("utf-8")

    lines = [f"mtllib {mtl_name}"]

    v_offset_brain = 1
    v_offset_tumor = 1 + (0 if brain_verts is None else len(brain_verts))

    if brain_verts is not None:
        for v in brain_verts:
            lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")

    if tumor_verts is not None:
        for v in tumor_verts:
            lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")

    if brain_faces is not None and brain_verts is not None and len(brain_faces) > 0:
        lines.append("o Brain")
        lines.append("usemtl brain_mat")
        for f in brain_faces:
            a, b, c = f + v_offset_brain
            lines.append(f"f {a} {b} {c}")

    if tumor_faces is not None and tumor_verts is not None and len(tumor_faces) > 0:
        lines.append("o Tumor")
        lines.append("usemtl tumor_mat")
        for f in tumor_faces:
            a, b, c = f + v_offset_tumor
            lines.append(f"f {a} {b} {c}")

    obj = "\n".join(lines).encode("utf-8")
    return obj, mtl, obj_name, mtl_name

def export_3mf_two_objects(brain_verts, brain_faces, tumor_verts, tumor_faces):
    try:
        import trimesh
    except Exception:
        return None, None, t("trimesh_missing")

    scene = trimesh.Scene()
    if brain_verts is not None and brain_faces is not None:
        brain = trimesh.Trimesh(vertices=brain_verts, faces=brain_faces, process=False)
        scene.add_geometry(brain, node_name="Brain")

    if tumor_verts is not None and tumor_faces is not None:
        tumor = trimesh.Trimesh(vertices=tumor_verts, faces=tumor_faces, process=False)
        scene.add_geometry(tumor, node_name="Tumor")

    data = scene.export(file_type="3mf")
    if isinstance(data, str):
        data = data.encode("utf-8")
    return data, "NeuroPlan_export.3mf", "ok"

# =========================
# TITLE + SAFETY
# =========================
st.title(t("app_title"))
st.warning(f"⚠️ {t('safety_warning')}")
st.warning(t("cortical_warning"))

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header(f"🧑‍⚕️ {t('patient_section')}")
    patient_path = st.text_input(t("patient_path"), placeholder="C:/.../BraTS-GLI-00001-000")

    st.divider()
    st.subheader(t("ensemble_section"))
    try:
        model_files = find_model_paths(MODEL_DIR)
        st.success(f"{len(model_files)} {t('models_detected')}")
        with st.expander(t("view_checkpoints")):
            for model_file in model_files:
                st.caption(os.path.basename(model_file))
    except Exception as e:
        st.error(str(e))

    st.divider()
    st.subheader(f"🧠 {t('brain_section')}")

    use_external_brain = st.checkbox(t("use_hdbet"), value=True)

    brain_input_modality = st.selectbox(
        t("brain_modality"),
        ["t1n", "t2f", "t2w", "t1c"],
        index=0,
        disabled=not use_external_brain
    )

    hdbet_exe = st.text_input(
        t("hdbet_command"),
        value="hd-bet",
        disabled=not use_external_brain
    )

    hdbet_device = st.selectbox(
        t("hdbet_device"),
        ["cpu", "cuda"],
        index=0,
        disabled=not use_external_brain
    )

    st.divider()
    st.subheader(f"🪚 {t('cuts_section')}")

    if 'img_orig' in st.session_state:
        shape = st.session_state['img_orig'].shape
        x_range = st.slider(t("sagittal_cut"), 0, shape[0], (0, shape[0]))
        y_range = st.slider(t("coronal_cut"), 0, shape[1], (0, shape[1]))
        z_range = st.slider(t("axial_cut"), 0, shape[2], (0, shape[2]))
        cut_tumor = st.toggle(t("apply_cuts_tumor"), value=True)
    else:
        x_range = y_range = z_range = (0, 1)
        cut_tumor = True

    st.divider()
    st.subheader(f"👁️ {t('tissue_options')}")

    # Start with tissue layers turned off.
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        show_brain = st.checkbox(t("show_brain"), value=False)
        subtract_mode = st.checkbox(t("cavity_mode"), value=False)
    with col_opt2:
        show_tumor = st.checkbox(t("show_tumor"), value=False)

    brain_opacity = st.slider(t("brain_opacity"), 0.1, 1.0, 1.0)

    st.divider()
    st.subheader(f"🩻 {t('validation_2d')}")
    show_debug_2d = st.checkbox(t("show_2d"), value=False)
    debug_axis = st.selectbox(t("axis"), ["Z", "X", "Y"], index=0)

    st.divider()
    st.subheader(f"📦 {t('export_section')}")

    export_format = st.selectbox(t("format"), ["OBJ+MTL (ZIP) ✅", "3MF (2 objects/objetos) ⭐"], index=1)
    export_brain = st.checkbox(t("include_brain"), value=True)
    export_tumor = st.checkbox(t("include_tumor"), value=True)

    st.markdown(f"**{t('postprocess')}**")
    sigma_blur = st.slider("Gaussian Blur", 0.0, 3.0, 1.0, 0.5)
    do_repair = st.checkbox(t("repair"), value=True)
    st.caption(t("advanced_options"))
    smooth_iters = st.slider(t("smooth_mesh"), 0, 20, 0, 1)
    smooth_lambda = st.slider("Smooth Lambda", 0.05, 0.95, 0.35, 0.05)

# =========================
# INFERENCE (loads once per patient)
# =========================
if patient_path and os.path.exists(patient_path):
    if 'current_patient' not in st.session_state or st.session_state['current_patient'] != patient_path:
        with st.spinner(f"🔍 {t('spinner_analyzing')}"):
            model = get_ai_model()
            try:
                if use_external_brain:
                    out = segment_patient(
                        model, patient_path,
                        brain_tool="hdbet",
                        brain_input_modality=brain_input_modality,
                        hdbet_exe=hdbet_exe,
                        hdbet_device=hdbet_device,
                    )
                else:
                    out = segment_patient(model, patient_path, brain_tool="none")

                if len(out) == 3:
                    img, seg, brain_ext = out
                else:
                    img, seg = out
                    brain_ext = None

            except Exception as e:
                st.error(f"{t('inference_error')}\n{e}")
                st.stop()

            st.session_state['img_orig'] = img
            st.session_state['seg_orig'] = seg
            st.session_state['brain_ext'] = brain_ext
            st.session_state['current_patient'] = patient_path
            st.success(f"✅ {t('analysis_done')}")

# =========================
# MAIN
# =========================
if 'img_orig' in st.session_state:
    img = st.session_state['img_orig']
    seg = st.session_state['seg_orig']
    brain_ext = st.session_state.get('brain_ext', None)

    # Cuts
    mask_crop = np.zeros_like(img, dtype=np.float32)
    mask_crop[x_range[0]:x_range[1], y_range[0]:y_range[1], z_range[0]:z_range[1]] = 1.0

    img_final = img * mask_crop
    seg_final = seg * mask_crop if cut_tumor else seg

    # Brain (HD-BET)
    if brain_ext is not None:
        brain_mask = (brain_ext > 0).astype(bool)
        brain_mask = np.logical_and(brain_mask, mask_crop > 0)
        brain_source = f"{t('brain_source_hdbet')} ✅"
    else:
        # If HD-BET is disabled, show a simple fallback mask.
        brain_mask = (img_final != 0)
        brain_mask = keep_largest_component(brain_mask)
        brain_source = f"{t('brain_source_fallback')} ⚠️"

    brain_mask_final = brain_mask.copy()
    if subtract_mode:
        brain_mask_final = np.logical_and(brain_mask_final, np.logical_not(seg_final > 0))

    vol_cm3 = calculate_volume(seg, voxel_size=VOXEL_SIZE_MM)
    stats = tumor_stats(seg > 0)

    def pos_labels(centroid_vox, shape):
        midx = (shape[0] - 1) / 2.0
        midy = (shape[1] - 1) / 2.0
        midz = (shape[2] - 1) / 2.0
        lr = t("left") if centroid_vox[0] < midx else t("right")
        ap = t("posterior") if centroid_vox[1] < midy else t("anterior")
        si = t("inferior") if centroid_vox[2] < midz else t("superior")
        return lr, ap, si

    col_info, col_3d = st.columns([1, 4])

    with col_info:
        st.metric(t("tumor_volume"), f"{vol_cm3:.2f} cm³")
        st.markdown("---")
        st.caption(brain_source)

        st.markdown(f"### 📍 {t('position_title')}")
        if stats is None:
            st.warning(t("no_tumor"))
        else:
            c = stats["centroid_vox"]
            mins = stats["bbox_min_vox"]
            maxs = stats["bbox_max_vox"]
            lr, ap, si = pos_labels(c, img.shape)

            st.write(f"**{t('laterality')}:** {lr}")
            st.write(f"**{t('position')}:** {ap}, {si}")
            st.write(f"**{t('centroid')}:** x={c[0]:.1f}, y={c[1]:.1f}, z={c[2]:.1f}")
            st.write(f"**BBox:** {mins.tolist()} - {maxs.tolist()}")

    with col_3d:
        fig = go.Figure()
        added_any = False

        if show_brain:
            mesh = create_mesh_data(brain_mask_final, 0.5, 'gray', brain_opacity, t("brain_mesh"))
            if mesh is not None:
                fig.add_trace(go.Mesh3d(
                    x=mesh['x'], y=mesh['y'], z=mesh['z'],
                    i=mesh['i'], j=mesh['j'], k=mesh['k'],
                    color=mesh['color'], opacity=mesh['opacity'],
                    name=mesh['name'], showscale=False,
                    lighting=dict(ambient=0.3, diffuse=0.6, roughness=0.1, specular=0.4)
                ))
                added_any = True

        if show_tumor and np.sum(seg_final) > 0:
            mesh_t = create_mesh_data(seg_final, 0.5, '#FF3333', 1.0, t("tumor_mesh"))
            if mesh_t is not None:
                fig.add_trace(go.Mesh3d(
                    x=mesh_t['x'], y=mesh_t['y'], z=mesh_t['z'],
                    i=mesh_t['i'], j=mesh_t['j'], k=mesh_t['k'],
                    color=mesh_t['color'], opacity=1.0,
                    name=mesh_t['name'], showscale=False
                ))
                added_any = True

        if not added_any:
            st.info(t("empty_3d"))

        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode='data',
                bgcolor='#0E1117',
                camera=FRONT_CAMERA,   # front-facing view
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            height=700
        )
        st.plotly_chart(fig, use_container_width=True)

    if show_debug_2d:
        st.markdown(f"## 🩻 {t('view_2d')}")
        idx = st.slider(t("slice_index"), 0, img.shape[{"X":0,"Y":1,"Z":2}[debug_axis]]-1)
        plot_slice_overlays(img_final, brain_mask, (seg_final > 0), axis=debug_axis, idx=idx)

    # =========================
    # EXPORT (OBJ+MTL / 3MF)
    # =========================
    st.markdown(f"## 📦 {t('generate_3d_files')}")
    st.info(f"{t('current_config')}: Gaussian Blur={sigma_blur} | {t('format')}={export_format}")

    if st.button(t("generate_export")):
        with st.spinner(t("spinner_mesh")):

            def get_smooth_mesh(mask_binary, sigma):
                if np.count_nonzero(mask_binary) == 0:
                    return None, None
                vol_float = mask_binary.astype(np.float32)
                if sigma > 0:
                    vol_float = gaussian_filter(vol_float, sigma=sigma)
                return marching_cubes_mesh_xyz(vol_float, level=0.5, voxel_size_mm=VOXEL_SIZE_MM)

            b_verts = b_faces = None
            t_verts = t_faces = None

            if export_brain:
                b_verts, b_faces = get_smooth_mesh(brain_mask_final, sigma_blur)
            if export_tumor:
                t_verts, t_faces = get_smooth_mesh(seg_final, sigma_blur)

            if b_verts is None and t_verts is None:
                st.error(t("empty_meshes"))
            else:
                used_trimesh = False
                info_msgs = []

                if b_verts is not None:
                    b_verts, b_faces, ok, info = try_repair_and_smooth_trimesh(
                        b_verts, b_faces, smooth_iters=smooth_iters, smooth_lambda=smooth_lambda, do_repair=do_repair
                    )
                    used_trimesh |= ok
                    info_msgs.append(f"{t('brain_info')}: {info}")

                if t_verts is not None:
                    t_verts, t_faces, ok, info = try_repair_and_smooth_trimesh(
                        t_verts, t_faces, smooth_iters=max(0, smooth_iters//2), smooth_lambda=smooth_lambda, do_repair=do_repair
                    )
                    used_trimesh |= ok
                    info_msgs.append(f"{t('tumor_info')}: {info}")

                if used_trimesh:
                    st.success(t("postprocess_ok"))
                else:
                    st.warning(t("no_trimesh"))

                if info_msgs:
                    st.caption(" | ".join(info_msgs))

                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                if export_format.startswith("OBJ"):
                    o, m, on, mn = export_obj_with_mtl(b_verts, b_faces, t_verts, t_faces)
                    buf = io.BytesIO()
                    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                        zf.writestr(on, o)
                        zf.writestr(mn, m)
                    st.download_button(
                        f"⬇️ {t('download_zip')}",
                        buf.getvalue(),
                        f"Export_{ts}.zip",
                        "application/zip"
                    )
                else:
                    d3mf, n3mf, msg = export_3mf_two_objects(b_verts, b_faces, t_verts, t_faces)
                    if d3mf:
                        st.download_button(
                            f"⬇️ {t('download_3mf')}",
                            d3mf,
                            f"Export_{ts}.3mf",
                            "model/3mf"
                        )
                    else:
                        st.error(f"{t('cannot_export_3mf')}: {msg} ({t('install_trimesh')}).")

else:
    st.info(f"👈 {t('enter_patient')}")
