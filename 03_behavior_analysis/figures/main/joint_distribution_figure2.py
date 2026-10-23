import os
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib.colors import to_rgba
from scipy.stats import chi2, f as f_dist
import matplotlib.patheffects as pe

# =====================================
# 1. Data loading
# =====================================
from pathlib import Path
repo_root = Path(__file__).resolve().parents[3]
data_root = Path(os.environ.get('PERSONA_DATA_ROOT', repo_root / 'data' / 'raw'))
path_base = str(data_root / 'llm_output')
human_path = str(data_root / 'data_long.txt')
output_dir = str(Path(os.environ.get('PERSONA_FIGURE_OUTPUT', repo_root / 'outputs' / 'figures')))
os.makedirs(output_dir, exist_ok=True)

DATASETS_RAW = {
    'GPT-4o-mini': 'simulation_output_flattened-gpt4omini.csv',
    'GPT-5.2': 'simulation_output_flattened-gpt52.csv',
    'Qwen-2.5-7B': 'simulation_output_flattened-qwen2.5-7b-instruct.csv',
    'Qwen-2.5-14B': 'simulation_output_flattened-qwen2.5-14b-instruct.csv',
    'Qwen-2.5-32B': 'simulation_output_flattened-qwen2.5-32b-instruct.csv',
    'Deepseek-3.2': 'simulation_output_flattened-deepseek3.2.csv',
    'Grok-3': 'simulation_output_flattened-grok3.csv',
    'Llama-3.1-8B': 'simulation_output_flattened_llama3.1-8b-instruct.csv',
    'Phi-mini': 'simulation_output_flattened-phi3-mini.csv',
    'Phi-small': 'simulation_output_flattened-phi3-small.csv',
    'Phi-medium': 'simulation_output_flattened-phi3-medium.csv'
}

SMALL_PLOT_NAMES = [
    'Human', 'GPT-4o-mini', 'GPT-5.2', 'Deepseek-3.2', 'Grok-3',
    'Llama-3.1-8B', 'Qwen-2.5-7B', 'Qwen-2.5-14B',
    'Qwen-2.5-32B', 'Phi-mini', 'Phi-small', 'Phi-medium'
]

SHORT_TRAITS = ['E', 'A', 'C', 'N', 'O']

# =====================================
# Output and memory-safe settings
# =====================================
FIG_WIDTH = 7.2
FIG_HEIGHT = 7.7

PNG_DPI = 450
PDF_DPI = 300

SMALL_PANEL_MAX_POINTS = 9000
MAIN_PANEL_MAX_POINTS = 18000
MAIN_KDE_MAX_POINTS = 6500

SAVE_PNG = True
SAVE_PDF = True

rng = np.random.default_rng(42)


def safe_read_csv(path):
    if not os.path.exists(path):
        print(f'File not found: {path}')
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding='gbk')


def sample_points(Z, max_points, rng):
    if Z.shape[0] <= max_points:
        return Z
    idx = rng.choice(Z.shape[0], max_points, replace=False)
    return Z[idx]


dfs_delta = {}
for name, fname in DATASETS_RAW.items():
    df = safe_read_csv(os.path.join(path_base, fname))
    if df.empty:
        continue

    missing_cols = []
    for t in SHORT_TRAITS:
        if f'State_{t}' not in df.columns:
            missing_cols.append(f'State_{t}')

    if missing_cols:
        print(f'Skip {name}: missing columns {missing_cols}')
        continue

    dfs_delta[name] = df[[f'State_{t}' for t in SHORT_TRAITS]].dropna().astype(float)

human = pd.read_csv(human_path, sep=r'\s+')
human = human.rename(columns={f'State_{t}_sum': f'State_{t}' for t in SHORT_TRAITS})
dfs_delta['Human'] = human[[f'State_{t}' for t in SHORT_TRAITS]].apply(
    pd.to_numeric, errors='coerce'
).dropna().astype(float)

if 'Human' not in dfs_delta:
    raise ValueError('Human dataset was not loaded. Please check path_base and filenames.')

# Human-based scaling
scaler = StandardScaler().fit(dfs_delta['Human'].to_numpy())
dfs_scaled = {
    name: pd.DataFrame(scaler.transform(data.to_numpy()), columns=SHORT_TRAITS)
    for name, data in dfs_delta.items()
}

# =====================================
# 2. Colors
# =====================================
llm_colors = {
    'Human': '#1B1B1B',
    'GPT-4o-mini': '#E68613',
    'GPT-5.2': '#D65F9E',
    'Deepseek-3.2': '#E64B35',
    'Grok-3': '#00A087',
    'Llama-3.1-8B': '#A6761D',
    'Qwen-2.5-7B': '#168ACB',
    'Qwen-2.5-14B': '#0868AC',
    'Qwen-2.5-32B': '#084081',
    'Phi-mini': '#F2B94B',
    'Phi-small': '#D7814A',
    'Phi-medium': '#9A5C73'
}

MODEL_DISPLAY_NAMES = {
    'Human': 'Human',
    'GPT-4o-mini': 'GPT-4o-mini',
    'GPT-5.2': 'GPT-5.2',
    'Deepseek-3.2': 'DeepSeek-3.2',
    'Grok-3': 'Grok-3',
    'Llama-3.1-8B': 'Llama-3.1-8B',
    'Qwen-2.5-7B': 'Qwen2.5-7B',
    'Qwen-2.5-14B': 'Qwen2.5-14B',
    'Qwen-2.5-32B': 'Qwen2.5-32B',
    'Phi-mini': 'Phi-3 mini',
    'Phi-small': 'Phi-3 small',
    'Phi-medium': 'Phi-3-medium'
}

SMALL_PANEL_FACE = '#F8F8F8'
SMALL_PANEL_EDGE = '#D9D9D9'

# 深色 Big Five 颜色
trait_colors = {
    'E': '#B67A00',
    'A': '#A95F00',
    'C': '#00876C',
    'N': '#A64D79',
    'O': '#0072B2'
}

# =====================================
# 3. Typography
# =====================================
BASE_FONT_SIZE = 7.0
SMALL_TICK_SIZE = 5.5
SMALL_TITLE_SIZE = 7.0
AXIS_LABEL_SIZE = 7.0
BAR_LABEL_SIZE = 6.0
BAR_VALUE_SIZE = 6.0
LEGEND_TEXT_SIZE = 6.0
LEGEND_TITLE_SIZE = 6.5
CENTROID_LABEL_SIZE = 6.0
PANEL_LABEL_SIZE = 8.0

FONT_FALLBACKS = ['Arial', 'Helvetica', 'DejaVu Sans']

_FONT_NAME_MAP = {
    font.name.replace(' ', '').replace('-', '').lower(): font.name
    for font in font_manager.fontManager.ttflist
}


def resolve_font_name(requested_name, fallback='Arial'):
    normalized = requested_name.replace(' ', '').replace('-', '').lower()
    if normalized in _FONT_NAME_MAP:
        return _FONT_NAME_MAP[normalized]

    family_match = next(
        (
            name for key, name in _FONT_NAME_MAP.items()
            if 'graphiknaturel' in key
        ),
        None
    )
    return family_match or fallback


FONT_REGULAR = resolve_font_name('GraphikNaturel-Regular')
FONT_SEMIBOLD = resolve_font_name('GraphikNaturel-Semibold')
FONT_BLACK = resolve_font_name('GraphikNaturel-Black')
FONT_ITALIC = resolve_font_name('GraphikNaturel-RegularItalic')
GRAPHIK_AVAILABLE = any(
    'graphiknaturel' in key for key in _FONT_NAME_MAP
)
PANEL_FONT_WEIGHT = 'black' if GRAPHIK_AVAILABLE else 'bold'

plt.rcParams.update({
    'font.family': [FONT_REGULAR, *FONT_FALLBACKS],
    'font.sans-serif': [FONT_REGULAR, *FONT_FALLBACKS],
    'font.size': BASE_FONT_SIZE,
    'axes.labelsize': AXIS_LABEL_SIZE,
    'axes.titlesize': SMALL_TITLE_SIZE,
    'xtick.labelsize': SMALL_TICK_SIZE,
    'ytick.labelsize': SMALL_TICK_SIZE,
    'legend.fontsize': LEGEND_TEXT_SIZE,
    'legend.title_fontsize': LEGEND_TITLE_SIZE,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

# =====================================
# 4. Helper functions
# =====================================
def get_cov_ellipse_params(x, y, level=0.95):
    data = np.column_stack([x, y])
    if data.shape[0] < 3:
        return None

    cov = np.cov(data, rowvar=False)
    mean = data.mean(axis=0)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = np.maximum(vals[order], 1e-12)
    vecs = vecs[:, order]

    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    scale = np.sqrt(chi2.ppf(level, df=2))
    major_radius = scale * np.sqrt(vals[0])
    minor_radius = scale * np.sqrt(vals[1])

    return {
        'mean': mean,
        'width': 2 * major_radius,
        'height': 2 * minor_radius,
        'angle': theta,
        'major_radius': major_radius,
        'minor_radius': minor_radius,
    }


def add_cov_ellipse(ax, x, y, color, level=0.95, lw=2.4, alpha=0.98, zorder=20):
    params = get_cov_ellipse_params(x, y, level=level)
    if params is None:
        return None

    ellipse = Ellipse(
        xy=params['mean'],
        width=params['width'],
        height=params['height'],
        angle=params['angle'],
        facecolor='none',
        edgecolor=color,
        linewidth=lw,
        alpha=alpha,
        zorder=zorder
    )
    ax.add_patch(ellipse)
    return ellipse


def ellipse_bounds(x, y, level=0.95):
    data = np.column_stack([x, y])
    if data.shape[0] < 3:
        return np.min(x), np.max(x), np.min(y), np.max(y)

    cov = np.cov(data, rowvar=False)
    mean = data.mean(axis=0)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-12)
    scale = np.sqrt(chi2.ppf(level, df=2))

    theta = np.linspace(0, 2 * np.pi, 400)
    circle = np.vstack([np.cos(theta), np.sin(theta)])
    ellipse = vecs @ np.diag(scale * np.sqrt(vals)) @ circle
    ellipse[0, :] += mean[0]
    ellipse[1, :] += mean[1]

    return ellipse[0, :].min(), ellipse[0, :].max(), ellipse[1, :].min(), ellipse[1, :].max()


def style_axis(ax, limit=4, show_ticks=True, xlim=None, ylim=None, axis_lw=0.7,
               equal_data=False, box_aspect=None):
    ax.axhline(0, color='#444444', lw=axis_lw, zorder=0)
    ax.axvline(0, color='#444444', lw=axis_lw, zorder=0)

    if xlim is None:
        xlim = (-limit, limit)
    if ylim is None:
        ylim = (-limit, limit)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if equal_data:
        ax.set_aspect('equal', adjustable='box')
    if box_aspect is not None:
        ax.set_box_aspect(box_aspect)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.grid(False)

    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.tick_params(length=2.2, width=0.55, colors='#333333')


def add_trait_arrows(
    ax,
    loadings,
    scale=2.6,
    lw=2.0,
    zorder=45,
    label=False,
    label_size=9.0,
    label_offset=0.10,
    outline_arrow=True,
    outline_label=True,
    mutation_scale=10.5,
    text_zorder=250,
    label_traits=None,
    label_perp_offsets=None,
    arrow_color=None,
    label_color=None,
    alpha=1.0,
    extend_to_boundary=False,
    boundary_limit=4.0,
    boundary_pad=0.24,
    guide_lw=0.45,
    guide_alpha=0.42
):
    for t in SHORT_TRAITS:
        dx = loadings.loc[t, 'PC1'] * scale
        dy = loadings.loc[t, 'PC2'] * scale
        norm = max((dx ** 2 + dy ** 2) ** 0.5, 1e-6)
        ux, uy = dx / norm, dy / norm
        perpendicular_offset = (
            label_perp_offsets.get(t, 0.0)
            if label_perp_offsets is not None
            else 0.0
        )

        if extend_to_boundary:
            usable_limit = boundary_limit - boundary_pad
            tx_scale = usable_limit / abs(ux) if abs(ux) > 1e-9 else np.inf
            ty_scale = usable_limit / abs(uy) if abs(uy) > 1e-9 else np.inf
            boundary_scale = min(tx_scale, ty_scale)
            guide_x = ux * boundary_scale
            guide_y = uy * boundary_scale

            ax.plot(
                [0, guide_x],
                [0, guide_y],
                color=arrow_color or trait_colors[t],
                linewidth=guide_lw,
                linestyle=(0, (2.2, 2.2)),
                alpha=guide_alpha,
                zorder=zorder - 0.5
            )

        ann = ax.annotate(
            '',
            xy=(dx, dy),
            xytext=(0, 0),
            arrowprops=dict(
                arrowstyle='-|>',
                lw=lw,
                color=arrow_color or trait_colors[t],
                alpha=alpha,
                shrinkA=0,
                shrinkB=0,
                mutation_scale=mutation_scale
            ),
            zorder=zorder
        )

        # 小图调用时 outline_arrow=False，因此不会出现白色底
        if outline_arrow and ann.arrow_patch is not None:
            ann.arrow_patch.set_path_effects([
                pe.Stroke(
                    linewidth=lw + 1.8,
                    foreground='white',
                    alpha=0.70 * alpha
                ),
                pe.Normal()
            ])

        show_label = label and (label_traits is None or t in label_traits)
        if show_label:
            if extend_to_boundary:
                tx = guide_x - label_offset * ux - perpendicular_offset * uy
                ty = guide_y - label_offset * uy + perpendicular_offset * ux
            else:
                tx = dx + label_offset * ux - perpendicular_offset * uy
                ty = dy + label_offset * uy + perpendicular_offset * ux

            txt = ax.text(
                tx,
                ty,
                t,
                fontsize=label_size,
                fontfamily=[FONT_REGULAR, *FONT_FALLBACKS],
                fontweight='normal',
                color=label_color or arrow_color or trait_colors[t],
                alpha=alpha,
                ha='center',
                va='center',
                zorder=text_zorder
            )

            if outline_label:
                txt.set_path_effects([
                    pe.Stroke(
                        linewidth=2.4,
                        foreground='white',
                        alpha=0.72 * alpha
                    ),
                    pe.Normal()
                ])


def choose_initial_offset(point, center=(0, 0), base=0.22):
    x, y = point
    cx, cy = center
    vx, vy = x - cx, y - cy
    norm = max((vx ** 2 + vy ** 2) ** 0.5, 1e-6)
    ux, uy = vx / norm, vy / norm

    dx = base * np.sign(ux if abs(ux) > 0.12 else (1 if x >= cx else -1))
    dy = base * np.sign(uy if abs(uy) > 0.12 else (1 if y >= cy else -1))

    return dx, dy


def add_centroid_label(ax, name, xy, color, offset=(0.20, 0.20)):
    x, y = xy
    dx, dy = offset
    display_name = MODEL_DISPLAY_NAMES.get(name, name)

    ann = ax.annotate(
        display_name,
        xy=(x, y),
        xytext=(x + dx, y + dy),
        textcoords='data',
        fontsize=CENTROID_LABEL_SIZE,
        color='#202020',
        ha='left' if dx >= 0 else 'right',
        va='bottom' if dy >= 0 else 'top',
        weight='bold' if name == 'Human' else 'normal',
        bbox=dict(
            boxstyle='round,pad=0.15',
            facecolor='white',
            edgecolor=color,
            linewidth=0.75,
            alpha=0.92
        ),
        arrowprops=dict(
            arrowstyle='-',
            color=color,
            lw=2.00,
            alpha=0.92,
            shrinkA=2,
            shrinkB=4
        ),
        zorder=260,
        annotation_clip=True,
    )

    # 让模型文字与其引线都处于最上层
    ann.set_zorder(260)
    if ann.arrow_patch is not None:
        ann.arrow_patch.set_zorder(255)

    return ann


def _bboxes_overlap(b1, b2, pad=1.2):
    return not (
        b1.x1 + pad < b2.x0 or b2.x1 + pad < b1.x0 or
        b1.y1 + pad < b2.y0 or b2.y1 + pad < b1.y0
    )


def repel_annotations(ax, annotations, expand_step=2.7, max_iter=90):
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    for _ in range(max_iter):
        moved = False
        bboxes = [
            ann.get_window_extent(renderer=renderer).expanded(1.02, 1.08)
            for ann in annotations
        ]
        centers = [
            ((b.x0 + b.x1) / 2.0, (b.y0 + b.y1) / 2.0)
            for b in bboxes
        ]
        shifts = [np.array([0.0, 0.0]) for _ in annotations]

        for i in range(len(annotations)):
            for j in range(i + 1, len(annotations)):
                if _bboxes_overlap(bboxes[i], bboxes[j], pad=1.0):
                    moved = True
                    xi, yi = centers[i]
                    xj, yj = centers[j]
                    vx, vy = xi - xj, yi - yj

                    if abs(vx) < 1e-3 and abs(vy) < 1e-3:
                        vx, vy = np.random.uniform(-1, 1), np.random.uniform(-1, 1)

                    norm = max((vx ** 2 + vy ** 2) ** 0.5, 1e-6)
                    push = np.array([vx / norm, vy / norm]) * expand_step
                    shifts[i] += push
                    shifts[j] -= push

        if not moved:
            break

        for ann, shift in zip(annotations, shifts):
            cur_xytext = ann.get_position()
            cur_disp = ax.transData.transform(cur_xytext)
            new_disp = cur_disp + shift
            new_data = ax.transData.inverted().transform(new_disp)
            ann.set_position(new_data)

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()


def clamp_annotations_to_axes(ax, annotations, pad_frac_x=0.048, pad_frac_y=0.055):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xr = x1 - x0
    yr = y1 - y0

    xmin = x0 + pad_frac_x * xr
    xmax = x1 - pad_frac_x * xr
    ymin = y0 + pad_frac_y * yr
    ymax = y1 - pad_frac_y * yr

    for ann in annotations:
        tx, ty = ann.get_position()
        tx = min(max(tx, xmin), xmax)
        ty = min(max(ty, ymin), ymax)
        ann.set_position((tx, ty))

        # clamp 后再次确认文字在最上层
        ann.set_zorder(260)
        if ann.arrow_patch is not None:
            ann.arrow_patch.set_zorder(255)


def add_panel_label(ax, label, x=-0.02, y=1.02):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=PANEL_LABEL_SIZE,
        fontfamily=[FONT_BLACK, FONT_SEMIBOLD, *FONT_FALLBACKS],
        fontweight=PANEL_FONT_WEIGHT,
        fontstyle='normal',
        color='#111111',
        ha='left',
        va='bottom',
        clip_on=False,
        zorder=400
    )


def decorate_small_panel(ax):
    ax.patch.set_edgecolor(SMALL_PANEL_EDGE)
    ax.patch.set_linewidth(0.55)


def draw_placeholder_panel(ax, name):
    ax.set_facecolor(SMALL_PANEL_FACE)

    if not name:
        ax.axis('off')
        return

    style_axis(
        ax,
        limit=4,
        show_ticks=True,
        axis_lw=0.45,
        equal_data=True,
        box_aspect=1
    )
    ax.set_xticks([-4, -2, 0, 2, 4])
    ax.set_yticks([-4, -2, 0, 2, 4])
    decorate_small_panel(ax)

    ax.text(
        0.5, 0.5, 'Reserved',
        transform=ax.transAxes,
        ha='center',
        va='center',
        fontsize=6.0,
        fontfamily=[FONT_ITALIC, FONT_REGULAR, *FONT_FALLBACKS],
        color='#777777',
        style='italic'
    )

    ax.set_title(
        MODEL_DISPLAY_NAMES.get(name, name),
        fontsize=SMALL_TITLE_SIZE,
        fontfamily=[FONT_REGULAR, *FONT_FALLBACKS],
        fontweight='normal',
        pad=3.2,
        color='#1A1A1A'
    )


def draw_small_panel(ax, name):
    if name not in dfs_scaled:
        draw_placeholder_panel(ax, name)
        return

    data_np = dfs_scaled[name].to_numpy()

    pca = PCA(n_components=2)
    Z = pca.fit_transform(data_np)
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=['PC1', 'PC2'],
        index=SHORT_TRAITS
    )

    n_angle = np.arctan2(
        loadings.loc['N', 'PC2'],
        loadings.loc['N', 'PC1']
    )
    rotation_angle = -n_angle
    cos_a = np.cos(rotation_angle)
    sin_a = np.sin(rotation_angle)
    rotation = np.array([
        [cos_a, sin_a],
        [-sin_a, cos_a]
    ])
    Z = Z @ rotation
    loadings[['PC1', 'PC2']] = loadings[['PC1', 'PC2']].to_numpy() @ rotation

    Z_draw = sample_points(Z, SMALL_PANEL_MAX_POINTS, rng)

    ax.set_facecolor(SMALL_PANEL_FACE)

    ax.scatter(
        Z_draw[:, 0],
        Z_draw[:, 1],
        s=0.65,
        alpha=0.10,
        c=llm_colors[name],
        edgecolors='none',
        rasterized=True,
        zorder=1
    )

    kde_n = min(Z_draw.shape[0], 4500)
    kde_idx = rng.choice(Z_draw.shape[0], kde_n, replace=False)
    Z_kde = Z_draw[kde_idx]

    sns.kdeplot(
        x=Z_kde[:, 0],
        y=Z_kde[:, 1],
        ax=ax,
        fill=True,
        levels=8,
        thresh=0.08,
        color=llm_colors[name],
        alpha=0.26,
        zorder=2
    )

    add_trait_arrows(
        ax,
        loadings,
        scale=2.42,
        lw=1.00,
        zorder=3,
        label=True,
        label_traits=SHORT_TRAITS,
        label_size=5.0,
        label_offset=0.18,
        label_perp_offsets={
            'E': -0.10,
            'A': 0.10,
            'C': -0.10,
            'N': 0.00,
            'O': 0.10,
        },
        outline_arrow=False,
        outline_label=True,
        mutation_scale=7.0,
        text_zorder=4,
        alpha=0.72,
        extend_to_boundary=True,
        boundary_limit=4.0,
        boundary_pad=0.28,
        guide_lw=0.42,
        guide_alpha=0.38
    )

    style_axis(
        ax,
        limit=4,
        show_ticks=True,
        axis_lw=0.35,
        equal_data=True,
        box_aspect=1
    )
    ax.set_xticks([-4, -2, 0, 2, 4])
    ax.set_yticks([-4, -2, 0, 2, 4])
    ax.tick_params(length=1.8, width=0.45, pad=1.5, labelsize=SMALL_TICK_SIZE, colors='#333333')
    decorate_small_panel(ax)

    ax.set_title(
        MODEL_DISPLAY_NAMES.get(name, name),
        fontsize=SMALL_TITLE_SIZE,
        fontfamily=[FONT_REGULAR, *FONT_FALLBACKS],
        fontweight='normal',
        pad=3.2,
        color='#1A1A1A'
    )


# =====================================
# 5. Layout
# =====================================
fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), facecolor='white')
fig.text(
    0.025,
    0.985,
    'a',
    fontsize=PANEL_LABEL_SIZE,
    fontfamily=[FONT_BLACK, FONT_SEMIBOLD, *FONT_FALLBACKS],
    fontweight=PANEL_FONT_WEIGHT,
    fontstyle='normal',
    color='#111111',
    ha='left',
    va='top'
)

outer = fig.add_gridspec(
    2, 1,
    height_ratios=[1.62, 3.25],
    hspace=0.10
)

gs_top = outer[0].subgridspec(
    2, 6,
    wspace=0.16,
    hspace=0.20
)

gs_bottom = outer[1].subgridspec(
    1, 2,
    width_ratios=[2.30, 1.18],
    wspace=0.18
)

ax_main = fig.add_subplot(gs_bottom[0, 0])

gs_right = gs_bottom[0, 1].subgridspec(
    2, 1,
    height_ratios=[1.30, 1.00],
    hspace=0.16
)
ax_bar_dist = fig.add_subplot(gs_right[0, 0])
ax_bar_radius = fig.add_subplot(gs_right[1, 0])


# =====================================
# 6. Small panels
# =====================================
for i in range(len(SMALL_PLOT_NAMES)):
    row = i // 6
    col = i % 6
    ax = fig.add_subplot(gs_top[row, col])
    draw_small_panel(ax, SMALL_PLOT_NAMES[i])


# =====================================
# 7. Main unified PCA plot
# =====================================
ax_main.set_facecolor('#FBFBFB')

pca_h = PCA(n_components=2).fit(dfs_scaled['Human'].to_numpy())
loadings_h = pd.DataFrame(
    pca_h.components_.T,
    columns=['PC1', 'PC2'],
    index=SHORT_TRAITS
)

flip = -1 if loadings_h.loc['N', 'PC1'] < 0 else 1
if flip == -1:
    loadings_h['PC1'] = -loadings_h['PC1']

centroids = {}
main_projected = {}
ellipse_ranges = []

for name, data in dfs_scaled.items():
    Z = pca_h.transform(data.to_numpy())
    Z[:, 0] *= flip
    main_projected[name] = Z

    cx, cy = np.mean(Z[:, 0]), np.mean(Z[:, 1])
    centroids[name] = np.array([cx, cy])
    ellipse_ranges.append(ellipse_bounds(Z[:, 0], Z[:, 1], level=0.95))

for name, Z in main_projected.items():
    Z = main_projected[name]
    Z_draw = sample_points(Z, MAIN_PANEL_MAX_POINTS, rng)
    ax_main.scatter(
        Z_draw[:, 0],
        Z_draw[:, 1],
        s=0.80 if name == 'Human' else 1.35,
        alpha=0.04 if name == 'Human' else 0.10,
        c=llm_colors[name],
        edgecolors='none',
        rasterized=True,
        zorder=1
    )

for name, Z in main_projected.items():
    Z_kde = sample_points(Z, MAIN_KDE_MAX_POINTS, rng)
    sns.kdeplot(
        x=Z_kde[:, 0],
        y=Z_kde[:, 1],
        ax=ax_main,
        fill=True,
        levels=7,
        thresh=0.10,
        color=llm_colors[name],
        alpha=0.30,
        zorder=6 if name != 'Human' else 5
    )

add_trait_arrows(
    ax_main,
    loadings_h,
    scale=3.25,
    lw=2.15,
    zorder=8,
    label=True,
    label_size=7.0,
    label_offset=0.18,
    outline_arrow=True,
    outline_label=True,
    mutation_scale=11.5,
    text_zorder=9,
    arrow_color=None,
    label_color=None,
    alpha=0.88
)

for name, c in centroids.items():
    ax_main.scatter(
        c[0],
        c[1],
        s=19 if name != 'Human' else 23.5,
        c=llm_colors[name],
        edgecolors='white',
        linewidth=0.65,
        marker='s',
        zorder=26
    )

all_Z = np.vstack(list(main_projected.values()))

x_min_p, x_max_p = np.percentile(all_Z[:, 0], [0.2, 99.8])
y_min_p, y_max_p = np.percentile(all_Z[:, 1], [0.2, 99.8])

ellipse_x_min = min(r[0] for r in ellipse_ranges)
ellipse_x_max = max(r[1] for r in ellipse_ranges)
ellipse_y_min = min(r[2] for r in ellipse_ranges)
ellipse_y_max = max(r[3] for r in ellipse_ranges)

x_min = min(x_min_p, ellipse_x_min)
x_max = max(x_max_p, ellipse_x_max)
y_min = min(y_min_p, ellipse_y_min)
y_max = max(y_max_p, ellipse_y_max)

main_xlim = (min(x_min - 0.55, -4.15), max(x_max + 0.55, 4.15))
main_ylim = (min(y_min - 0.55, -4.05), max(y_max + 0.55, 4.05))

style_axis(
    ax_main,
    show_ticks=True,
    xlim=main_xlim,
    ylim=main_ylim,
    axis_lw=0.70,
    equal_data=True
)
ax_main.set_anchor('N')
ax_main.axhline(0, color='#777777', lw=0.60, zorder=12)
ax_main.axvline(0, color='#777777', lw=0.60, zorder=12)
ax_main.tick_params(length=2.5, width=0.60, labelsize=6.0, colors='#333333')

pc1_var, pc2_var = pca_h.explained_variance_ratio_[:2] * 100
ax_main.set_xlabel(f'PC1 ({pc1_var:.1f}% variance)', fontsize=7.0, labelpad=4)
ax_main.set_ylabel(f'PC2 ({pc2_var:.1f}% variance)', fontsize=7.0, labelpad=4)
add_panel_label(ax_main, 'b', x=0.00, y=1.01)

cloud_center = np.mean(np.vstack(list(centroids.values())), axis=0)

MAIN_LABEL_MODELS = set()

manual_extra = {
    'Human': (0.13, 0.06),
    'GPT-4o-mini': (0.10, -0.02),
    'GPT-5.2': (0.12, 0.13),
    'Deepseek-3.2': (0.12, 0.09),
    'Grok-3': (0.30, -0.28),
    'Llama-3.1-8B': (-0.10, 0.12),
    'Qwen-2.5-7B': (-0.20, 0.20),
    'Qwen-2.5-14B': (0.12, 0.09),
    'Qwen-2.5-32B': (0.10, -0.13),
    'Phi-mini': (0.16, -0.14),
    'Phi-small': (-0.14, 0.14),
    'Phi-medium': (0.00, 0.00),
}

annotations = []
for name, c in centroids.items():
    if name not in MAIN_LABEL_MODELS:
        continue

    auto_dx, auto_dy = choose_initial_offset(c, center=cloud_center, base=0.19)
    mdx, mdy = manual_extra.get(name, (0.0, 0.0))

    ann = add_centroid_label(
        ax_main,
        name,
        c,
        llm_colors[name],
        offset=(auto_dx + mdx, auto_dy + mdy)
    )
    annotations.append(ann)

repel_annotations(ax_main, annotations, expand_step=2.7, max_iter=90)
clamp_annotations_to_axes(ax_main, annotations, pad_frac_x=0.048, pad_frac_y=0.055)

# =====================================
# 8. Legends
# =====================================
legend_elements_trait = [
    Line2D([0], [0], color=trait_colors[t], lw=2.5, label=t)
    for t in SHORT_TRAITS
]

legend_elements_llm = [
    Line2D(
        [0], [0],
        marker='s',
        color='w',
        label=MODEL_DISPLAY_NAMES.get(model, model),
        markerfacecolor=color,
        markeredgecolor='white',
        markeredgewidth=0.9,
        markersize=4.2,
        linestyle='None'
    )
    for model, color in llm_colors.items()
]

trait_legend = ax_main.legend(
    handles=legend_elements_trait,
    title='Trait loadings',
    loc='upper right',
    ncol=5,
    frameon=True,
    fancybox=False,
    framealpha=0.94,
    facecolor='white',
    edgecolor='#DDDDDD',
    fontsize=6.0,
    title_fontsize=6.5,
    columnspacing=0.80,
    handlelength=1.30,
    handletextpad=0.35,
    borderpad=0.30
)
ax_main.add_artist(trait_legend)

ax_main.legend(
    handles=legend_elements_llm,
    title='Models',
    loc='upper center',
    bbox_to_anchor=(0.50, -0.115),
    ncol=4,
    frameon=True,
    fancybox=False,
    framealpha=0.94,
    facecolor='white',
    edgecolor='#DDDDDD',
    fontsize=6.0,
    title_fontsize=6.5,
    columnspacing=0.90,
    handletextpad=0.30,
    borderpad=0.30,
    labelspacing=0.28
)


# =====================================
# 9. Full-sample distribution profiles
# =====================================
profile_order = [name for name in SMALL_PLOT_NAMES if name in dfs_scaled]
dispersion_records = []
dispersion_stats = {}
ellipse_major_radii = {}
centroid_records = []

for name in profile_order:
    model = dfs_scaled[name].to_numpy()
    model_mean = model.mean(axis=0)
    within_distance = np.linalg.norm(model - model_mean, axis=1)
    dispersion_stats[name] = np.percentile(
        within_distance, [5, 25, 50, 75, 95]
    )

    display_name = MODEL_DISPLAY_NAMES.get(name, name)
    dispersion_records.append(pd.DataFrame({
        'Model': display_name,
        'Value': within_distance
    }))

    projected = main_projected[name]
    ellipse_params = get_cov_ellipse_params(
        projected[:, 0],
        projected[:, 1],
        level=0.95
    )
    ellipse_major_radii[name] = ellipse_params['major_radius']

dispersion_df = pd.concat(dispersion_records, ignore_index=True)
display_order = [MODEL_DISPLAY_NAMES.get(name, name) for name in profile_order]
profile_palette = {
    MODEL_DISPLAY_NAMES.get(name, name): llm_colors[name]
    for name in profile_order
}

human_reference = main_projected['Human']
human_reference_mean = human_reference.mean(axis=0)

human_point_distance = np.linalg.norm(
    human_reference - human_reference_mean,
    axis=1
)
centroid_records.append({
    'Model': 'Human',
    'Display': MODEL_DISPLAY_NAMES['Human'],
    'Centroid distance': 0.0,
    'Values': human_point_distance,
    'Stats': np.percentile(human_point_distance, [5, 25, 50, 75, 95]),
    'p-value': np.nan
})

for name in profile_order:
    if name == 'Human':
        continue

    model = main_projected[name]
    model_mean = model.mean(axis=0)
    point_distance = np.linalg.norm(model - human_reference_mean, axis=1)
    centroid_distance = np.linalg.norm(model_mean - human_reference_mean)

    n_model, n_human = model.shape[0], human_reference.shape[0]
    p_dim = model.shape[1]
    cov_model = np.cov(model, rowvar=False)
    cov_human = np.cov(human_reference, rowvar=False)
    pooled_cov = (
        (n_model - 1) * cov_model + (n_human - 1) * cov_human
    ) / (n_model + n_human - 2)
    mean_diff = model_mean - human_reference_mean
    t_squared = (n_model * n_human / (n_model + n_human)) * (
        mean_diff @ np.linalg.pinv(pooled_cov) @ mean_diff
    )
    f_value = (
        (n_model + n_human - p_dim - 1)
        / (p_dim * (n_model + n_human - 2))
    ) * t_squared
    p_value = f_dist.sf(
        f_value,
        p_dim,
        n_model + n_human - p_dim - 1
    )

    centroid_records.append({
        'Model': name,
        'Display': MODEL_DISPLAY_NAMES.get(name, name),
        'Centroid distance': centroid_distance,
        'Values': point_distance,
        'Stats': np.percentile(point_distance, [5, 25, 50, 75, 95]),
        'p-value': p_value,
        'n_model_records': n_model,
        'n_human_records': n_human,
        'T_squared': t_squared,
        'F_statistic': f_value,
        'df_numerator': p_dim,
        'df_denominator': n_model + n_human - p_dim - 1
    })

# Export the actual complete-case sample sizes and exploratory statistics.
# These are record-level tests: neither repeated-observation clustering nor
# multiplicity is accounted for. Do not treat the exported P values as
# confirmatory participant/persona-level inference.
centroid_test_table = pd.DataFrame([
    {
        'model': record['Display'],
        'centroid_distance': record['Centroid distance'],
        'n_model_records': record['n_model_records'],
        'n_human_records': record['n_human_records'],
        'T_squared': record['T_squared'],
        'F_statistic': record['F_statistic'],
        'df_numerator': record['df_numerator'],
        'df_denominator': record['df_denominator'],
        'p_value_unadjusted': record['p-value'],
        'test': 'two-sample Hotelling T-squared (pooled covariance)',
        'analysis_unit': 'complete-case state record',
        'clustering_adjusted': False,
        'multiplicity_adjusted': False,
        'interpretation': 'exploratory only'
    }
    for record in centroid_records if record['Model'] != 'Human'
])
centroid_test_table.to_csv(
    os.path.join(output_dir, 'figure2_centroid_tests_exploratory.csv'),
    index=False
)

centroid_records.sort(key=lambda record: record['Centroid distance'])
centroid_order = [record['Display'] for record in centroid_records]
centroid_frame = pd.concat([
    pd.DataFrame({'Model': record['Display'], 'Value': record['Values']})
    for record in centroid_records
], ignore_index=True)


def format_p_value(p_value):
    if np.isnan(p_value):
        return '—'
    if p_value < 1e-4:
        return '****'
    if p_value < 1e-3:
        return '***'
    if p_value < 1e-2:
        return '**'
    if p_value < 0.05:
        return '*'
    return 'ns'


def draw_combined_profiles(ax_top, ax_bottom):
    centroid_palette = {
        record['Display']: llm_colors[record['Model']]
        for record in centroid_records
    }

    sns.violinplot(
        data=centroid_frame,
        x='Model',
        y='Value',
        hue='Model',
        order=centroid_order,
        hue_order=centroid_order,
        palette=centroid_palette,
        legend=False,
        inner=None,
        cut=0,
        density_norm='width',
        bw_adjust=0.75,
        width=0.72,
        linewidth=0.85,
        saturation=1.0,
        ax=ax_top
    )

    for body, record in zip(
        ax_top.collections[:len(centroid_records)],
        centroid_records
    ):
        color = llm_colors[record['Model']]
        body.set_facecolor(to_rgba(color, 0.26))
        body.set_edgecolor(to_rgba(color, 0.86))
        body.set_linewidth(0.85)

    max_p95 = max(record['Stats'][4] for record in centroid_records)
    significance_y = max_p95 * 1.06

    for i, record in enumerate(centroid_records):
        values = record['Values']
        color = llm_colors[record['Model']]
        sample_size = min(420, values.size)
        sample_idx = rng.choice(values.size, sample_size, replace=False)
        jitter = rng.uniform(-0.15, 0.15, size=sample_size)

        ax_top.scatter(
            i + jitter,
            values[sample_idx],
            s=4.2,
            color=to_rgba(color, 0.12),
            edgecolors='none',
            rasterized=True,
            zorder=3
        )

        p05, q1, median, q3, p95 = record['Stats']
        ax_top.plot([i, i], [p05, p95], color=color, lw=1.0, alpha=0.82, zorder=5)
        ax_top.plot([i, i], [q1, q3], color=color, lw=3.5, alpha=0.90, zorder=6)
        ax_top.scatter(
            i,
            median,
            s=15,
            facecolor='white',
            edgecolor=color,
            linewidth=0.75,
            zorder=7
        )
        ax_top.scatter(
            i,
            record['Centroid distance'],
            s=19,
            marker='D',
            facecolor=color,
            edgecolor='white',
            linewidth=0.70,
            zorder=8
        )
        if not np.isnan(record['p-value']):
            ax_top.text(
                i,
                significance_y,
                format_p_value(record['p-value']),
                fontsize=5.0,
                color='#333333',
                fontweight='bold',
                ha='center',
                va='bottom'
            )

    ax_top.text(
        0.985,
        1.045,
        r'vs Human' + '\n' + r'**** $P < 10^{-4}$',
        transform=ax_top.transAxes,
        fontsize=4.8,
        ha='right',
        va='bottom'
    )

    marker_legend = [
        Line2D(
            [0], [0],
            marker='D',
            markersize=3.5,
            markerfacecolor='#666666',
            markeredgecolor='white',
            markeredgewidth=0.75,
            linestyle='None',
            label='Centroid'
        ),
        Line2D(
            [0], [0],
            marker='o',
            markersize=3.3,
            markerfacecolor='white',
            markeredgecolor='#555555',
            markeredgewidth=0.85,
            linestyle='None',
            label='Median'
        )
    ]
    ax_top.legend(
        handles=marker_legend,
        loc='upper left',
        bbox_to_anchor=(0.18, 1.060),
        ncol=2,
        frameon=False,
        fontsize=5.0,
        handletextpad=0.22,
        columnspacing=0.50,
        borderaxespad=0
    )

    n_models = len(centroid_records)
    x_positions = np.arange(n_models)
    ax_top.set_xlim(-0.5, n_models - 0.5)
    ax_top.set_ylim(0, significance_y * 1.12)
    add_panel_label(ax_top, 'c', x=0.00, y=1.08)
    ax_top.set_xlabel('')
    ax_top.set_ylabel('Distance', fontsize=6.2, labelpad=3)
    ax_top.set_xticks(x_positions)
    ax_top.set_xticklabels([])
    ax_top.tick_params(axis='x', length=0)
    ax_top.tick_params(axis='y', labelsize=5.5, length=2.1, width=0.50)
    ax_top.yaxis.grid(True, color='#E5E5E5', linewidth=0.52)
    ax_top.set_axisbelow(True)

    for spine in ['top', 'right', 'bottom']:
        ax_top.spines[spine].set_visible(False)
    ax_top.spines['left'].set_linewidth(0.60)
    ax_top.spines['left'].set_color('#555555')

    bar_values = np.array([
        ellipse_major_radii[record['Model']]
        for record in centroid_records
    ])
    colors = [
        to_rgba(llm_colors[record['Model']], 0.50)
        for record in centroid_records
    ]
    edge_colors = [
        to_rgba(llm_colors[record['Model']], 0.60)
        for record in centroid_records
    ]

    ax_bottom.bar(
        x_positions,
        -bar_values,
        width=0.44,
        color=colors,
        edgecolor=edge_colors,
        linewidth=0.45,
        zorder=2
    )

    for i, record in enumerate(centroid_records):
        ax_bottom.text(
            i,
            -0.06,
            record['Display'],
            rotation=90,
            rotation_mode='anchor',
            ha='right',
            va='center',
            fontsize=5.1,
            color='#222222',
            zorder=5
        )

    value_offset = max(bar_values) * 0.035
    for x, value in zip(x_positions, bar_values):
        ax_bottom.text(
            x,
            -value - value_offset,
            f'{value:.2f}',
            va='top',
            ha='center',
            fontsize=4.8,
            color='#444444'
        )

    max_radius = max(bar_values)
    radius_ticks = np.arange(0, np.floor(max_radius) + 1, 1)
    ax_bottom.set_xlim(-0.5, n_models - 0.5)
    ax_bottom.set_ylim(-max_radius * 1.24, 0)
    ax_bottom.set_xticks(x_positions)
    ax_bottom.set_xticklabels([])
    ax_bottom.set_yticks(-radius_ticks)
    ax_bottom.set_yticklabels([
        '' if tick == 0 else f'{tick:g}'
        for tick in radius_ticks
    ])
    ax_bottom.set_xlabel('')
    ax_bottom.set_ylabel(
        '95% ellipse\nmajor radius',
        fontsize=5.8,
        labelpad=3
    )
    ax_bottom.tick_params(axis='x', length=0)
    ax_bottom.tick_params(axis='y', labelsize=5.3, length=2.1, width=0.50)
    ax_bottom.yaxis.grid(True, color='#E5E5E5', linewidth=0.52)
    ax_bottom.set_axisbelow(True)

    for spine in ['right', 'bottom']:
        ax_bottom.spines[spine].set_visible(False)
    ax_bottom.spines['top'].set_linewidth(0.65)
    ax_bottom.spines['top'].set_color('#555555')
    ax_bottom.spines['left'].set_linewidth(0.60)
    ax_bottom.spines['left'].set_color('#555555')


def draw_combined_profiles_horizontal(ax_top, ax_bottom):
    """Restore the original horizontal c/d summary panels."""
    y = np.arange(len(centroid_records))
    values = [record['Values'] for record in centroid_records]
    parts = ax_top.violinplot(
        values,
        positions=y,
        vert=False,
        widths=0.72,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        points=160,
        bw_method=0.25
    )
    for body, record in zip(parts['bodies'], centroid_records):
        color = llm_colors[record['Model']]
        body.set_facecolor(to_rgba(color, 0.22))
        body.set_edgecolor(to_rgba(color, 0.92))
        body.set_linewidth(0.85)

    xmax = 0.0
    for yi, record in zip(y, centroid_records):
        color = llm_colors[record['Model']]
        vals = record['Values']
        sample_size = min(360, vals.size)
        idx = rng.choice(vals.size, sample_size, replace=False)
        jitter = rng.uniform(-0.13, 0.13, sample_size)
        ax_top.scatter(vals[idx], yi + jitter, s=3.0,
                       color=to_rgba(color, 0.10), edgecolors='none',
                       rasterized=True, zorder=2)
        p05, q1, med, q3, p95 = record['Stats']
        ax_top.plot([p05, p95], [yi, yi], color=color, lw=0.85, zorder=4)
        ax_top.plot([q1, q3], [yi, yi], color=color, lw=3.0, zorder=5)
        ax_top.scatter(med, yi, s=12, facecolor='white', edgecolor=color,
                       linewidth=0.65, zorder=6)
        ax_top.scatter(record['Centroid distance'], yi, s=17, marker='D',
                       facecolor=color, edgecolor='white', linewidth=0.60, zorder=7)
        xmax = max(xmax, p95)

    ax_top.set_yticks(y)
    ax_top.set_yticklabels([r['Display'] for r in centroid_records], fontsize=BAR_LABEL_SIZE)
    ax_top.invert_yaxis()
    ax_top.set_xlabel('Distance', fontsize=6.2, labelpad=2)
    ax_top.set_xlim(0, xmax * 1.16)
    ax_top.tick_params(axis='x', labelsize=5.3, length=2.1, width=0.50)
    ax_top.tick_params(axis='y', length=0, pad=2)
    ax_top.xaxis.grid(True, color='#E5E5E5', linewidth=0.48)
    ax_top.set_axisbelow(True)
    for spine in ['top', 'right', 'left']:
        ax_top.spines[spine].set_visible(False)
    ax_top.spines['bottom'].set_linewidth(0.60)
    ax_top.spines['bottom'].set_color('#555555')
    add_panel_label(ax_top, 'c', x=-0.18, y=1.03)
    ax_top.legend(
        handles=[
            Line2D([0], [0], marker='D', markersize=3.2, markerfacecolor='#666666',
                   markeredgecolor='white', linestyle='None', label='Centroid'),
            Line2D([0], [0], marker='o', markersize=3.0, markerfacecolor='white',
                   markeredgecolor='#555555', linestyle='None', label='Median')
        ],
        loc='upper center', bbox_to_anchor=(0.50, 1.04), ncol=2,
        frameon=False, fontsize=5.0, handletextpad=0.25, columnspacing=0.55
    )
    ax_top.text(0.99, 1.04, r'vs Human' + '\n' + r'**** $P < 10^{-4}$',
                transform=ax_top.transAxes, ha='right', va='bottom', fontsize=4.8)
    for yi, record in zip(y, centroid_records):
        if not np.isnan(record['p-value']):
            ax_top.text(xmax * 1.10, yi, format_p_value(record['p-value']),
                        ha='center', va='center', fontsize=5.0,
                        color='#333333', fontweight='bold', clip_on=False)

    radius_records = sorted(
        [{'Model': name, 'Display': MODEL_DISPLAY_NAMES.get(name, name),
          'Value': value} for name, value in ellipse_major_radii.items()],
        key=lambda row: row['Value'], reverse=True
    )
    yr = np.arange(len(radius_records))
    ax_bottom.barh(
        yr,
        [row['Value'] for row in radius_records],
        height=0.55,
        color=[to_rgba(llm_colors[row['Model']], 0.58) for row in radius_records],
        edgecolor=[to_rgba(llm_colors[row['Model']], 0.88) for row in radius_records],
        linewidth=0.45
    )
    ax_bottom.set_yticks(yr)
    ax_bottom.set_yticklabels([row['Display'] for row in radius_records], fontsize=BAR_LABEL_SIZE)
    ax_bottom.invert_yaxis()
    max_radius = max(row['Value'] for row in radius_records)
    for yi, row in zip(yr, radius_records):
        ax_bottom.text(row['Value'] + max_radius * 0.018, yi, f"{row['Value']:.2f}",
                       ha='left', va='center', fontsize=4.7, color='#444444')
    ax_bottom.set_xlim(0, max_radius * 1.16)
    ax_bottom.set_xlabel('95% ellipse major radius', fontsize=5.8, labelpad=2)
    ax_bottom.tick_params(axis='x', labelsize=5.1, length=2.1, width=0.50)
    ax_bottom.tick_params(axis='y', length=0, pad=2)
    ax_bottom.xaxis.grid(True, color='#E5E5E5', linewidth=0.48)
    ax_bottom.set_axisbelow(True)
    for spine in ['top', 'right', 'left']:
        ax_bottom.spines[spine].set_visible(False)
    ax_bottom.spines['bottom'].set_linewidth(0.60)
    ax_bottom.spines['bottom'].set_color('#555555')
    add_panel_label(ax_bottom, 'd', x=-0.18, y=1.00)


draw_combined_profiles_horizontal(ax_bar_dist, ax_bar_radius)


# =====================================
# 10. Save
# =====================================
plt.subplots_adjust(
    left=0.055,
    right=0.990,
    top=0.980,
    bottom=0.045
)

output_stem = os.path.join(output_dir, 'PCA_latest')
png_path = f'{output_stem}.png'
pdf_path = f'{output_stem}.pdf'

if SAVE_PNG:
    plt.savefig(
        png_path,
        dpi=PNG_DPI,
        bbox_inches='tight',
        pad_inches=0.04,
        facecolor='white'
    )

if SAVE_PDF:
    plt.savefig(
        pdf_path,
        dpi=PDF_DPI,
        bbox_inches='tight',
        pad_inches=0.04,
        facecolor='white'
    )

plt.close(fig)
print(f'Saved figure files to: {output_stem}.png/.pdf')
