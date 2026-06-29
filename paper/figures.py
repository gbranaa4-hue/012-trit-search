import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

os.makedirs("paper/figures", exist_ok=True)

plt.rcParams.update({
    'font.family'      : 'DejaVu Sans',
    'font.size'        : 11,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'axes.grid'        : True,
    'grid.alpha'       : 0.3,
    'figure.dpi'       : 150,
})

COLORS = {
    'ResNet18'     : '#4C72B0',
    'StandardCNN'  : '#DD8452',
    'TritCognition': '#55A868',
}

data = {
    "CIFAR-10": {
        "ResNet18"     : {0:85.84, 45:33.06, 90:32.28, 135:21.85, 180:35.78, 270:33.63},
        "StandardCNN"  : {0:85.04, 45:26.43, 90:35.40, 135:19.69, 180:45.00, 270:35.64},
        "TritCognition": {0:79.81, 45:31.59, 90:30.98, 135:23.99, 180:42.19, 270:31.69},
    },
    "STL-10": {
        "ResNet18"     : {0:52.80, 45:22.05, 90:20.00, 135:16.50, 180:31.00, 270:17.90},
        "StandardCNN"  : {0:64.25, 45:13.50, 90:31.05, 135:13.00, 180:45.60, 270:29.50},
        "TritCognition": {0:56.35, 45:19.80, 90:24.35, 135:16.65, 180:39.50, 270:22.45},
    }
}

angles = [0, 45, 90, 135, 180, 270]
models = ["ResNet18", "StandardCNN", "TritCognition"]

# ── Figure 1: Rotation Robustness ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Rotation Robustness: Accuracy vs. Rotation Angle", fontsize=14, fontweight='bold', y=1.02)

for ax, ds in zip(axes, ["CIFAR-10", "STL-10"]):
    x     = np.arange(len(angles))
    width = 0.25
    for i, model in enumerate(models):
        vals = [data[ds][model][a] for a in angles]
        bars = ax.bar(x + (i-1)*width, vals, width, label=model,
                      color=COLORS[model], alpha=0.85, edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, vals):
            if val > 5:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{val:.0f}', ha='center', va='bottom', fontsize=7, color='#333')
    stab = {m: data[ds][m][0] - min(data[ds][m].values()) for m in models}
    ax.set_title(f"{ds}\nStability → ResNet18:{stab['ResNet18']:.1f}pp  StdCNN:{stab['StandardCNN']:.1f}pp  TritCog:{stab['TritCognition']:.1f}pp", fontsize=10)
    ax.set_xlabel("Rotation Angle (°)")
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}°" for a in angles])
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', fontsize=9)
    ax.axvspan(4-0.5, 4+0.5, alpha=0.06, color='green')
    ax.text(4, 2, '180°\nbest gap', ha='center', fontsize=7, color='green', style='italic')

plt.tight_layout()
plt.savefig("paper/figures/fig1_rotation_robustness.pdf", bbox_inches='tight')
plt.savefig("paper/figures/fig1_rotation_robustness.png", bbox_inches='tight')
plt.show()
print("Figure 1 saved.")

# ── Figure 2: Training Loss Curves ───────────────────────────────────────────
epochs = np.arange(1, 41)
baseline_loss = np.interp(epochs, [1,10,20,30,40], [1.54,0.65,0.43,0.26,0.18])
trit_loss     = np.interp(epochs, [1,8,9,10,20,30,40], [1.51,1.22,1.70,1.60,1.09,0.88,0.55])
std_loss      = np.interp(epochs, [1,10,20,30,40], [1.45,0.67,0.48,0.36,0.31])

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(epochs, baseline_loss, color=COLORS['ResNet18'],      linewidth=2,   label='ResNet18 (binary)')
ax.plot(epochs, std_loss,      color=COLORS['StandardCNN'],   linewidth=2,   label='StandardCNN (ablation)', linestyle='--')
ax.plot(epochs, trit_loss,     color=COLORS['TritCognition'], linewidth=2.5, label='TritCognition (012)')
ax.axvline(x=8.5, color='gray', linestyle=':', linewidth=1.5)
ax.text(4,  0.12, 'Float\nwarmup',            ha='center', fontsize=9, color='gray')
ax.text(24, 0.12, 'Ternary weights active',   ha='center', fontsize=9, color='gray')
ax.annotate('Quantization shock\n(weights → {-1,0,+1})',
            xy=(9, 1.70), xytext=(15, 1.85),
            arrowprops=dict(arrowstyle='->', color=COLORS['TritCognition']),
            fontsize=9, color=COLORS['TritCognition'])
ax.set_xlabel("Epoch")
ax.set_ylabel("Training Loss")
ax.set_title("Training Loss Curves — CIFAR-10", fontweight='bold')
ax.legend(fontsize=10)
ax.set_xlim(1, 40)
ax.set_ylim(0, 2.1)
plt.tight_layout()
plt.savefig("paper/figures/fig2_training_curves.pdf", bbox_inches='tight')
plt.savefig("paper/figures/fig2_training_curves.png", bbox_inches='tight')
plt.show()
print("Figure 2 saved.")

# ── Figure 3: Architecture Diagram ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis('off')
ax.set_facecolor('#FAFAFA'); fig.patch.set_facecolor('#FAFAFA')

def box(ax, x, y, w, h, color, label, sublabel='', fontsize=10):
    ax.add_patch(mpatches.FancyBboxPatch((x,y), w, h, boxstyle="round,pad=0.1",
        facecolor=color, edgecolor='white', linewidth=2, alpha=0.9))
    ax.text(x+w/2, y+h/2+(0.15 if sublabel else 0), label, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color='white')
    if sublabel:
        ax.text(x+w/2, y+h/2-0.25, sublabel, ha='center', va='center', fontsize=8, color='white', alpha=0.85)

def arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->', color='#555', lw=1.5))

box(ax, 0.3, 3.0, 1.4, 1.0, '#888888', 'Input x',  'B×C×H×W')
box(ax, 2.2, 5.2, 1.6, 0.9, '#4C72B0', 'Conv 1×1', 'Observer (0)')
box(ax, 2.2, 3.05,1.6, 0.9, '#DD8452', 'Conv 3×3', 'Shadow (1)')
box(ax, 2.2, 0.9, 1.6, 0.9, '#55A868', 'Conv 5×5', 'Light (2)')
box(ax, 4.2, 5.2, 1.2, 0.9, '#4C72B0', 'σ (gate)', '[0, 1]',  fontsize=9)
box(ax, 4.2, 3.05,1.2, 0.9, '#DD8452', 'tanh',     '[-1, 1]', fontsize=9)
box(ax, 4.2, 0.9, 1.2, 0.9, '#55A868', 'tanh',     '[-1, 1]', fontsize=9)
box(ax, 6.0, 2.8, 1.8, 1.4, '#8B5CF6', 'Triadic\nMerge', 's₁(1-s₀)+s₂s₀', fontsize=9)
box(ax, 8.2, 2.9, 1.6, 1.2, '#EC4899', 'Consensus\nGate', 'sign(a+b+c)', fontsize=9)
box(ax, 10.2,3.0, 1.5, 1.0, '#059669', 'Output',   'B×C\'×H×W')
box(ax, 6.0, 0.1, 1.8, 0.75,'#6B7280', 'Predictor','Conv 1×1', fontsize=9)

for y in [5.65, 3.5, 1.35]:
    ax.annotate('', xy=(2.2,y), xytext=(1.7,3.5), arrowprops=dict(arrowstyle='->', color='#555', lw=1.2))
for y in [5.65, 3.5, 1.35]:
    arrow(ax, 3.8, y, 4.2, y)
for y in [5.65, 3.5, 1.35]:
    ax.annotate('', xy=(6.0,3.5), xytext=(5.4,y), arrowprops=dict(arrowstyle='->', color='#555', lw=1.2))
arrow(ax, 7.8, 3.5, 8.2, 3.5)
arrow(ax, 9.8, 3.5, 10.2,3.5)
ax.annotate('', xy=(6.9,0.85), xytext=(6.9,2.8), arrowprops=dict(arrowstyle='->', color='#6B7280', lw=1.2, linestyle='dashed'))

ax.text(1.85,5.65,'s₀',fontsize=9,color='#4C72B0',ha='right',fontweight='bold')
ax.text(1.85,3.50,'s₁',fontsize=9,color='#DD8452',ha='right',fontweight='bold')
ax.text(1.85,1.35,'s₂',fontsize=9,color='#55A868',ha='right',fontweight='bold')
ax.text(7.05,1.8,'pred\nloss',fontsize=8,color='#6B7280',ha='left')
ax.text(3.0,6.55,'⚡ Ternary weights {-1,0,+1}',fontsize=8,color='#7C3AED',ha='center',
        bbox=dict(boxstyle='round,pad=0.3',facecolor='#EDE9FE',edgecolor='#7C3AED',alpha=0.8))
ax.set_title("TritCognition — Triadic Convolutional Block", fontsize=13, fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig("paper/figures/fig3_architecture.pdf", bbox_inches='tight')
plt.savefig("paper/figures/fig3_architecture.png", bbox_inches='tight')
plt.show()
print("Figure 3 saved.")
print("\nAll figures saved to paper/figures/")
