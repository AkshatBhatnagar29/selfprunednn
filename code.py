import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from IPython.display import clear_output

# =========================================================
# 1. PRUNABLE LINEAR LAYER (CORE REQUIREMENT)
# =========================================================
# This custom layer replaces a standard Linear layer.
# Each weight is multiplied by a learnable gate:
#   gate = sigmoid(gate_score)
# If gate → 0 → connection is pruned
# If gate → 1 → connection is active

class PrunableLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, sharpness: float = 5.0):
        super().__init__()
        self.sharpness = sharpness
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))
        nn.init.kaiming_uniform_(self.weight)
        self.gate_scores = nn.Parameter(torch.randn(out_features, in_features) * 0.01)

    def gates(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_scores * self.sharpness)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pruned_weight = self.weight * self.gates()
        return F.linear(x, pruned_weight, self.bias)

    def extra_repr(self) -> str:
        w, b = self.weight.shape
        return f"in={b}, out={w}, sharpness={self.sharpness}"


# =========================================================
# 2. MODEL DEFINITION USING PRUNABLE LAYERS
# =========================================================
# Feedforward Neural Network (MLP) using PrunableLinear

class PrunableMLP(nn.Module):
    def __init__(self, sharpness: float = 5.0):
        super().__init__()
        kw = dict(sharpness=sharpness)
        self.fc1 = PrunableLinear(3072, 512, **kw)
        self.fc2 = PrunableLinear(512,  256, **kw)
        self.fc3 = PrunableLinear(256,  128, **kw)
        self.fc4 = PrunableLinear(128,   10, **kw)
        self.prunable_layers      = [self.fc1, self.fc2, self.fc3, self.fc4]
        self.prunable_layer_names = ["fc1",    "fc2",    "fc3",    "fc4"   ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

    # Separate learning rates for weights and gates
    # IMPORTANT: gate_lr > weight_lr for effective pruning

    def get_param_groups(self, weight_lr: float, gate_lr: float) -> list:
        weight_params, gate_params = [], []
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                weight_params += [m.weight, m.bias]
                gate_params.append(m.gate_scores)
        return [
            {"params": weight_params, "lr": weight_lr},
            {"params": gate_params,   "lr": gate_lr},
        ]
# =========================================================
# 3. SPARSITY LOSS (L1 REGULARIZATION ON GATES)
# =========================================================
# Encourages gates → 0 → pruning

def compute_sparsity_loss(model: nn.Module) -> torch.Tensor:
    layer_means = []
    for m in model.modules():
        if isinstance(m, PrunableLinear):
            layer_means.append(m.gates().mean())
    return torch.stack(layer_means).mean()

# =========================================================
# 4. SPARSITY METRIC (EVALUATION)
# =========================================================
# Measures % of gates below threshold

def compute_sparsity_ratio(model: nn.Module, threshold: float = 0.5) -> float:
    total = pruned = 0
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, PrunableLinear):
                g      = m.gates()
                total  += g.numel()
                pruned += (g < threshold).sum().item()
    return pruned / total if total > 0 else 0.0

# =========================================================
# 5. DATA LOADING (CIFAR-10)
# =========================================================

def get_cifar10_loaders(batch_size: int = 128):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)
    # Data augmentation for training
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    # Normal preprocessing for testing
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_ds = torchvision.datasets.CIFAR10('./data', train=True,  download=True, transform=train_tf)
    test_ds  = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=test_tf)
    train_ld = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_ld  = torch.utils.data.DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_ld, test_ld

# =========================================================
# 6. TRAINING LOOP
# =========================================================
# Loss = CrossEntropy + λ * SparsityLoss

def train_one_epoch(model, loader, optimizer, lambda_sparse: float, device) -> tuple:
    model.train()
    total_ce = total_sp = n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out     = model(x)
        ce_loss = F.cross_entropy(out, y)
        sp_loss = compute_sparsity_loss(model)
        loss    = ce_loss + lambda_sparse * sp_loss
        loss.backward()
        optimizer.step()
        total_ce += ce_loss.item()
        total_sp += sp_loss.item()
        n        += 1
    return total_ce / n, total_sp / n

# =========================================================
# 7. EVALUATION LOOP (CORE REQUIREMENT)
# =========================================================
# Computes test accuracy

@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y    = x.to(device), y.to(device)
        preds   = model(x).argmax(dim=1)
        total   += y.size(0)
        correct += (preds == y).sum().item()
    return 100.0 * correct / total

# =========================================================
# 9. TRAINING DASHBOARD (VISUALIZATION DURING TRAINING)
# =========================================================
# Displays:
# - Accuracy vs Epoch
# - Sparsity vs Epoch
# - Loss breakdown
# - Layer-wise gate heatmaps
# - Gate distribution histogram

def plot_training_dashboard(history, model, epoch, total_epochs, lambda_sparse):
    AMBER = '#EF9F27'; TEAL  = '#1D9E75'
    PURP  = '#7F77DD'; CORAL = '#D85A30'
    BG = '#0f0f0f'; PANEL = '#1a1a2e'; GRID = '#333355'
    FAINT = '#aaaaaa'; TITLE = '#ccccee'
    LAYER_COLORS = [AMBER, TEAL, PURP, CORAL]
    clear_output(wait=True)
    fig = plt.figure(figsize=(16, 12), facecolor=BG)
    gs  = gridspec.GridSpec(4, 4, figure=fig, hspace=0.50, wspace=0.38, left=0.06, right=0.97, top=0.93,  bottom=0.05)
    ax_acc  = fig.add_subplot(gs[0, :2])
    ax_sp   = fig.add_subplot(gs[0, 2:])
    ax_loss = fig.add_subplot(gs[1, :])
    ax_hm   = [fig.add_subplot(gs[2, i]) for i in range(4)]
    ax_hist = fig.add_subplot(gs[3, :])

    def style(ax, title):
        ax.set_facecolor(PANEL); ax.set_title(title, color=TITLE, fontsize=10, pad=5)
        ax.tick_params(colors=FAINT, labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID)

    epochs = history['epoch']
    style(ax_acc, 'Test Accuracy %')
    ax_acc.plot(epochs, history['acc'], color=TEAL, lw=2)
    ax_acc.fill_between(epochs, history['acc'], alpha=0.12, color=TEAL)
    ax_acc.set_xlabel('Epoch', color=FAINT, fontsize=8)
    ax_acc.set_ylabel('Acc %', color=FAINT, fontsize=8)
    if history['acc']:
        ax_acc.annotate(f"{history['acc'][-1]:.2f}%", xy=(epochs[-1], history['acc'][-1]), xytext=(4, 0), textcoords='offset points', color=TEAL, fontsize=9, fontweight='bold')

    style(ax_sp, 'Sparsity — gates < 0.5 (%)')
    sp_pct = [s * 100 for s in history['sp']]
    ax_sp.plot(epochs, sp_pct, color=AMBER, lw=2)
    ax_sp.fill_between(epochs, sp_pct, alpha=0.12, color=AMBER)
    ax_sp.set_ylim(0, 105)
    ax_sp.set_xlabel('Epoch', color=FAINT, fontsize=8)
    ax_sp.set_ylabel('Pruned %', color=FAINT, fontsize=8)
    if sp_pct:
        ax_sp.annotate(f"{sp_pct[-1]:.1f}%", xy=(epochs[-1], sp_pct[-1]), xytext=(4, 0), textcoords='offset points', color=AMBER, fontsize=9, fontweight='bold')

    style(ax_loss, 'Loss breakdown')
    weighted = [lambda_sparse * s for s in history['sp_loss']]
    total    = [c + w for c, w in zip(history['ce'], weighted)]
    ax_loss.plot(epochs, history['ce'], color=TEAL,  lw=2,   label='CE loss')
    ax_loss.plot(epochs, weighted,      color=AMBER, lw=2,   label=f'λ·SP  (λ={lambda_sparse})')
    ax_loss.plot(epochs, total, color='white', lw=1.5, linestyle='--', alpha=0.6, label='Total loss')
    ax_loss.set_xlabel('Epoch', color=FAINT, fontsize=8)
    ax_loss.set_ylabel('Loss',  color=FAINT, fontsize=8)
    ax_loss.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TITLE, loc='upper right')

    all_gates = []
    with torch.no_grad():
        for i, (name, m) in enumerate(zip(model.prunable_layer_names, model.prunable_layers)):
            g = m.gates().cpu().numpy()
            all_gates.append(g.flatten())
            gr = g[:64, :64]
            ax = ax_hm[i]; ax.set_facecolor(PANEL)
            im = ax.imshow(gr, cmap='inferno', vmin=0, vmax=1, aspect='auto', interpolation='nearest')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            pct = (g < 0.5).mean() * 100
            ax.set_title(f'{name}  —  {pct:.0f}% pruned\nshape {g.shape}', color=TITLE, fontsize=8, pad=3)
            ax.set_xlabel('input neurons',  color=FAINT, fontsize=7)
            ax.set_ylabel('output neurons', color=FAINT, fontsize=7)
            ax.tick_params(colors=FAINT, labelsize=7)
            for sp in ax.spines.values(): sp.set_edgecolor(GRID)

    combined = np.concatenate(all_gates)
    pct_total = (combined < 0.5).mean() * 100
    style(ax_hist, f'Gate distribution — epoch {epoch}/{total_epochs}  |  {pct_total:.1f}% of gates below 0.5  (bimodal split = pruning working)')
    bins = np.linspace(0, 1, 51)
    for i, (name, gv) in enumerate(zip(model.prunable_layer_names, all_gates)):
        ax_hist.hist(gv, bins=bins, alpha=0.55, color=LAYER_COLORS[i], label=name, histtype='stepfilled', edgecolor='none')
    ax_hist.axvspan(0, 0.1,  alpha=0.14, color='#ff4444', label='dead zone (< 0.1)')
    ax_hist.axvline(0.5, color='white', lw=1, linestyle='--', alpha=0.5, label='threshold 0.5')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_xlabel('Gate value', color=FAINT, fontsize=8)
    ax_hist.set_ylabel('Count',      color=FAINT, fontsize=8)
    ax_hist.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TITLE, ncol=3)

    fig.suptitle(f'Self-Pruning MLP  ·  CIFAR-10  ·  λ={lambda_sparse}  ·  Epoch {epoch}/{total_epochs}', color='white', fontsize=13, fontweight='bold')
    plt.savefig(f'pruning_cifar_lam{lambda_sparse}_ep{epoch:03d}.png', dpi=100, bbox_inches='tight', facecolor=BG)
    plt.show()

def plot_gate_distribution_final(model, lambda_sparse, accuracy, sparsity):
    AMBER = '#EF9F27'; TEAL  = '#1D9E75'
    PURP  = '#7F77DD'; CORAL = '#D85A30'
    BG = '#0f0f0f'; PANEL = '#1a1a2e'; GRID = '#333355'
    FAINT = '#aaaaaa'; TITLE = '#ccccee'
    LAYER_COLORS = [AMBER, TEAL, PURP, CORAL]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
    fig.suptitle(f'Final Gate Distribution  ·  CIFAR-10  ·  λ={lambda_sparse}  ·  Acc={accuracy:.2f}%  ·  Sparsity@0.5={sparsity:.1f}%', color='white', fontsize=12, fontweight='bold')
    all_gates = []
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, PrunableLinear):
                all_gates.append(m.gates().cpu().numpy().flatten())
    combined = np.concatenate(all_gates)

    ax = axes[0]
    ax.set_facecolor(PANEL)
    bins = np.linspace(0, 1, 61)
    for i, (name, gv) in enumerate(zip(model.prunable_layer_names, all_gates)):
        ax.hist(gv, bins=bins, alpha=0.6, color=LAYER_COLORS[i], label=name, histtype='stepfilled', edgecolor='none')
    ax.axvspan(0, 0.1, alpha=0.15, color='#ff4444', label='dead zone (gate < 0.1)')
    ax.axvline(0.5, color='white', lw=1.5, linestyle='--', alpha=0.7, label='threshold = 0.5')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Gate value', color=FAINT, fontsize=10)
    ax.set_ylabel('Number of gates', color=FAINT, fontsize=10)
    ax.set_title('All layers — gate distribution\n(spike near 0 = pruned  ·  cluster right = active)', color=TITLE, fontsize=10)
    ax.tick_params(colors=FAINT, labelsize=9)
    ax.legend(fontsize=9, facecolor=PANEL, edgecolor=GRID, labelcolor=TITLE)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)

    ax = axes[1]
    ax.set_facecolor(PANEL)
    sorted_gates = np.sort(combined)
    cdf = np.arange(1, len(sorted_gates) + 1) / len(sorted_gates) * 100
    ax.plot(sorted_gates, cdf, color=AMBER, lw=2)
    for thr, col, ls in [(0.01, CORAL, '--'), (0.1, PURP, '--'), (0.5, TEAL, '-.')]:
        pct = (combined < thr).mean() * 100
        ax.axvline(thr, color=col, lw=1.2, linestyle=ls, label=f'< {thr}  →  {pct:.1f}% pruned')
        ax.axhline(pct, color=col, lw=0.6, linestyle=':', alpha=0.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 105)
    ax.set_xlabel('Gate threshold', color=FAINT, fontsize=10)
    ax.set_ylabel('Cumulative % of gates pruned', color=FAINT, fontsize=10)
    ax.set_title('Cumulative sparsity vs threshold\n(steep early rise = heavy pruning near 0)', color=TITLE, fontsize=10)
    ax.tick_params(colors=FAINT, labelsize=9)
    ax.legend(fontsize=9, facecolor=PANEL, edgecolor=GRID, labelcolor=TITLE)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    plt.tight_layout()
    fname = f'gate_distribution_final_lam{lambda_sparse}.png'
    plt.savefig(fname, dpi=120, bbox_inches='tight', facecolor=BG)
    plt.show()
    print(f"  Saved → {fname}")

def plot_lambda_sweep(results: dict):
    AMBER = '#EF9F27'; TEAL = '#1D9E75'; PURP = '#7F77DD'; CORAL = '#D85A30'
    BG = '#0f0f0f'; PANEL = '#1a1a2e'; GRID = '#333355'
    FAINT = '#aaaaaa'; TITLE = '#ccccee'
    lams = list(results.keys())
    accs = [results[l]['acc']  for l in lams]
    s50  = [results[l]['s50']  for l in lams]
    s10  = [results[l]['s10']  for l in lams]
    s01  = [results[l]['s01']  for l in lams]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=BG)
    fig.suptitle('Lambda Sweep — CIFAR-10 Self-Pruning MLP', color='white', fontsize=13, fontweight='bold')

    def style(ax, title, xlabel, ylabel):
        ax.set_facecolor(PANEL); ax.set_title(title, color=TITLE, fontsize=11)
        ax.set_xlabel(xlabel, color=FAINT, fontsize=10)
        ax.set_ylabel(ylabel, color=FAINT, fontsize=10)
        ax.tick_params(colors=FAINT, labelsize=9)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID)

    ax = axes[0]
    style(ax, 'Accuracy vs λ', 'λ (lambda)', 'Test Accuracy %')
    ax.plot(lams, accs, color=TEAL, lw=2.5, marker='o', markersize=8)
    for l, a in zip(lams, accs):
        ax.annotate(f'{a:.1f}%', xy=(l, a), xytext=(0, 9), textcoords='offset points', color=TEAL, fontsize=8, ha='center')

    ax = axes[1]
    style(ax, 'Sparsity vs λ', 'λ (lambda)', 'Pruned gates %')
    ax.plot(lams, s50, color=AMBER, lw=2, marker='o', ms=6, label='gate < 0.5')
    ax.plot(lams, s10, color=PURP,  lw=2, marker='s', ms=6, label='gate < 0.1')
    ax.plot(lams, s01, color=CORAL, lw=2, marker='^', ms=6, label='gate < 0.01')
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9, facecolor=PANEL, edgecolor=GRID, labelcolor=TITLE)

    ax = axes[2]
    style(ax, 'Pareto: Accuracy vs Sparsity', 'Sparsity @ 0.01 (%)', 'Test Accuracy %')
    ax.plot(s01, accs, color=AMBER, lw=2.5, marker='o', markersize=9)
    for l, s, a in zip(lams, s01, accs):
        ax.annotate(f'λ={l}', xy=(s, a), xytext=(5, 4), textcoords='offset points', color=FAINT, fontsize=8)
    best_acc = max(accs)
    ax.axhline(best_acc - 1.0, color=TEAL,  lw=1, linestyle='--', alpha=0.7, label='−1% accuracy drop')
    ax.axhline(best_acc - 2.0, color=CORAL, lw=1, linestyle='--', alpha=0.7, label='−2% accuracy drop')
    ax.legend(fontsize=9, facecolor=PANEL, edgecolor=GRID, labelcolor=TITLE)
    plt.tight_layout()
    plt.savefig('lambda_sweep_cifar.png', dpi=120, bbox_inches='tight', facecolor=BG)
    plt.show()
    print("  Saved → lambda_sweep_cifar.png")

# =========================================================
# 8. MAIN EXPERIMENT LOOP (λ SWEEP)
# =========================================================

LAMBDAS    = [0.0, 0.01, 0.05, 0.1]
EPOCHS     = 50
BATCH_SIZE = 128
WEIGHT_LR  = 1e-3
GATE_LR    = 2e-2
SHARPNESS  = 5.0
PLOT_EVERY = 10
THRESHOLD  = 1e-2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {device}")
print(f"Lambdas: {LAMBDAS}  |  Epochs: {EPOCHS}")
print("\nDownloading / loading CIFAR-10 ...")
train_ld, test_ld = get_cifar10_loaders(BATCH_SIZE)

results = {}

for lam in LAMBDAS:
    print(f"\n{'═'*55}")
    print(f"  λ = {lam}")
    print(f"{'═'*55}")

    model     = PrunableMLP(sharpness=SHARPNESS).to(device)
    optimizer = torch.optim.Adam(model.get_param_groups(WEIGHT_LR, GATE_LR))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = dict(epoch=[], acc=[], sp=[], ce=[], sp_loss=[])

    for epoch in range(1, EPOCHS + 1):
        ce_avg, sp_avg = train_one_epoch(model, train_ld, optimizer, lam, device)
        scheduler.step()

        acc = evaluate(model, test_ld, device)
        sp  = compute_sparsity_ratio(model, threshold=0.5)

        history['epoch'].append(epoch)
        history['acc'].append(acc)
        history['sp'].append(sp)
        history['ce'].append(ce_avg)
        history['sp_loss'].append(sp_avg)

        print(f"  Ep {epoch:>3}/{EPOCHS} | CE={ce_avg:.4f} | SP={sp_avg:.4f} | λ·SP={lam*sp_avg:.4f} | Acc={acc:.2f}% | Sparsity={sp*100:.1f}%")

        if epoch % PLOT_EVERY == 0 or epoch == EPOCHS:
            plot_training_dashboard(history, model, epoch, EPOCHS, lam)

    final_acc = history['acc'][-1]
    s50  = compute_sparsity_ratio(model, 0.50) * 100
    s10  = compute_sparsity_ratio(model, 0.10) * 100
    s01  = compute_sparsity_ratio(model, THRESHOLD) * 100
    results[lam] = dict(acc=final_acc, s50=s50, s10=s10, s01=s01, model=model)

    print(f"\n  Final  λ={lam}  Acc={final_acc:.2f}%  Sp@0.5={s50:.1f}%  Sp@0.1={s10:.1f}%  Sp@1e-2={s01:.1f}%")
    
    print(f"\n  Final State of the Network (Layer-wise Sparsity @ threshold={THRESHOLD}):")
    with torch.no_grad():
        for name, m in zip(model.prunable_layer_names, model.prunable_layers):
            g = m.gates()
            layer_total = g.numel()
            layer_pruned = (g < THRESHOLD).sum().item()
            layer_sp = (layer_pruned / layer_total) * 100.0
            print(f"  - {name}: {layer_sp:.2f}% pruned")

    plot_gate_distribution_final(model, lam, final_acc, s50)

plot_lambda_sweep(results)

baseline = results[0.0]['acc']
print(f"\n{'═'*65}")
print(f"  CIFAR-10 SELF-PRUNING — FINAL SUMMARY")
print(f"{'═'*65}")
print(f"  {'λ':>6} | {'Acc%':>7} | {'Drop':>6} | {'Sp@0.5':>8} | {'Sp@0.1':>8} | {'Sp@1e-2':>9}")
print(f"  {'─'*6}-+-{'─'*7}-+-{'─'*6}-+-{'─'*8}-+-{'─'*8}-+-{'─'*9}")
for lam, r in results.items():
    drop = r['acc'] - baseline
    candidates = {l: v for l, v in results.items() if abs(v['acc'] - baseline) < 2.0}
    best_lam = max(candidates,key=lambda l: candidates[l]['s01'] - abs(candidates[l]['acc'] - baseline))
    tag = "  ← best λ" if lam == best_lam else ""
    print(f"  {lam:>6} | {r['acc']:>7.2f} | {drop:>+6.2f} | {r['s50']:>7.1f}% | {r['s10']:>7.1f}% | {r['s01']:>8.1f}%{tag}")

print(f"\n  Saved plots: pruning_cifar_*.png, gate_distribution_*.png, lambda_sweep_cifar.png")

