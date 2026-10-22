# -*- coding: utf-8 -*-
"""
第 6 周 · 进阶档
任务（原文）：推导/讲解 解释 KKT 条件的几何含义，推导支持向量满足的条件；
             实现：说明软间隔中惩罚参数 C 的作用并实验验证。
对应教材：第 6 章

运行：
    python3 第6周_进阶档.py        # 需 numpy / scikit-learn / matplotlib

输出：
    终端：C 扫描表（支持向量数 / 间隔宽度 / 训练与测试准确率 / 违反间隔数 / KKT 三类样本统计）
    图片：wk6_adv_C_curves.png、wk6_adv_boundaries.png
"""
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mplcache"))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams["figure.dpi"] = 130

C_LIST = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]


# ---------------------------------------------------------------- 1. 数据
def make_overlap_data(random_state=3):
    """有重叠的二维数据：软间隔才有发挥空间（硬间隔会过拟合到噪声点）。"""
    X, y01 = make_blobs(
        n_samples=240,
        centers=[[0.0, 0.0], [2.2, 2.0]],
        cluster_std=1.15,              # 故意让两类交界处混住
        random_state=random_state,
    )
    X = StandardScaler().fit_transform(X)
    y = np.where(y01 == 1, 1, -1)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                          random_state=0, stratify=y)
    return Xtr, Xte, ytr, yte


# ---------------------------------------------------------------- 2. 单点实验
def probe_C(Xtr, ytr, Xte, yte, C):
    """训练一个 C，把和 KKT 相关的量全部取出来。"""
    clf = SVC(kernel="linear", C=C, tol=1e-4).fit(Xtr, ytr)

    dual = clf.dual_coef_.ravel()          # α_i y_i
    alpha = np.abs(dual)                   # α_i
    w = Xtr[clf.support_].T @ dual
    b = float(clf.intercept_[0])

    f = Xtr @ w + b
    yf = ytr * f
    # 三类样本（KKT 三条分支）：
    n_sv = len(clf.support_)
    n_at_C = int((alpha > C - 1e-6).sum())        # α_i = C：落在间隔内或误分
    n_free = int(((alpha > 1e-6) & (alpha < C - 1e-6)).sum())  # 0<α<C：正好在间隔边界上
    n_viol = int((yf < 1.0 - 1e-9).sum())         # 违反间隔（含误分）的样本数

    return dict(
        C=C,
        n_sv=n_sv,
        n_at_C=n_at_C,
        n_free=n_free,
        width=2.0 / np.linalg.norm(w),
        train_acc=(clf.predict(Xtr) == ytr).mean(),
        test_acc=(clf.predict(Xte) == yte).mean(),
        n_viol=n_viol,
        w=w, b=b, alpha=alpha, clf=clf,
    )


# ---------------------------------------------------------------- 3. 画图
def plot_curves(rows, path):
    C = np.array([r["C"] for r in rows])
    n_sv = [r["n_sv"] for r in rows]
    n_at_C = [r["n_at_C"] for r in rows]
    width = [r["width"] for r in rows]
    tr = [r["train_acc"] for r in rows]
    te = [r["test_acc"] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    ax.plot(C, n_sv, "o-", color="#c0392b", label="#support vectors")
    ax.plot(C, n_at_C, "s--", color="#e67e22", label="#SV with a_i = C")
    ax.set_xscale("log")
    ax.set_xlabel("C (log scale)")
    ax.set_ylabel("count")
    ax.set_title("(a) support vectors grow as C shrinks")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(C, width, "o-", color="#1f3b73")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("C (log scale)")
    ax.set_ylabel("2 / |w|")
    ax.set_title("(b) margin width shrinks as C grows")
    ax.grid(alpha=0.25)

    ax = axes[2]
    ax.plot(C, tr, "o-", color="#2471a3", label="train acc")
    ax.plot(C, te, "s--", color="#27ae60", label="test acc")
    ax.set_xscale("log")
    ax.set_xlabel("C (log scale)")
    ax.set_ylabel("accuracy")
    ax.set_title("(c) accuracy vs C")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.25)

    fig.suptitle("Soft-margin SVM: effect of C", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_boundaries(Xtr, ytr, rows, cs, path):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.6))
    pad = 0.5
    xx = np.linspace(Xtr[:, 0].min() - pad, Xtr[:, 0].max() + pad, 300)
    yy = np.linspace(Xtr[:, 1].min() - pad, Xtr[:, 1].max() + pad, 300)
    XX, YY = np.meshgrid(xx, yy)

    for ax, c in zip(axes.ravel(), cs):
        r = next(x for x in rows if abs(x["C"] - c) < 1e-12)
        clf = r["clf"]
        Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape)
        ax.contourf(XX, YY, Z, levels=[-1e9, -1, 1, 1e9],
                    colors=["#dbe6f6", "#f7dcd7"], alpha=0.5)
        ax.contour(XX, YY, Z, levels=[-1, 0, 1],
                   colors=["#888", "#1f3b73", "#888"],
                   linestyles=["--", "-", "--"], linewidths=[1.1, 1.8, 1.1])
        for sign, color in [(1, "#c0392b"), (-1, "#2471a3")]:
            m = ytr == sign
            ax.scatter(Xtr[m, 0], Xtr[m, 1], c=color, s=26, edgecolors="w",
                       linewidths=0.6, zorder=3)
        sv = clf.support_
        ax.scatter(Xtr[sv, 0], Xtr[sv, 1], s=110, facecolors="none",
                   edgecolors="#27ae60", linewidths=1.4, zorder=4)
        ax.set_title(f"C = {c:g} | #SV = {r['n_sv']} | "
                     f"width = {r['width']:.2f} | test acc = {r['test_acc']:.3f}",
                     fontsize=10)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.grid(alpha=0.2)

    fig.suptitle("Decision boundary vs C  (green circles = support vectors)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 4. main
def main():
    Xtr, Xte, ytr, yte = make_overlap_data()
    rows = [probe_C(Xtr, ytr, Xte, yte, c) for c in C_LIST]

    print("=" * 96)
    print("第 6 周 进阶档：软间隔惩罚参数 C 的作用（KKT 几何含义 + 实验验证）")
    print("=" * 96)
    print(f"训练集 {len(ytr)} 条 / 测试集 {len(yte)} 条，特征 2 维（已标准化）\n")
    header = (f"{'C':>7} {'#SV':>5} {'a=C':>5} {'free':>5} {'a=0':>5} "
              f"{'2/|w|':>8} {'train':>7} {'test':>7} {'viol':>5}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['C']:>7g} {r['n_sv']:>5d} {r['n_at_C']:>5d} {r['n_free']:>5d} "
              f"{len(ytr) - r['n_sv']:>5d} {r['width']:>8.3f} {r['train_acc']:>7.4f} "
              f"{r['test_acc']:>7.4f} {r['n_viol']:>5d}")

    print("\n说明：")
    print("  a=0  -> 非支持向量，位于间隔外侧，y_i f(x_i) >= 1")
    print("  0<a<C -> 落在间隔边界上，y_i f(x_i) = 1（“自由支持向量”，b 由它们定）")
    print("  a=C  -> 位于间隔内部甚至被误分，y_i f(x_i) <= 1（“边界支持向量”，决定 ξ）")

    r_small = min(rows, key=lambda r: r["C"])
    r_large = max(rows, key=lambda r: r["C"])
    print("\n两端对比（写进讲稿的结论）：")
    print(f"  C={r_small['C']:g}: #SV={r_small['n_sv']}（占 {r_small['n_sv'] / len(ytr):.0%}），"
          f"间隔宽 {r_small['width']:.2f}，训练/测试 acc={r_small['train_acc']:.3f}/{r_small['test_acc']:.3f}")
    print(f"  C={r_large['C']:g}: #SV={r_large['n_sv']}（占 {r_large['n_sv'] / len(ytr):.0%}），"
          f"间隔宽 {r_large['width']:.2f}，训练/测试 acc={r_large['train_acc']:.3f}/{r_large['test_acc']:.3f}")
    print("  -> C 小：宽容训练误差，间隔宽、支持向量多（模型简单，偏差大、方差小）")
    print("  -> C 大：重罚训练误差，间隔窄、支持向量少（模型复杂，逼近硬间隔，易过拟合）")

    plot_curves(rows, os.path.join(HERE, "wk6_adv_C_curves.png"))
    plot_boundaries(Xtr, ytr, rows, [0.03, 0.3, 3.0, 300.0],
                    os.path.join(HERE, "wk6_adv_boundaries.png"))
    print("\n图片已保存：wk6_adv_C_curves.png / wk6_adv_boundaries.png")


if __name__ == "__main__":
    main()
