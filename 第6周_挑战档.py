# -*- coding: utf-8 -*-
"""
第 6 周 · 挑战档
任务（原文）：推导/讲解 推导对偶问题求解后 w 与 b 的表达式；
             实现：从零用简化 SMO 思想实现线性 SVM 求解，验证对偶解与原始解一致。
对应教材：第 6 章

运行：
    python3 第6周_挑战档.py        # 需 numpy / scikit-learn / matplotlib

输出：
    终端：自实现 SMO 与 sklearn SVC 的 w / b / 支持向量集合 / 预测 / 目标函数对比表
    图片：wk6_chal_compare.png
"""
import os
import tempfile
import time

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mplcache"))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams["figure.dpi"] = 130


# ---------------------------------------------------------------- 1. 数据
def make_data(random_state=11, n=120):
    X, y01 = make_blobs(n_samples=n, centers=[[0.0, 0.0], [2.0, 1.8]],
                        cluster_std=1.05, random_state=random_state)
    X = StandardScaler().fit_transform(X)
    y = np.where(y01 == 1, 1, -1).astype(float)
    return X, y


# ---------------------------------------------------------------- 2. 简化 SMO
def smo(X, y, C=1.0, tol=1e-3, eps=1e-8, max_passes=10, max_iter=5000):
    """从零实现的两变量 SMO（线性核）。

    记号
    ----
    f_i = w·x_i + b,   E_i = f_i - y_i
    v_i = 1 - y_i f_i   —— 对偶目标 D(α) 对 α_i 的梯度
    可行性：α_i y_i + α_j y_j = 常数，故 α_j ∈ [L, H]
    解析步长：Δα_j = y_j (E_i - E_j) / η,  η = K_ii + K_jj - 2K_ij > 0

    关键实现细节（讲稿里要点出来）：
    * 不能只挑 |E_i - E_j| 最大的 j —— 该对可能被 [L,H] 裁剪到原地不动，
      这里按 |E_i - E_j| 从大到小**依次尝试**，取第一个能真正下降目标的 j。
    * 每个接受的对必须让对偶目标上升（dD > 0），否则跳过。
    * 收敛后用 b_from_free_sv() 把 b 终值化：递推 b 在 S 为空时可能落到可行区间外。
    """
    m = len(y)
    K = X @ X.T
    alpha = np.zeros(m)
    b = 0.0
    f = K @ (alpha * y) + b        # 初始全 0
    passes, it, n_pairs = 0, 0, 0
    hist = []

    while passes < max_passes and it < max_iter:
        it += 1
        changed = 0
        for i in range(m):
            vi = 1.0 - y[i] * f[i]
            # KKT：α_i < C 时要求 v_i <= 0；α_i > 0 时要求 v_i >= 0
            if not ((vi > tol and alpha[i] < C) or (vi < -tol and alpha[i] > 0)):
                continue

            E = f - y
            order = np.argsort(-np.abs(E - E[i]))       # 按 |E_i - E_j| 降序试
            for j in order:
                if j == i:
                    continue

                # 可行性区间
                if y[j] == y[i]:
                    L = max(0.0, alpha[i] + alpha[j] - C)
                    H = min(float(C), alpha[i] + alpha[j])
                else:
                    L = max(0.0, alpha[j] - alpha[i])
                    H = min(float(C), C + alpha[j] - alpha[i])
                if L >= H - 1e-12:
                    continue

                eta = K[i, i] + K[j, j] - 2.0 * K[i, j]
                if eta <= 1e-12:
                    continue

                # 解析解 + 裁剪
                aj_new = alpha[j] + y[j] * (E[i] - E[j]) / eta
                aj_new = min(H, max(L, aj_new))
                if abs(aj_new - alpha[j]) < eps:
                    continue
                ai_new = alpha[i] - y[i] * y[j] * (aj_new - alpha[j])

                # 目标函数变化量（对偶目标是最大化，必须 dD > 0 才接受）
                dD = (vi * (ai_new - alpha[i]) +
                      (1.0 - y[j] * f[j]) * (aj_new - alpha[j]) -
                      0.5 * eta * (aj_new - alpha[j]) ** 2)
                if dD <= 0:
                    continue

                ai_old, aj_old = alpha[i], alpha[j]
                alpha[i], alpha[j] = ai_new, aj_new

                # b 更新：优先用 0 < α < C 的分量（它们严格落在间隔边界上）
                b1 = b - E[i] - y[i] * (ai_new - ai_old) * K[i, i] - y[j] * (aj_new - aj_old) * K[i, j]
                b2 = b - E[j] - y[i] * (ai_new - ai_old) * K[i, j] - y[j] * (aj_new - aj_old) * K[j, j]
                if 0.0 < ai_new < C:
                    b = b1
                elif 0.0 < aj_new < C:
                    b = b2
                else:
                    b = 0.5 * (b1 + b2)

                f = K @ (alpha * y) + b
                changed += 1
                n_pairs += 1
                break

        passes = passes + 1 if changed == 0 else 0
        hist.append(0.5 * float(((alpha * y) @ X) @ ((alpha * y) @ X)) +
                    C * np.maximum(0.0, 1.0 - y * f).sum())

    w = (alpha * y) @ X
    # 终止后统一“终值化” b：递推 b 在 S 为空（没有自由支持向量）时会落到可行区间外，
    # 所以改用自由支持向量平均（S 为空时用可行区间中点）来定最终 b。
    b_final, _, _ = b_from_free_sv(X, y, alpha, w, C)
    info = dict(iters=it, pairs=n_pairs, hist=hist, b_rec=b,
                converged=(passes >= max_passes))
    return w, b_final, alpha, info


# ---------------------------------------------------------------- 3. 目标函数与 KKT
def primal_objective(X, y, w, b, C):
    """原始问题：½||w||² + C Σ ξ_i,  ξ_i = max(0, 1 - y_i(w·x_i+b))"""
    return 0.5 * float(w @ w) + C * float(np.maximum(0.0, 1.0 - y * (X @ w + b)).sum())


def dual_objective(alpha, w):
    """对偶问题：Σα_i - ½||w||²（因为 w = Σα_i y_i x_i，二次型就是 ½||w||²）"""
    return float(alpha.sum()) - 0.5 * float(w @ w)


def kkt_violation(X, y, alpha, b, C):
    w = (alpha * y) @ X
    yf = y * (X @ w + b)
    v = np.where(alpha < 1e-8, np.maximum(0.0, 1.0 - yf),
                 np.where(alpha > C - 1e-8, np.maximum(0.0, yf - 1.0),
                          np.abs(yf - 1.0)))
    return v, float(v.max())


def b_from_free_sv(X, y, alpha, w, C, tol=1e-8):
    """给定 w 与最终的 α，反解 b。

    情形一（|S| > 0）：自由支持向量 0 < α_i < C 由互补松弛有 y_i f(x_i) = 1，故
        b = y_i - w·x_i   （对所有 i∈S 取平均，抑制数值误差）
    情形二（|S| = 0）：没有等式可用，只能由 KKT 的两侧不等式求 b 的可行区间：
        α_i<C 且 y=+1 → b ≥ 1 - w·x_i        α_i=C 且 y=+1 → b ≤ 1 - w·x_i
        α_i<C 且 y=-1 → b ≤ -1 - w·x_i       α_i=C 且 y=-1 → b ≥ -1 - w·x_i
        b = (b_low + b_high) / 2

    返回 (b, S, 各自由支持向量给出的 b 与均值的最大偏差)。
    """
    f = X @ w
    S = np.where((alpha > tol) & (alpha < C - tol))[0]
    if S.size:
        vals = y[S] - f[S]
        return float(vals.mean()), S, float(np.abs(vals - vals.mean()).max())

    at_zero = alpha <= tol
    at_C = alpha >= C - tol
    pos, neg = (y == 1), (y == -1)

    lower = [1.0 - f[at_zero & pos], -1.0 - f[at_C & neg]]
    upper = [-1.0 - f[at_zero & neg], 1.0 - f[at_C & pos]]
    lower = [v for v in lower if v.size]
    upper = [v for v in upper if v.size]

    b_low = max(float(v.max()) for v in lower) if lower else -np.inf
    b_high = min(float(v.min()) for v in upper) if upper else np.inf
    if np.isfinite(b_low) and np.isfinite(b_high):
        return float(0.5 * (b_low + b_high)), S, 0.0
    b = b_low if np.isfinite(b_low) else b_high
    return float(b), S, 0.0


# ---------------------------------------------------------------- 4. sklearn 基线
def sklearn_baseline(X, y, C):
    clf = SVC(kernel="linear", C=C, tol=1e-3).fit(X, y)
    alpha = np.zeros(len(y))
    alpha[clf.support_] = np.abs(clf.dual_coef_.ravel())
    w = X[clf.support_].T @ clf.dual_coef_.ravel()
    return clf, w, float(clf.intercept_[0]), alpha


# ---------------------------------------------------------------- 5. 画图
def plot(X, y, w_s, b_s, w_k, b_k, alpha_s, alpha_k, hist, path):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))

    ax = axes[0]
    pad = 0.5
    xx = np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, 300)
    yy = np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, 300)
    XX, YY = np.meshgrid(xx, yy)
    G = np.c_[XX.ravel(), YY.ravel()]
    Zs = (G @ w_s + b_s).reshape(XX.shape)
    Zk = (G @ w_k + b_k).reshape(XX.shape)
    ax.contour(XX, YY, Zs, levels=[-1, 0, 1], colors="#27ae60",
               linestyles=["--", "-", "--"], linewidths=[1.2, 2.0, 1.2])
    ax.contour(XX, YY, Zk, levels=[0], colors="#c0392b",
               linestyles=":", linewidths=1.8)
    for sign, color in [(1, "#c0392b"), (-1, "#2471a3")]:
        m = y == sign
        ax.scatter(X[m, 0], X[m, 1], c=color, s=26, edgecolors="w",
                   linewidths=0.6, zorder=3)
    sv = alpha_s > 1e-4
    ax.scatter(X[sv, 0], X[sv, 1], s=120, facecolors="none", edgecolors="#1e8449",
               linewidths=1.3, zorder=4, label="SV (self SMO)")
    ax.set_title("(a) self SMO (solid) vs sklearn (dotted)", fontsize=10)
    ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1]
    ax.scatter(alpha_k, alpha_s, s=30, c="#1f3b73", edgecolors="w", linewidths=0.5)
    lim = max(alpha_s.max(), alpha_k.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.set_xlabel("alpha from sklearn")
    ax.set_ylabel("alpha from self SMO")
    ax.set_title("(b) dual coefficients agree", fontsize=10)
    ax.grid(alpha=0.2)

    ax = axes[2]
    ax.plot(range(1, len(hist) + 1), hist, "o-", ms=3, color="#8e44ad")
    ax.set_xlabel("outer pass")
    ax.set_ylabel("primal objective P(w, b)")
    ax.set_title("(c) SMO convergence", fontsize=10)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 6. main
def main():
    np.set_printoptions(precision=5, suppress=True)
    X, y = make_data()
    C = 1.0

    print("=" * 80)
    print("第 6 周 挑战档：简化 SMO 从零实现 + 与 sklearn 对拍")
    print("=" * 80)
    print(f"样本数 {len(y)}，特征 2 维（已标准化），C = {C}")

    t0 = time.time()
    w_s, b_s, alpha_s, info = smo(X, y, C=C, tol=1e-3)
    dt = time.time() - t0

    _, w_k, b_k, alpha_k = sklearn_baseline(X, y, C)

    status = "收敛" if info["converged"] else "达到迭代上限，未完全收敛"
    print(f"\n[自实现] 外层轮数 = {info['iters']}（{status}），"
          f"接受的乘子对 = {info['pairs']}，耗时 {dt:.2f}s")
    print(f"[自实现] w = [{w_s[0]:+.6f}, {w_s[1]:+.6f}]   b = {b_s:+.6f}")
    print(f"[sklearn] w = [{w_k[0]:+.6f}, {w_k[1]:+.6f}]   b = {b_k:+.6f}")

    dw = np.linalg.norm(w_s - w_k) / np.linalg.norm(w_k)
    print("\n--- 一致性 ---")
    print(f"||Δw|| / ||w_sklearn|| = {dw:.2e}    (讲稿口径：< 1e-2 视为一致)")
    print(f"|Δb| = {abs(b_s - b_k):.2e}")

    sv_s = set(np.where(alpha_s > 1e-4)[0])
    sv_k = set(np.where(alpha_k > 1e-4)[0])
    jac = len(sv_s & sv_k) / len(sv_s | sv_k)
    print(f"支持向量集合: 自实现 {len(sv_s)} 个, sklearn {len(sv_k)} 个, "
          f"Jaccard = {jac:.3f}")

    pred_s = np.where(X @ w_s + b_s >= 0, 1, -1)
    pred_k = np.where(X @ w_k + b_k >= 0, 1, -1)
    print(f"训练集预测一致率 = {(pred_s == pred_k).mean():.4f}")
    print(f"训练准确率: 自实现 {np.mean(pred_s == y):.4f} / sklearn {np.mean(pred_k == y):.4f}")

    P_s, D_s = primal_objective(X, y, w_s, b_s, C), dual_objective(alpha_s, w_s)
    P_k, D_k = primal_objective(X, y, w_k, b_k, C), dual_objective(alpha_k, w_k)
    print("\n--- 原始/对偶目标（强对偶校验，gap 应≈0）---")
    print(f"自实现 : P = {P_s:.6f}, D = {D_s:.6f}, gap = {P_s - D_s:.2e}")
    print(f"sklearn: P = {P_k:.6f}, D = {D_k:.6f}, gap = {P_k - D_k:.2e}")

    # b 的两种复核：自由支持向量平均 / （退化时）可行区间中点
    _, S, dev = b_from_free_sv(X, y, alpha_s, w_s, C)
    print("\n--- b 的复核 ---")
    print(f"自由支持向量 |S| = {S.size}（α 严格位于 (0, C) 的点），"
          f"下标 {S.tolist()}")
    if S.size:
        print(f"|S| 个点各自给出的 b 与均值的最大偏差 = {dev:.2e}（数值误差量级）")
    else:
        print("|S| = 0：没有自由支持向量，b 由 KKT 不等式给出的可行区间中点确定")
    print(f"SMO 递推 b = {info['b_rec']:+.6f}   终值化 b = {b_s:+.6f}"
          f"   差 = {abs(info['b_rec'] - b_s):.2e}")
    print(f"sklearn  b = {b_k:+.6f}     终值化 b 与 sklearn 差 = {abs(b_s - b_k):.2e}")

    _, vmax_s = kkt_violation(X, y, alpha_s, b_s, C)
    _, vmax_k = kkt_violation(X, y, alpha_k, b_k, C)
    print(f"\n最大 KKT 违反度: 自实现 = {vmax_s:.2e}, sklearn = {vmax_k:.2e}   (tol = 1e-3)")

    plot(X, y, w_s, b_s, w_k, b_k, alpha_s, alpha_k, info["hist"],
         os.path.join(HERE, "wk6_chal_compare.png"))
    print("\n图片已保存: wk6_chal_compare.png")


if __name__ == "__main__":
    main()
