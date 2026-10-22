# -*- coding: utf-8 -*-
"""
第 6 周 · 基础档
任务（原文）：推导/讲解 SVM 间隔最大化原始问题到对偶问题的拉格朗日转换；
             实现：用 sklearn 实现线性 SVM 并可视化支持向量。
对应教材：第 6 章 式 6.1–6.24

运行：
    python3 第6周_基础档.py        # 需 numpy / scikit-learn / matplotlib

输出：
    终端：w、b、间隔宽度、支持向量下标、对偶系数稀疏性、KKT 自检
    图片：wk6_basic_svm.png（与本文档同目录）
"""
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mplcache"))

import numpy as np
import matplotlib

matplotlib.use("Agg")           # 无显示环境也能出图
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams["figure.dpi"] = 130


# ---------------------------------------------------------------- 1. 数据
def make_linear_data(random_state=7):
    """二维、近似线性可分的数据。

    先标准化再训练：SVM 的间隔宽度 2/||w|| 依赖特征尺度，
    不标准化的话“间隔”没有几何意义，C 的效应也会被尺度淹没。
    """
    X, y01 = make_blobs(
        n_samples=60,
        centers=[[0.0, 0.0], [2.8, 2.2]],
        cluster_std=0.45,
        random_state=random_state,
    )
    X = StandardScaler().fit_transform(X)
    y = np.where(y01 == 1, 1, -1)          # 用 ±1 标签，公式里 y_i 直接就是符号
    return X, y


# ---------------------------------------------------------------- 2. 训练
def fit_linear_svm(X, y, C=10.0):
    """sklearn 线性 SVM，并把 w、b 从对偶解里“挖”出来。"""
    clf = SVC(kernel="linear", C=C)
    clf.fit(X, y)

    # dual_coef_ 存的就是 α_i y_i，只有支持向量处非零 —— 稀疏性的直接证据
    dual = clf.dual_coef_.ravel()
    alpha = np.abs(dual)                    # y_i = ±1，所以 α_i = |α_i y_i|
    w = X[clf.support_].T @ dual           # w = Σ α_i y_i x_i
    b = float(clf.intercept_[0])
    return clf, w, b, alpha


# ---------------------------------------------------------------- 3. 画图
def plot(clf, X, y, w, b, save_path):
    fig, ax = plt.subplots(figsize=(6.6, 5.6))

    pad = 0.6
    xx = np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, 400)
    yy = np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, 400)
    XX, YY = np.meshgrid(xx, yy)
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape)

    # 决策边界 f(x)=0 与两条间隔边界 f(x)=±1
    ax.contour(XX, YY, Z, levels=[-1.0, 0.0, 1.0],
               colors=["#888888", "#1f3b73", "#888888"],
               linestyles=["--", "-", "--"], linewidths=[1.4, 2.0, 1.4])
    ax.contourf(XX, YY, Z, levels=[-1e9, -1, 1, 1e9],
                     colors=["#dbe6f6", "#f7dcd7"], alpha=0.55)

    for sign, color, name in [(1, "#c0392b", "class +1"), (-1, "#2471a3", "class -1")]:
        m = y == sign
        ax.scatter(X[m, 0], X[m, 1], c=color, s=46, edgecolors="w",
                   linewidths=0.8, label=name, zorder=3)

    sv = clf.support_
    ax.scatter(X[sv, 0], X[sv, 1], s=190, facecolors="none",
               edgecolors="#27ae60", linewidths=1.8,
               label=f"support vectors (n={len(sv)})", zorder=4)

    # 标注支持向量的序号，答辩时方便逐个点名
    for k in sv:
        ax.annotate(str(k), (X[k, 0], X[k, 1]), textcoords="offset points",
                    xytext=(6, 6), fontsize=8, color="#1e8449")

    width = 2.0 / np.linalg.norm(w)
    ax.set_title(f"Linear SVM (sklearn), C=10, margin width 2/|w| = {width:.3f}")
    ax.set_xlabel("x1 (standardized)")
    ax.set_ylabel("x2 (standardized)")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.25)

    txt = (f"w = [{w[0]:+.3f}, {w[1]:+.3f}]\n"
           f"b = {b:+.3f}\n"
           f"|w| = {np.linalg.norm(w):.3f}, margin = 2/|w| = {width:.3f}")
    ax.text(0.98, 0.03, txt, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#bbbbbb"))
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 4. 自检
def kkt_check(X, y, w, b):
    f = X @ w + b
    yf = y * f
    return f, yf


def main():
    np.set_printoptions(precision=4, suppress=True)
    X, y = make_linear_data()
    clf, w, b, alpha = fit_linear_svm(X, y, C=10.0)
    f, yf = kkt_check(X, y, w, b)
    sv = clf.support_

    print("=" * 68)
    print("第 6 周 基础档：sklearn 线性 SVM + 支持向量可视化")
    print("=" * 68)
    print(f"样本数        : {len(y)}   (正类 {(y == 1).sum()} / 负类 {(y == -1).sum()})")
    print(f"特征维度      : {X.shape[1]}")
    print(f"支持向量个数  : {len(sv)}  -> 下标 {sv.tolist()}")
    print(f"对偶系数非零数: {(alpha > 1e-8).sum()}  (等价于支持向量个数，体现解的稀疏性)")
    print()
    print("--- 从对偶解还原原始解 ---")
    print(f"w = Σ α_i y_i x_i = [{w[0]:+.6f}, {w[1]:+.6f}]")
    print(f"b = {b:+.6f}")
    print(f"||w|| = {np.linalg.norm(w):.6f}   间隔宽度 2/||w|| = {2 / np.linalg.norm(w):.6f}")
    print()
    print("--- 对偶系数 (α_i y_i) ---")
    print("sklearn dual_coef_ =", clf.dual_coef_.ravel())
    print()
    print("--- KKT 自检 ---")
    print(f"支持向量上 y_i f(x_i) 的最大偏差 |y f - 1| = {np.abs(yf[sv] - 1).max():.2e}  (应≈0)")
    nonsv = np.setdiff1d(np.arange(len(y)), sv)
    print(f"非支持向量上 min y_i f(x_i) = {yf[nonsv].min():.4f}  (应 ≥ 1)")
    print(f"非支持向量上 α_i = 0 的个数 = {len(y) - len(sv)}")
    print(f"训练集准确率  : {(clf.predict(X) == y).mean():.4f}")
    print()

    plot(clf, X, y, w, b, os.path.join(HERE, "wk6_basic_svm.png"))
    print(f"图片已保存: {os.path.join(HERE, 'wk6_basic_svm.png')}")


if __name__ == "__main__":
    main()
