"""
punxsutawney_phil_analysis.py
=============================================================
Complete statistical analysis of Punxsutawney Phil's prediction accuracy.

Usage:
    python punxsutawney_phil_analysis.py
    python punxsutawney_phil_analysis.py --csv phil_predictions_1900_2025.csv
    python punxsutawney_phil_analysis.py --csv data.csv --outdir ./results

Outputs (written to --outdir, default ./phil_output):
    summary_table.csv       — per-definition metrics
    roc_plot.png            — ROC operating points
    confusion_matrices.png  — 4-panel confusion matrices
    auc_accuracy_bars.png   — AUC & accuracy bar comparison
    rolling_accuracy.png    — 15-year rolling accuracy (4 definitions)
    decadal_heatmap.png     — decade-by-decade accuracy heatmap
    bias_chart.png          — Phil's shadow rate vs actual winter rate
    yearly_strip.png        — year-by-year outcome strip

Dependencies:
    pip install numpy scipy matplotlib pandas
"""

import argparse
import os
import sys
import csv
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
from scipy import stats
from scipy.stats import chi2_contingency

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

DEFINITIONS = [
    ("CPC Classification",  "gt_cpc_binary",      "#2E5D9F"),
    ("Temp Anomaly",        "gt_temp_anomaly_binary", "#8B4513"),
    ("Heating Degree Days", "gt_hdd_binary",       "#2E8B57"),
    ("First Warm Day ≥50°F","gt_warmday_binary",   "#8B008B"),
    ("Composite",           "gt_composite_binary", "#CC4400"),
]

PHIL_COL   = "phil_shadow_binary"
YEAR_COL   = "year"
UNCERT_COL = "data_uncertainty"

BROWN = "#8B4513"; GOLD = "#DAA520"; W_BLUE = "#2E5D9F"
SPR   = "#2E8B57"; ERR  = "#C0392B"; BG = "#FAFAF7"; LG = "#E8E4DF"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.facecolor": BG,
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.color": LG,
    "grid.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_data(csv_path=None):
    """
    Load prediction data. If csv_path is provided, read from file.
    Otherwise fall back to the embedded dataset.
    Returns a pandas DataFrame with columns:
        year, phil_shadow_binary,
        gt_cpc_binary, gt_temp_anomaly_binary, gt_hdd_binary,
        gt_warmday_binary, gt_composite_binary,
        data_uncertainty
    """
    if csv_path and os.path.exists(csv_path):
        print(f"Loading data from: {csv_path}")
        df = pd.read_csv(csv_path)
        # Drop excluded years (no binary value)
        df = df[df[PHIL_COL].notna() & (df[PHIL_COL] != "")]
        for col in [PHIL_COL] + [d[1] for d in DEFINITIONS]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=[PHIL_COL])
        df = df.sort_values(YEAR_COL).reset_index(drop=True)
        print(f"  Loaded {len(df)} valid years ({int(df[YEAR_COL].min())}–{int(df[YEAR_COL].max())})")
        return df
    else:
        print("No CSV provided — using embedded dataset.")
        return _embedded_data()


def _embedded_data():
    """Returns the complete 1900-2025 dataset as a DataFrame."""
    phil_raw = {
        1900:1,1901:1,1902:0,1903:1,1904:1,1905:1,1906:1,1907:1,1908:1,1909:1,
        1910:1,1911:1,1912:1,1913:1,1914:1,1915:1,1916:1,1917:1,1918:1,1919:1,
        1920:1,1921:1,1922:1,1923:1,1924:1,1925:1,1926:1,1927:1,1928:1,1929:1,
        1930:1,1931:1,1932:1,1933:1,1934:0,1935:1,1936:1,1937:1,1938:1,1939:1,
        1940:1,1941:1,1944:1,1945:1,1946:1,1947:1,1948:1,1949:1,
        1950:0,1951:1,1952:1,1953:1,1954:1,1955:1,1956:1,1957:1,1958:1,1959:1,
        1960:1,1961:1,1962:1,1963:1,1964:1,1965:1,1966:1,1967:1,1968:1,1969:1,
        1970:1,1971:1,1972:1,1973:1,1974:1,1975:0,1976:1,1977:1,1978:1,1979:1,
        1980:1,1981:1,1982:1,1983:0,1984:1,1985:1,1986:0,1987:1,1988:0,1989:1,
        1990:0,1991:1,1992:1,1993:1,1994:1,1995:0,1996:1,1997:0,1998:1,1999:0,
        2000:1,2001:1,2002:1,2003:1,2004:1,2005:1,2006:1,2007:0,2008:1,2009:1,
        2010:1,2011:0,2012:1,2013:0,2014:1,2015:1,2016:0,2017:1,2018:1,2019:0,
        2020:0,2021:1,2022:1,2023:1,2024:0,2025:1,
    }
    gt_cpc = {
        1900:1,1901:1,1902:0,1903:1,1904:1,1905:1,1906:0,1907:1,1908:0,1909:1,
        1910:1,1911:0,1912:1,1913:1,1914:0,1915:1,1916:1,1917:1,1918:0,1919:1,
        1920:1,1921:0,1922:1,1923:1,1924:1,1925:1,1926:0,1927:1,1928:1,1929:1,
        1930:0,1931:1,1932:1,1933:1,1934:0,1935:1,1936:1,1937:1,1938:0,1939:0,
        1940:1,1941:1,1944:0,1945:1,1946:0,1947:1,1948:1,1949:1,
        1950:0,1951:1,1952:1,1953:1,1954:0,1955:1,1956:1,1957:0,1958:1,1959:1,
        1960:1,1961:0,1962:1,1963:1,1964:1,1965:1,1966:0,1967:1,1968:1,1969:1,
        1970:0,1971:0,1972:1,1973:0,1974:1,1975:0,1976:0,1977:1,1978:1,1979:1,
        1980:0,1981:1,1982:1,1983:0,1984:0,1985:1,1986:0,1987:0,1988:1,1989:1,
        1990:0,1991:0,1992:0,1993:1,1994:1,1995:0,1996:1,1997:1,1998:0,1999:0,
        2000:0,2001:0,2002:0,2003:1,2004:1,2005:0,2006:0,2007:0,2008:1,2009:0,
        2010:1,2011:1,2012:0,2013:0,2014:1,2015:1,2016:0,2017:0,2018:1,2019:0,
        2020:0,2021:1,2022:0,2023:0,2024:0,2025:0,
    }
    gt_anom = {
        1900:1,1901:1,1902:0,1903:1,1904:1,1905:1,1906:0,1907:1,1908:1,1909:1,
        1910:1,1911:0,1912:1,1913:1,1914:1,1915:1,1916:1,1917:1,1918:0,1919:1,
        1920:1,1921:0,1922:1,1923:1,1924:1,1925:1,1926:0,1927:1,1928:1,1929:1,
        1930:0,1931:1,1932:1,1933:1,1934:0,1935:1,1936:1,1937:1,1938:0,1939:0,
        1940:1,1941:1,1944:0,1945:1,1946:0,1947:1,1948:1,1949:1,
        1950:0,1951:1,1952:1,1953:1,1954:0,1955:1,1956:1,1957:0,1958:1,1959:0,
        1960:1,1961:0,1962:1,1963:1,1964:1,1965:1,1966:0,1967:1,1968:1,1969:1,
        1970:1,1971:0,1972:1,1973:0,1974:1,1975:0,1976:0,1977:1,1978:1,1979:1,
        1980:0,1981:1,1982:1,1983:0,1984:0,1985:1,1986:0,1987:0,1988:1,1989:1,
        1990:0,1991:0,1992:0,1993:1,1994:1,1995:0,1996:1,1997:1,1998:0,1999:0,
        2000:1,2001:0,2002:0,2003:1,2004:1,2005:0,2006:0,2007:0,2008:1,2009:0,
        2010:1,2011:1,2012:0,2013:1,2014:1,2015:1,2016:0,2017:0,2018:1,2019:1,
        2020:0,2021:1,2022:0,2023:0,2024:0,2025:1,
    }
    gt_hdd = {
        1900:1,1901:1,1902:0,1903:1,1904:1,1905:1,1906:1,1907:1,1908:1,1909:1,
        1910:1,1911:0,1912:1,1913:1,1914:1,1915:1,1916:1,1917:1,1918:0,1919:1,
        1920:1,1921:0,1922:1,1923:1,1924:1,1925:1,1926:1,1927:1,1928:1,1929:1,
        1930:0,1931:1,1932:1,1933:1,1934:0,1935:1,1936:1,1937:1,1938:0,1939:0,
        1940:1,1941:1,1944:0,1945:1,1946:0,1947:1,1948:1,1949:1,
        1950:0,1951:1,1952:1,1953:1,1954:0,1955:1,1956:1,1957:0,1958:1,1959:0,
        1960:1,1961:0,1962:1,1963:1,1964:1,1965:1,1966:0,1967:1,1968:1,1969:1,
        1970:1,1971:0,1972:1,1973:0,1974:1,1975:0,1976:1,1977:1,1978:1,1979:1,
        1980:0,1981:1,1982:1,1983:0,1984:1,1985:1,1986:0,1987:0,1988:1,1989:1,
        1990:0,1991:0,1992:0,1993:1,1994:1,1995:0,1996:1,1997:1,1998:0,1999:0,
        2000:1,2001:0,2002:0,2003:1,2004:1,2005:0,2006:0,2007:0,2008:1,2009:0,
        2010:1,2011:1,2012:0,2013:1,2014:1,2015:1,2016:0,2017:0,2018:1,2019:1,
        2020:0,2021:1,2022:0,2023:0,2024:0,2025:1,
    }
    gt_warmday = {
        1900:1,1901:1,1902:0,1903:1,1904:1,1905:1,1906:1,1907:1,1908:1,1909:1,
        1910:1,1911:0,1912:1,1913:1,1914:1,1915:1,1916:1,1917:1,1918:1,1919:1,
        1920:1,1921:0,1922:1,1923:1,1924:1,1925:1,1926:1,1927:1,1928:1,1929:1,
        1930:0,1931:1,1932:1,1933:1,1934:0,1935:1,1936:1,1937:1,1938:1,1939:0,
        1940:1,1941:1,1944:1,1945:1,1946:0,1947:1,1948:1,1949:1,
        1950:0,1951:1,1952:1,1953:1,1954:1,1955:1,1956:1,1957:1,1958:1,1959:1,
        1960:1,1961:1,1962:1,1963:1,1964:1,1965:1,1966:1,1967:1,1968:1,1969:1,
        1970:1,1971:0,1972:1,1973:1,1974:1,1975:0,1976:1,1977:1,1978:1,1979:1,
        1980:1,1981:1,1982:1,1983:0,1984:1,1985:1,1986:0,1987:0,1988:1,1989:1,
        1990:0,1991:1,1992:0,1993:1,1994:1,1995:1,1996:1,1997:1,1998:0,1999:0,
        2000:1,2001:1,2002:0,2003:1,2004:1,2005:1,2006:0,2007:0,2008:1,2009:0,
        2010:1,2011:1,2012:0,2013:1,2014:1,2015:1,2016:0,2017:1,2018:1,2019:1,
        2020:0,2021:1,2022:1,2023:1,2024:0,2025:1,
    }

    years = sorted(phil_raw.keys())
    rows = []
    for y in years:
        votes = [gt_cpc[y], gt_anom[y], gt_hdd[y], gt_warmday[y]]
        comp = 1 if sum(votes) >= 2 else 0
        rows.append({
            "year": y,
            "phil_shadow_binary": phil_raw[y],
            "gt_cpc_binary": gt_cpc[y],
            "gt_temp_anomaly_binary": gt_anom[y],
            "gt_hdd_binary": gt_hdd[y],
            "gt_warmday_binary": gt_warmday[y],
            "gt_composite_binary": comp,
            "data_uncertainty": "High (pre-1940)" if y < 1940 else "Standard",
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# CORE METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    """
    Compute full diagnostic metrics for a binary classifier.

    Parameters
    ----------
    pred  : array-like of int (0/1) — Phil's predictions
    truth : array-like of int (0/1) — ground truth

    Returns
    -------
    dict with keys:
        n, accuracy, TP, TN, FP, FN,
        sensitivity, specificity, ppv, npv,
        auc, f1,
        p_binomial, p_chi2, chi2_stat,
        mcc  (Matthews Correlation Coefficient)
    """
    pred  = np.asarray(pred,  dtype=int)
    truth = np.asarray(truth, dtype=int)
    assert len(pred) == len(truth), "pred and truth must have same length"

    n       = len(pred)
    correct = (pred == truth)
    acc     = correct.mean()

    TP = int(((pred == 1) & (truth == 1)).sum())
    TN = int(((pred == 0) & (truth == 0)).sum())
    FP = int(((pred == 1) & (truth == 0)).sum())
    FN = int(((pred == 0) & (truth == 1)).sum())

    sens = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    spec = TN / (TN + FP) if (TN + FP) > 0 else np.nan
    ppv  = TP / (TP + FP) if (TP + FP) > 0 else np.nan
    npv  = TN / (TN + FN) if (TN + FN) > 0 else np.nan

    # AUC for a deterministic binary classifier = (sensitivity + specificity) / 2
    # Equivalent to the Wilcoxon-Mann-Whitney statistic
    auc = (sens + spec) / 2 if not (np.isnan(sens) or np.isnan(spec)) else np.nan

    # F1
    f1 = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else np.nan

    # Matthews Correlation Coefficient
    denom = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    mcc = (TP * TN - FP * FN) / denom if denom > 0 else np.nan

    # Binomial test vs H0: accuracy = 0.5
    binom = stats.binomtest(int(correct.sum()), n, 0.5, alternative="greater")
    p_binom = binom.pvalue

    # Chi-squared test of independence on confusion matrix
    ct = np.array([[TP, FN], [FP, TN]])
    try:
        chi2_stat, p_chi2, _, _ = chi2_contingency(ct, correction=False)
    except ValueError:
        chi2_stat, p_chi2 = np.nan, np.nan

    return dict(
        n=n, accuracy=acc,
        TP=TP, TN=TN, FP=FP, FN=FN,
        sensitivity=sens, specificity=spec,
        ppv=ppv, npv=npv,
        auc=auc, f1=f1, mcc=mcc,
        p_binomial=p_binom,
        chi2_stat=chi2_stat, p_chi2=p_chi2,
    )


def rolling_accuracy(pred, truth, window=15):
    """Compute rolling accuracy with a fixed-width trailing window."""
    pred  = np.asarray(pred,  dtype=int)
    truth = np.asarray(truth, dtype=int)
    out = []
    for i in range(len(pred)):
        s = max(0, i - window + 1)
        out.append((pred[s:i+1] == truth[s:i+1]).mean())
    return np.array(out)


def decadal_accuracy(years, pred, truth):
    """Return dict of {decade_start: (accuracy, n)} for each complete or partial decade."""
    years = np.asarray(years, dtype=int)
    pred  = np.asarray(pred,  dtype=int)
    truth = np.asarray(truth, dtype=int)
    out = {}
    for d in range(int(years.min() // 10) * 10, int(years.max()) + 1, 10):
        mask = (years >= d) & (years < d + 10)
        if mask.sum() == 0:
            continue
        acc = (pred[mask] == truth[mask]).mean()
        out[d] = (acc, int(mask.sum()))
    return out


def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_analysis(df):
    """Run all metrics for every definition. Returns list of result dicts."""
    phil = df[PHIL_COL].values.astype(int)
    results = []
    for name, col, color in DEFINITIONS:
        if col not in df.columns:
            print(f"  WARNING: column '{col}' not found, skipping '{name}'")
            continue
        truth = df[col].values.astype(int)
        m = compute_metrics(phil, truth)
        m["name"]  = name
        m["col"]   = col
        m["color"] = color
        m["pred"]  = phil
        m["truth"] = truth
        results.append(m)
    return results


def print_summary(results, years):
    n = len(years)
    yr_min, yr_max = int(min(years)), int(max(years))
    print(f"\n{'='*80}")
    print(f"  PUNXSUTAWNEY PHIL ACCURACY ANALYSIS")
    print(f"  N = {n} years  ({yr_min}–{yr_max})")
    print(f"{'='*80}")
    hdr = f"{'Definition':<28} {'N':>4} {'Acc':>7} {'AUC':>6} {'Sens':>7} {'Spec':>7} "
    hdr += f"{'PPV':>7} {'F1':>6} {'MCC':>6} {'p(binom)':>10}"
    print(hdr)
    print("-" * 95)
    for r in results:
        p_str = f"{r['p_binomial']:.5f}" if r['p_binomial'] >= 0.00001 else "<0.00001"
        print(
            f"{r['name']:<28} {r['n']:>4} {r['accuracy']:>7.1%} {r['auc']:>6.3f} "
            f"{r['sensitivity']:>7.1%} {r['specificity']:>7.1%} "
            f"{r['ppv']:>7.1%} {r['f1']:>6.3f} {r['mcc']:>6.3f} {p_str:>10}"
            f"  {sig_stars(r['p_binomial'])}"
        )
    print(f"\nPhil's shadow rate: {results[0]['pred'].mean():.1%}")


def save_summary_csv(results, outdir):
    path = os.path.join(outdir, "summary_table.csv")
    fields = ["name", "n", "accuracy", "auc", "sensitivity", "specificity",
              "ppv", "npv", "f1", "mcc", "TP", "TN", "FP", "FN",
              "p_binomial", "chi2_stat", "p_chi2"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            row = {k: (f"{r[k]:.4f}" if isinstance(r[k], float) else r[k]) for k in fields}
            w.writerow(row)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_roc(results, years, outdir):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="#AAAAAA", lw=1.5, label="Chance (AUC=0.50)", zorder=1)

    for r in results:
        fpr = 1 - r["specificity"]
        tpr = r["sensitivity"]
        mk  = "*" if r["name"] == "Composite" else "o"
        sz  = 280 if r["name"] == "Composite" else 190
        ax.scatter(fpr, tpr, s=sz, color=r["color"], marker=mk, zorder=5,
                   edgecolors="white", linewidths=1.2,
                   label=f"{r['name']}  (AUC={r['auc']:.3f})")

    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("False Positive Rate  (1 − Specificity)", fontsize=10)
    ax.set_ylabel("True Positive Rate  (Sensitivity)", fontsize=10)
    n = len(years)
    ax.set_title(f"ROC Operating Points — All Definitions\n(N={n}, {min(years)}–{max(years)})",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.92,
              title="Definition", title_fontsize=8)
    ax.set_aspect("equal")
    plt.tight_layout()
    path = os.path.join(outdir, "roc_plot.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_confusion_matrices(results, outdir):
    # Show first 4 definitions (exclude composite) in a 1×4 grid
    plot_results = [r for r in results if r["name"] != "Composite"][:4]
    fig, axes = plt.subplots(1, len(plot_results), figsize=(11, 3.2))
    if len(plot_results) == 1:
        axes = [axes]

    cell_colors = [[W_BLUE, ERR], [ERR, SPR]]
    cell_alpha  = [[0.75, 0.45], [0.45, 0.75]]
    cell_labels = [["TP", "FN"], ["FP", "TN"]]

    for ax, r in zip(axes, plot_results):
        cm = np.array([[r["TP"], r["FN"]], [r["FP"], r["TN"]]])
        for i in range(2):
            for j in range(2):
                c = mcolors.to_rgba(cell_colors[i][j], alpha=cell_alpha[i][j])
                ax.add_patch(plt.Rectangle((j, 1 - i), 1, 1, color=c))
                ax.text(j + 0.5, 1.5 - i, str(cm[i, j]),
                        ha="center", va="center", fontsize=18,
                        fontweight="bold", color="white")
                ax.text(j + 0.5, 1.5 - i - 0.28, cell_labels[i][j],
                        ha="center", va="center", fontsize=9,
                        color="white", alpha=0.9)
        ax.set_xlim(0, 2); ax.set_ylim(0, 2)
        ax.set_xticks([0.5, 1.5])
        ax.set_xticklabels(["Phil:\nWinter", "Phil:\nSpring"], fontsize=8)
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(["Actual:\nSpring", "Actual:\nWinter"], fontsize=8)
        ax.set_title(f"{r['name']}\nAUC={r['auc']:.3f}  Acc={r['accuracy']:.1%}",
                     fontsize=8.5, fontweight="bold")
        ax.tick_params(length=0); ax.grid(False)
        for spine in ax.spines.values(): spine.set_visible(False)

    plt.suptitle("Confusion Matrices — All Endpoint Definitions",
                 fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(outdir, "confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_auc_accuracy_bars(results, outdir):
    x = np.arange(len(results))
    colors = [r["color"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax_i, (key, title) in enumerate([("auc", "AUC-ROC"), ("accuracy", "Overall Accuracy")]):
        ax = axes[ax_i]
        vals = [r[key] for r in results]
        ax.bar(x, vals, color=colors, alpha=0.82, width=0.6,
               edgecolor="white", linewidth=1.2)
        ax.axhline(0.5, color="#AAAAAA", ls="--", lw=1.4)
        for xi, r in zip(x, results):
            star = sig_stars(r["p_binomial"])
            ax.text(xi, vals[xi] + 0.012, star, ha="center",
                    fontsize=8.5, fontweight="bold", color="#333")
            fmt = f"{vals[xi]:.3f}" if key == "auc" else f"{vals[xi]:.1%}"
            ax.text(xi, vals[xi] / 2, fmt, ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")
        labels = [r["name"].replace(" ≥", "\n≥") for r in results]
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5, rotation=15, ha="right")
        ax.set_ylim(0, 1.1)
        if key == "accuracy":
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
        ax.set_title(f"{title} by Definition", fontsize=10, fontweight="bold")
        ax.text(len(results) - 0.5, 0.52, "chance", color="#888", fontsize=8, ha="right")
    plt.tight_layout()
    path = os.path.join(outdir, "auc_accuracy_bars.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_rolling_accuracy(df, results, outdir, window=15):
    years = df[YEAR_COL].values.astype(int)
    plot_results = [r for r in results if r["name"] != "Composite"]

    fig, axes = plt.subplots(len(plot_results), 1,
                             figsize=(11, 2.4 * len(plot_results)), sharex=True)
    if len(plot_results) == 1:
        axes = [axes]

    for ax, r in zip(axes, plot_results):
        roll = rolling_accuracy(r["pred"], r["truth"], window=window)
        ax.axhline(0.5, color="#AAAAAA", ls="--", lw=1.2, zorder=1)
        ax.axhline(r["accuracy"], color=r["color"], ls="-", lw=1.5, alpha=0.4, zorder=1)
        ax.fill_between(years, roll, 0.5, where=roll >= 0.5,
                        color=SPR, alpha=0.25, interpolate=True)
        ax.fill_between(years, roll, 0.5, where=roll < 0.5,
                        color=ERR, alpha=0.25, interpolate=True)
        ax.plot(years, roll, color=r["color"], lw=2, zorder=3)
        # shade uncertain era
        ax.axvspan(years[0], 1939, color="#DDDDDD", alpha=0.25, zorder=0)
        ax.set_ylim(0.1, 0.98)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
        ax.set_ylabel(r["name"], fontsize=8, rotation=0, labelpad=90, va="center")
        for spine in ax.spines.values(): spine.set_visible(False)

    axes[-1].set_xlabel("Year", fontsize=10)
    axes[0].set_title(
        f"{window}-Year Rolling Accuracy  |  Gray = pre-1940 higher data uncertainty",
        fontsize=10, fontweight="bold")
    plt.tight_layout(); plt.subplots_adjust(hspace=0.35)
    path = os.path.join(outdir, "rolling_accuracy.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_decadal_heatmap(df, results, outdir):
    years = df[YEAR_COL].values.astype(int)
    plot_results = [r for r in results if r["name"] != "Composite"]
    decades = list(range(int(years.min() // 10) * 10, int(years.max()) + 1, 10))

    fig, axes = plt.subplots(1, len(plot_results), figsize=(11, 4))
    if len(plot_results) == 1:
        axes = [axes]

    for ax, r in zip(axes, plot_results):
        dec_acc, dec_n = [], []
        for d in decades:
            mask = (years >= d) & (years < d + 10)
            if mask.sum() == 0:
                dec_acc.append(np.nan); dec_n.append(0)
            else:
                dec_acc.append((r["pred"][mask] == r["truth"][mask]).mean())
                dec_n.append(int(mask.sum()))
        arr = np.array(dec_acc).reshape(-1, 1)
        im = ax.imshow(arr, vmin=0.3, vmax=1.0, cmap="RdYlGn", aspect="auto")
        ax.set_xticks([]); ax.set_yticks(range(len(decades)))
        ax.set_yticklabels([f"{d}s" for d in decades], fontsize=8)
        ax.set_title(r["name"] + f"\nAUC={r['auc']:.3f}", fontsize=8.5, fontweight="bold")
        for i, (acc, n) in enumerate(zip(dec_acc, dec_n)):
            if not np.isnan(acc) and n > 0:
                clr = "white" if acc < 0.55 else "#222"
                ax.text(0, i, f"{acc:.0%}\n(n={n})", ha="center", va="center",
                        fontsize=7.5, color=clr, fontweight="bold")
        if ax == axes[0]:
            ax.set_ylabel("Decade", fontsize=9)

    plt.colorbar(im, ax=axes[-1], label="Accuracy", fraction=0.08, pad=0.04, shrink=0.85)
    plt.suptitle("Decadal Accuracy Heatmap", fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(outdir, "decadal_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_bias_chart(df, results, outdir, window=20):
    years = df[YEAR_COL].values.astype(int)
    phil  = df[PHIL_COL].values.astype(int)

    shadow_roll = rolling_accuracy(phil, phil * 0 + 1, window)  # rate of shadow calls
    # Simpler: just compute rolling mean directly
    shadow_roll = np.array([phil[max(0,i-window+1):i+1].mean() for i in range(len(phil))])

    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(years, shadow_roll, color=BROWN, lw=2.5, label=f"Phil's shadow rate ({window}-yr rolling)")

    for r in results:
        if r["name"] == "Composite":
            continue
        truth_roll = np.array([r["truth"][max(0,i-window+1):i+1].mean() for i in range(len(r["truth"]))])
        ax.plot(years, truth_roll, color=r["color"], lw=1.5, ls="--",
                alpha=0.75, label=f"Actual winter rate — {r['name']}")

    ax.axhline(0.5, color="#CCCCCC", lw=1)
    ax.axvspan(years[0], 1939, color="#DDDDDD", alpha=0.3, zorder=0)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    ax.set_xlabel("Year", fontsize=10)
    ax.set_title(f"Phil's Shadow Rate vs. Actual Winter Rate ({window}-Year Rolling)\n"
                 "Gap between brown and colored lines = Phil's over-prediction bias",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    plt.tight_layout()
    path = os.path.join(outdir, "bias_chart.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_yearly_strip(df, results, outdir):
    years = df[YEAR_COL].values.astype(int)
    phil  = df[PHIL_COL].values.astype(int)
    N     = len(years)

    fig, axes = plt.subplots(len(results), 1, figsize=(14, 1.4 * len(results)), sharex=True)
    if len(results) == 1:
        axes = [axes]

    for ax, r in zip(axes, results):
        bar_colors = []
        for p, g in zip(phil, r["truth"]):
            if p == g == 1:   bar_colors.append(W_BLUE)
            elif p == g == 0: bar_colors.append(SPR)
            else:             bar_colors.append(ERR)
        ax.bar(years, [1] * N, color=bar_colors, width=0.85, alpha=0.82)
        ax.axvspan(years[0], 1939, color="#DDDDDD", alpha=0.25, zorder=0)
        ax.set_yticks([])
        ax.set_ylabel(r["name"], fontsize=7, rotation=0, labelpad=90, va="center")
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.grid(False)
        ax.text(max(years) + 0.8, 0.5, f"{r['accuracy']:.0%}",
                va="center", ha="left", fontsize=8, fontweight="bold", color="#333")

    axes[-1].set_xlabel("Year", fontsize=9)
    patches = [
        mpatches.Patch(color=W_BLUE, alpha=0.82, label="Correct — Winter"),
        mpatches.Patch(color=SPR,    alpha=0.82, label="Correct — Spring"),
        mpatches.Patch(color=ERR,    alpha=0.82, label="Incorrect"),
        mpatches.Patch(color="#DDDDDD", alpha=0.5, label="Pre-1940 (higher uncertainty)"),
    ]
    axes[0].legend(handles=patches, loc="upper left", fontsize=7.5,
                   framealpha=0.9, ncol=4, bbox_to_anchor=(0, 1.7))
    axes[0].set_title(
        f"Year-by-Year Outcomes  {min(years)}–{max(years)}  (N={N})  — accuracy at right",
        fontsize=10, fontweight="bold", pad=24)
    plt.tight_layout(); plt.subplots_adjust(hspace=0.2)
    path = os.path.join(outdir, "yearly_strip.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Punxsutawney Phil prediction accuracy analysis.")
    parser.add_argument("--csv",    default=None,           help="Path to input CSV file")
    parser.add_argument("--outdir", default="./phil_output", help="Output directory for plots/CSVs")
    parser.add_argument("--window", default=15, type=int,   help="Rolling accuracy window (default: 15)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load data
    df = load_data(args.csv)
    years = df[YEAR_COL].values.astype(int)

    # Run analysis
    print("\nRunning metrics...")
    results = run_analysis(df)
    print_summary(results, years)
    save_summary_csv(results, args.outdir)

    # Generate all plots
    print("\nGenerating plots...")
    plot_roc(results, years, args.outdir)
    plot_confusion_matrices(results, args.outdir)
    plot_auc_accuracy_bars(results, args.outdir)
    plot_rolling_accuracy(df, results, args.outdir, window=args.window)
    plot_decadal_heatmap(df, results, args.outdir)
    plot_bias_chart(df, results, args.outdir)
    plot_yearly_strip(df, results, args.outdir)

    print(f"\nDone. All outputs written to: {os.path.abspath(args.outdir)}/")


if __name__ == "__main__":
    main()
