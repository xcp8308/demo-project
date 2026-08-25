"""复现 DID 主回归：双向固定效应（TWFE）模型。

模型设定
    log_tfp ~ digital + capital_intensity + export_share
              + C(soe) + C(firm_id) + C(year)

    - 固定效应：企业（firm_id）+ 年份（year）双向固定效应
    - 标准误：按企业（firm_id）聚类稳健
    - 目标估计量：ATT（digital 系数）

输出
    output/python_did_results.csv（与 output/stata_did_results.csv 对照）

运行
    conda activate test
    python scripts/analyze_baseline.py
"""

from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "raw" / "digital_transformation_firm_panel.csv"
RESULTS_FILE = ROOT / "output" / "python_did_results.csv"
STATA_RESULTS_FILE = ROOT / "output" / "stata_did_results.csv"

# 用户指定且与 Stata 基准完全一致的主回归公式
BASELINE_FORMULA = (
    "log_tfp ~ digital + capital_intensity + export_share "
    "+ C(soe) + C(firm_id) + C(year)"
)


def load_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"缺失 {DATA_FILE}，请先运行 scripts/generate_synthetic_did.py 生成数据。"
        )
    data = pd.read_csv(DATA_FILE)
    if data.isna().any().any():
        raise ValueError("原始数据存在缺失值，请检查 data/raw/ 下的数据。")
    return data


def estimate_baseline(data: pd.DataFrame):
    """估计 TWFE 主回归，标准误按企业聚类。"""
    model = smf.ols(BASELINE_FORMULA, data=data)
    # 若出现少数单元（singular matrix）等运行时告警，仅忽略不影响结果的 warning
    result = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": data["firm_id"]},
    )
    return result


def save_results(result) -> None:
    """写入与 Stata 输出同构的结果表。"""
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    digital = result.params["digital"]
    se = result.bse["digital"]
    t = result.tvalues["digital"]
    p = result.pvalues["digital"]
    n_obs = int(result.nobs)

    table = pd.DataFrame(
        [
            {
                "term": "digital",
                "estimate": digital,
                "std_error": se,
                "t_value": t,
                "p_value": p,
                "n_obs": n_obs,
            }
        ]
    )
    table.to_csv(RESULTS_FILE, index=False)
    return table


def compare_with_stata(table: pd.DataFrame) -> str:
    """与 Stata 基准结果对比，返回校验摘要。"""
    if not STATA_RESULTS_FILE.exists():
        return "（未找到 output/stata_did_results.csv，跳过对照校验。）"
    stata = pd.read_csv(STATA_RESULTS_FILE)
    row = table.loc[0]
    stata_row = stata.loc[0]

    diff_est = abs(row["estimate"] - stata_row["estimate"])
    diff_se = abs(row["std_error"] - stata_row["std_error"])
    match_est = diff_est < 1e-4
    match_se = diff_se < 1e-4
    match_n = int(row["n_obs"]) == int(stata_row["n_obs"])

    lines = [
        "Stata vs Python 对照校验：",
        f"  digital 系数      : Python {row['estimate']:.8f} | Stata {stata_row['estimate']:.8f} | 一致: {match_est}",
        f"  聚类标准误        : Python {row['std_error']:.8f} | Stata {stata_row['std_error']:.8f} | 一致: {match_se}",
        f"  样本量 n_obs      : Python {row['n_obs']} | Stata {stata_row['n_obs']} | 一致: {match_n}",
        f"  全部一致: {all([match_est, match_se, match_n])}",
    ]
    return "\n".join(lines)


def main() -> None:
    data = load_data()
    result = estimate_baseline(data)
    table = save_results(result)
    print("主回归估计（digital）：")
    print(f"  系数        : {table.loc[0, 'estimate']:.8f}")
    print(f"  聚类标准误  : {table.loc[0, 'std_error']:.8f}")
    print(f"  t 值        : {table.loc[0, 't_value']:.4f}")
    print(f"  p 值        : {table.loc[0, 'p_value']:.4e}")
    print(f"  观测值数量  : {table.loc[0, 'n_obs']}")
    print(f"结果已写入 {RESULTS_FILE.relative_to(ROOT)}")
    print(compare_with_stata(table))


if __name__ == "__main__":
    main()
