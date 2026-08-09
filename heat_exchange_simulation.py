"""生成换热站供热季模拟传感器数据，并进行基础异常检测。"""

from pathlib import Path

import matplotlib

# 兼容无图形界面的运行环境。
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest


RANDOM_SEED = 42
START_TIME = "2026-11-15 00:00:00"
PERIODS = 15 * 24 * 4
OUTPUT_DIR = Path(__file__).resolve().parent

# 优先使用常见中文字体，避免图表标题、坐标轴和图例出现缺字方框。
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans SC",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def generate_data() -> pd.DataFrame:
    """生成 15 天、15 分钟粒度的换热站传感器数据。"""
    rng = np.random.default_rng(RANDOM_SEED)
    timestamps = pd.date_range(START_TIME, periods=PERIODS, freq="15min")
    elapsed_hours = np.arange(PERIODS) / 4
    hour_of_day = (elapsed_hours % 24).astype(float)

    # 室外温度约在凌晨 2 点最低、下午 2 点最高，并叠加平滑扰动和高斯噪声。
    daily_cycle = -5 + 9 * np.sin(2 * np.pi * (hour_of_day - 8) / 24)
    weather_variation = 1.2 * np.sin(2 * np.pi * elapsed_hours / (24 * 5) + 0.8)
    outside_temp = np.clip(
        daily_cycle + weather_variation + rng.normal(0, 0.55, PERIODS), -15, 5
    )

    # 室外越冷，供水温度越高。
    supply_temp = np.clip(
        58 - 0.8 * outside_temp + rng.normal(0, 0.7, PERIODS), 45, 75
    )

    # 使用上一时刻供水温度构造 15 分钟滞后，体现回水温度的延迟响应。
    lagged_supply = pd.Series(supply_temp).shift(1).bfill().to_numpy()
    return_temp = np.clip(
        lagged_supply - 10.5 + rng.normal(0, 0.5, PERIODS), 35, 55
    )

    valve_opening = np.clip(
        48 + 1.8 * (supply_temp - 55) + rng.normal(0, 2.2, PERIODS), 30, 100
    )
    flow_rate = np.clip(
        96 + 1.35 * (supply_temp - 55) + 0.18 * (valve_opening - 60)
        + rng.normal(0, 3.5, PERIODS),
        80,
        150,
    )

    # 采用简单的一阶惯性模拟室内温度，不让室内温度随瞬时噪声剧烈跳变。
    indoor_temp = np.empty(PERIODS)
    indoor_temp[0] = 21.0
    for index in range(1, PERIODS):
        target = 21 + 0.08 * (supply_temp[index] - 60)
        indoor_temp[index] = (
            0.94 * indoor_temp[index - 1]
            + 0.06 * target
            + rng.normal(0, 0.035)
        )
    indoor_temp = np.clip(indoor_temp, 18, 24)

    data = pd.DataFrame(
        {
            "timestamp": timestamps,
            "outside_temp": outside_temp,
            "supply_temp": supply_temp,
            "return_temp": return_temp,
            "flow_rate": flow_rate,
            "indoor_temp": indoor_temp,
            "valve_opening": valve_opening,
            "fault_type": "正常",
        }
    )

    # 两个异常时段均采用左闭右开区间，共 16 行（4 小时）。
    leak_mask = (data["timestamp"] >= "2026-11-17 14:00:00") & (
        data["timestamp"] < "2026-11-17 18:00:00"
    )
    imbalance_mask = (data["timestamp"] >= "2026-11-22 08:00:00") & (
        data["timestamp"] < "2026-11-22 12:00:00"
    )

    # 跑冒滴漏：流量上升、回水温度下降、供水温度保持原有状态。
    data.loc[leak_mask, "flow_rate"] = np.clip(
        data.loc[leak_mask, "flow_rate"] * 1.4, 80, 150
    )
    data.loc[leak_mask, "return_temp"] = np.clip(
        data.loc[leak_mask, "return_temp"] - 6.5, 25, 55
    )
    data.loc[leak_mask, "fault_type"] = "跑冒滴漏"

    # 水力失调：回水温度显著降低，阀门开度升高，流量降低。
    data.loc[imbalance_mask, "return_temp"] = np.clip(
        data.loc[imbalance_mask, "return_temp"] - 18, 20, 55
    )
    data.loc[imbalance_mask, "valve_opening"] = np.clip(
        data.loc[imbalance_mask, "valve_opening"] + 22, 30, 100
    )
    data.loc[imbalance_mask, "flow_rate"] = np.clip(
        data.loc[imbalance_mask, "flow_rate"] * 0.58, 20, 150
    )
    data.loc[imbalance_mask, "fault_type"] = "水力失调"

    return data


def save_visualizations(data: pd.DataFrame) -> None:
    """保存温度趋势图和流量-回水温度散点图。"""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    figure, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    figure.suptitle("换热站供热季 15 天温度变化", fontsize=16, fontweight="bold")
    axes[0].plot(data["timestamp"], data["supply_temp"], color="#d1495b", linewidth=1)
    axes[0].set_ylabel("供水温度 (°C)")
    axes[1].plot(data["timestamp"], data["return_temp"], color="#00798c", linewidth=1)
    axes[1].set_ylabel("回水温度 (°C)")
    axes[2].plot(data["timestamp"], data["outside_temp"], color="#30638e", linewidth=1)
    axes[2].set_ylabel("室外温度 (°C)")
    axes[2].set_xlabel("时间")
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "heat_exchange_temperature_trends.png", dpi=150)
    plt.close(figure)

    colors = {"正常": "#9aa0a6", "跑冒滴漏": "#d1495b", "水力失调": "#edae49"}
    figure, axis = plt.subplots(figsize=(11, 7))
    for fault_type, group in data.groupby("fault_type", sort=False):
        axis.scatter(
            group["flow_rate"],
            group["return_temp"],
            s=18 if fault_type == "正常" else 42,
            alpha=0.55 if fault_type == "正常" else 0.9,
            color=colors[fault_type],
            label=fault_type,
            edgecolors="none",
        )
    axis.set_title("流量与回水温度关系（异常时段高亮）", fontsize=15, fontweight="bold")
    axis.set_xlabel("流量 (t/h)")
    axis.set_ylabel("回水温度 (°C)")
    axis.legend(title="状态")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "heat_exchange_flow_return_scatter.png", dpi=150)
    plt.close(figure)


def run_isolation_forest(data: pd.DataFrame) -> pd.DataFrame:
    """使用孤立森林检测异常并保存带预测结果的数据。"""
    feature_columns = [
        "supply_temp",
        "return_temp",
        "flow_rate",
        "valve_opening",
        "outside_temp",
    ]
    model = IsolationForest(
        contamination=0.05,
        random_state=RANDOM_SEED,
        n_estimators=200,
    )
    data = data.copy()
    data["iforest_prediction"] = model.fit_predict(data[feature_columns])
    output_path = OUTPUT_DIR / "heat_exchange_data_with_iforest.csv"
    data.to_csv(output_path, index=False, encoding="utf-8-sig")

    injected_mask = data["fault_type"] != "正常"
    predicted_anomaly_mask = data["iforest_prediction"] == -1
    anomaly_count = int(predicted_anomaly_mask.sum())
    recalled_count = int((injected_mask & predicted_anomaly_mask).sum())
    injected_count = int(injected_mask.sum())
    recall = recalled_count / injected_count if injected_count else 0.0

    print("\n孤立森林检测汇总")
    print(f"模型标记异常点数量：{anomaly_count}")
    print(f"注入异常点数量：{injected_count}")
    print(f"召回的注入异常点数量：{recalled_count}")
    print(f"简单召回率：{recall:.2%}")
    return data


def main() -> None:
    data = generate_data()
    raw_path = OUTPUT_DIR / "heat_exchange_data.csv"
    data.to_csv(raw_path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d %H:%M:%S")
    save_visualizations(data)
    run_isolation_forest(data)
    print(f"\n已生成 {len(data)} 行数据，输出目录：{OUTPUT_DIR}")
    print(f"原始数据：{raw_path.name}")
    print("检测结果：heat_exchange_data_with_iforest.csv")
    print("图片：heat_exchange_temperature_trends.png")
    print("图片：heat_exchange_flow_return_scatter.png")


if __name__ == "__main__":
    main()
