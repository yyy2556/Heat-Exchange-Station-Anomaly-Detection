"""生成换热站供热季模拟传感器数据，并进行基础异常检测。"""

import argparse
from pathlib import Path

import matplotlib

# 兼容无图形界面的运行环境。
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest


RANDOM_SEED = 42
START_TIME = "2026-11-15 00:00:00"
PERIODS = 15 * 24 * 4
OUTPUT_DIR = Path(__file__).resolve().parent
DOCS_DIR = OUTPUT_DIR / "docs"
DEFAULT_REAL_DATA_PATH = OUTPUT_DIR / "data" / "raw" / "PodstanicaL8.csv"

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


def load_real_data(input_path: Path) -> pd.DataFrame:
    """读取 DHS 换热站 CSV，并转换为项目统一字段。"""
    raw = pd.read_csv(input_path, encoding="utf-8-sig")
    required_columns = {"datum", "tsp", "tns", "tps", "qizm"}
    missing_columns = required_columns.difference(raw.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"真实数据缺少必要列：{missing}")

    # 该数据集使用日/月/年格式，例如 2/10/2017 表示 2017 年 10 月 2 日。
    timestamp = pd.to_datetime(
        raw["datum"], format="mixed", dayfirst=True, errors="coerce"
    )
    if timestamp.isna().any():
        invalid_count = int(timestamp.isna().sum())
        raise ValueError(f"timestamp 有 {invalid_count} 行无法解析，请检查 datum 列格式")

    # DHS 数据集中的缩写含义：tsp 为室外温度，tns/tps 为二次侧供回水温度，
    # qizm 为传输热量/热功率（kW），不是水流量。tsr、tnp、tpp、e、dt 等
    # 原始列会保留在输出中供后续分析。
    data = raw.rename(
        columns={
            "tsp": "outside_temp",
            "tns": "supply_temp",
            "tps": "return_temp",
            "qizm": "heat_power",
        }
    ).copy()
    data["timestamp"] = timestamp

    # 真实数据没有阀门开度、室内温度和人工故障标签，明确标记为缺失/未标注。
    data["indoor_temp"] = np.nan
    data["valve_opening"] = np.nan
    data["fault_type"] = "未标注"

    numeric_columns = [
        "outside_temp",
        "supply_temp",
        "return_temp",
        "heat_power",
        "indoor_temp",
        "valve_opening",
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    # 仅对参与模型的真实传感器列进行时间序列插值，补齐少量缺失采样点。
    model_columns = ["outside_temp", "supply_temp", "return_temp", "heat_power"]
    data[model_columns] = data[model_columns].interpolate(limit_direction="both")
    data = data.dropna(subset=model_columns).sort_values("timestamp").reset_index(drop=True)
    return data


def save_visualizations(data: pd.DataFrame) -> None:
    """保存温度趋势图和带状态标记的关系散点图。"""
    DOCS_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    figure, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    data_kind = "真实数据" if "heat_power" in data.columns else "模拟数据"
    figure.suptitle(f"换热站{data_kind}温度变化", fontsize=16, fontweight="bold")
    axes[0].plot(data["timestamp"], data["supply_temp"], color="#d1495b", linewidth=1)
    axes[0].set_ylabel("供水温度 (°C)")
    axes[1].plot(data["timestamp"], data["return_temp"], color="#00798c", linewidth=1)
    axes[1].set_ylabel("回水温度 (°C)")
    axes[2].plot(data["timestamp"], data["outside_temp"], color="#30638e", linewidth=1)
    axes[2].set_ylabel("室外温度 (°C)")
    axes[2].set_xlabel("时间")
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(DOCS_DIR / "heat_exchange_temperature_trends.png", dpi=150)
    plt.close(figure)

    colors = {
        "正常": "#9aa0a6",
        "未标注": "#6c757d",
        "模型正常": "#9aa0a6",
        "模型异常": "#d1495b",
        "跑冒滴漏": "#d1495b",
        "水力失调": "#edae49",
    }
    if "flow_rate" in data.columns and data["flow_rate"].notna().any():
        x_column = "flow_rate"
        x_label = "流量 (t/h)"
    elif "heat_power" in data.columns and data["heat_power"].notna().any():
        x_column = "heat_power"
        x_label = "换热功率 (kW)"
    else:
        raise ValueError("散点图缺少可用的流量或换热功率列")

    figure, axis = plt.subplots(figsize=(11, 7))
    is_unlabeled = (
        "fault_type" in data.columns
        and data["fault_type"].eq("未标注").all()
        and "iforest_prediction" in data.columns
    )
    if is_unlabeled:
        plot_groups = [
            ("模型正常", data["iforest_prediction"] == 1),
            ("模型异常", data["iforest_prediction"] == -1),
        ]
    else:
        plot_groups = [
            (fault_type, data["fault_type"] == fault_type)
            for fault_type in data["fault_type"].drop_duplicates()
        ]

    for label, group_mask in plot_groups:
        group = data.loc[group_mask]
        if group.empty:
            continue
        axis.scatter(
            group[x_column],
            group["return_temp"],
            s=18 if label in {"正常", "模型正常"} else 42,
            alpha=0.55 if label in {"正常", "模型正常"} else 0.9,
            color=colors.get(label, "#6c757d"),
            label=label,
            edgecolors="none",
        )
    title_suffix = "（按孤立森林结果着色）" if is_unlabeled else "（按故障标签着色）"
    axis.set_title(f"{x_label}与回水温度关系{title_suffix}", fontsize=15, fontweight="bold")
    axis.set_xlabel(x_label)
    axis.set_ylabel("回水温度 (°C)")
    axis.legend(title="状态")
    figure.tight_layout()
    figure.savefig(DOCS_DIR / "heat_exchange_flow_return_scatter.png", dpi=150)
    plt.close(figure)


def save_anomaly_timeline(data: pd.DataFrame) -> None:
    """保存主运行变量、室外温度和孤立森林异常点的时间序列图。"""
    DOCS_DIR.mkdir(exist_ok=True)
    required_columns = {"timestamp", "outside_temp", "iforest_prediction"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        print(f"跳过异常时间序列图，缺少列：{missing}")
        return

    if "heat_power" in data.columns and data["heat_power"].notna().any():
        primary_column = "heat_power"
        primary_label = "换热功率 (kW)"
        primary_color = "#0077be"
    elif "flow_rate" in data.columns and data["flow_rate"].notna().any():
        primary_column = "flow_rate"
        primary_label = "流量 (t/h)"
        primary_color = "#0077be"
    else:
        primary_column = "supply_temp"
        primary_label = "供水温度 (°C)"
        primary_color = "#d1495b"

    anomaly_mask = data["iforest_prediction"].eq(-1)
    visible_anomaly_mask = anomaly_mask & data[primary_column].notna()
    figure, axis_primary = plt.subplots(figsize=(16, 6))
    axis_primary.plot(
        data["timestamp"],
        data[primary_column],
        color=primary_color,
        alpha=0.65,
        linewidth=0.8,
        label=primary_label,
    )
    axis_primary.set_xlabel("时间")
    axis_primary.set_ylabel(primary_label, color=primary_color)
    axis_primary.tick_params(axis="y", labelcolor=primary_color)

    if visible_anomaly_mask.any():
        axis_primary.scatter(
            data.loc[visible_anomaly_mask, "timestamp"],
            data.loc[visible_anomaly_mask, primary_column],
            color="#d1495b",
            s=16,
            alpha=0.8,
            label="孤立森林标记异常",
            zorder=5,
        )

    axis_secondary = axis_primary.twinx()
    axis_secondary.plot(
        data["timestamp"],
        data["outside_temp"],
        color="#30638e",
        alpha=0.65,
        linewidth=0.8,
        label="室外温度 (°C)",
    )
    axis_secondary.set_ylabel("室外温度 (°C)", color="#30638e")
    axis_secondary.tick_params(axis="y", labelcolor="#30638e")

    locator = mdates.AutoDateLocator()
    axis_primary.xaxis.set_major_locator(locator)
    axis_primary.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    timestamps = pd.to_datetime(data["timestamp"])
    start_time = timestamps.min().strftime("%Y-%m-%d")
    end_time = timestamps.max().strftime("%Y-%m-%d")
    axis_primary.set_title(
        f"换热站异常时间序列（{start_time} 至 {end_time}，标记 {int(anomaly_mask.sum())} 个异常点）",
        fontsize=14,
        fontweight="bold",
    )

    lines_primary, labels_primary = axis_primary.get_legend_handles_labels()
    lines_secondary, labels_secondary = axis_secondary.get_legend_handles_labels()
    axis_primary.legend(
        lines_primary + lines_secondary,
        labels_primary + labels_secondary,
        loc="upper left",
    )
    figure.tight_layout()
    figure.savefig(DOCS_DIR / "heat_exchange_anomaly_timeline.png", dpi=150)
    plt.close(figure)


def run_isolation_forest(data: pd.DataFrame) -> pd.DataFrame:
    """使用孤立森林检测异常并保存带预测结果的数据。"""
    feature_candidates = [
        "supply_temp",
        "return_temp",
        "flow_rate",
        "heat_power",
        "valve_opening",
        "outside_temp",
    ]
    feature_columns = [
        column
        for column in feature_candidates
        if column in data.columns and data[column].notna().all()
    ]
    if len(feature_columns) < 2:
        raise ValueError("可用于孤立森林的有效数值特征少于 2 个")

    print(f"\n孤立森林使用特征：{', '.join(feature_columns)}")
    model = IsolationForest(
        contamination=0.05,
        random_state=RANDOM_SEED,
        n_estimators=200,
    )
    data = data.copy()
    data["iforest_prediction"] = model.fit_predict(data[feature_columns])
    output_path = OUTPUT_DIR / "heat_exchange_data_with_iforest.csv"
    data.to_csv(output_path, index=False, encoding="utf-8-sig")

    injected_mask = data["fault_type"].isin(["跑冒滴漏", "水力失调"])
    predicted_anomaly_mask = data["iforest_prediction"] == -1
    anomaly_count = int(predicted_anomaly_mask.sum())
    injected_count = int(injected_mask.sum())

    print("孤立森林检测汇总")
    print(f"模型标记异常点数量：{anomaly_count}")
    if injected_count:
        recalled_count = int((injected_mask & predicted_anomaly_mask).sum())
        recall = recalled_count / injected_count
        print(f"注入异常点数量：{injected_count}")
        print(f"召回的注入异常点数量：{recalled_count}")
        print(f"简单召回率：{recall:.2%}")
    else:
        print("真实数据没有人工故障标签，跳过召回率计算")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="换热站模拟数据生成与异常检测")
    parser.add_argument(
        "--source",
        choices=["auto", "real", "simulated"],
        default="auto",
        help="数据来源：auto 优先读取真实 CSV，找不到时使用模拟数据",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_REAL_DATA_PATH,
        help="真实 CSV 路径，默认是 data/raw/PodstanicaL8.csv",
    )
    args = parser.parse_args()

    use_real_data = args.source == "real" or (
        args.source == "auto" and args.input.exists()
    )
    if use_real_data:
        data = load_real_data(args.input)
        print(f"读取真实数据：{args.input}")
    else:
        data = generate_data()
        print("未找到真实 CSV，使用模拟数据")

    raw_path = OUTPUT_DIR / "heat_exchange_data.csv"
    data.to_csv(raw_path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d %H:%M:%S")
    data = run_isolation_forest(data)
    save_visualizations(data)
    save_anomaly_timeline(data)
    print(f"\n已生成 {len(data)} 行数据，输出目录：{OUTPUT_DIR}")
    print(f"原始数据：{raw_path.name}")
    print("检测结果：heat_exchange_data_with_iforest.csv")
    print("图片：docs/heat_exchange_temperature_trends.png")
    print("图片：docs/heat_exchange_flow_return_scatter.png")
    print("图片：docs/heat_exchange_anomaly_timeline.png")


if __name__ == "__main__":
    main()
