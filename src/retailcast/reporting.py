import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def markdown_table(frame, percent_columns=()):
    f = frame.copy()
    for c in f.select_dtypes(include="number"):
        f[c] = f[c].map(lambda v: f"{v:.1%}" if c in percent_columns else f"{v:,.3f}")
    lines = ["| " + " | ".join(f.columns) + " |", "| " + " | ".join(["---"]*len(f.columns)) + " |"]
    return "\n".join(lines + ["| " + " | ".join(map(str, r)) + " |" for r in f.itertuples(index=False, name=None)])


def build_reports(root, panel, summary, cv, holdout, future):
    root = Path(root)
    report = root / "reports"
    figures = report / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.labelcolor": "#334155", "text.color": "#142033",
                         "axes.edgecolor": "#dce4ed", "grid.color": "#e5eaf0", "figure.facecolor": "#f7f9fc",
                         "axes.facecolor": "white", "savefig.facecolor": "#f7f9fc"})
    champion = holdout.loc[holdout.model == "selected_champion"]
    leaderboard = pd.read_csv(report / "validation_leaderboard.csv")
    fig, ax = plt.subplots(figsize=(10, 4.6), layout="constrained")
    ordered = leaderboard.sort_values("MASE", ascending=False)
    ax.barh(ordered.model, ordered.MASE, color=["#12a58c" if str(m).startswith("gbm") else "#466586" for m in ordered.model])
    ax.set_xlabel("Mean validation MASE across markets / folds (lower is better)")
    ax.set_title("Four 28-day windows. Model selection ends before calibration.", loc="left", fontweight="bold", pad=16)
    ax.grid(axis="x", alpha=.5)
    ax.set_axisbelow(True)
    fig.savefig(figures / "validation_leaderboard.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(len(panel.columns), 1, figsize=(12, 3.1*len(panel.columns)), layout="constrained")
    for ax, market in zip(np.atleast_1d(axes), panel.columns):
        g = champion.loc[champion.market == market]
        context = panel.loc[(panel.index >= g.date.min() - pd.Timedelta(days=28)) & (panel.index <= g.date.max()), market]
        ax.plot(context.index, context, color="#263e59", lw=1.4, label="Observed sales")
        ax.fill_between(g.date, g.lower_95, g.upper_95, color="#12a58c", alpha=.12, label="95% interval")
        ax.fill_between(g.date, g.lower_80, g.upper_80, color="#12a58c", alpha=.23, label="80% interval")
        ax.plot(g.date, g.prediction, color="#009b83", lw=2, label="Locked champion")
        naive = holdout.loc[(holdout.market == market) & (holdout.model == "seasonal_naive")]
        ax.plot(naive.date, naive.prediction, color="#dc9c41", lw=1.2, ls="--", label="Seasonal naive")
        ax.axvline(g.date.min(), color="#a0aab8", ls=":")
        ax.set_title(f"{market}  |  {summary['selection'][market]}", loc="left", fontweight="bold")
        ax.set_ylabel("Units / day")
        ax.grid(axis="y", alpha=.45)
    axes[0].legend(ncol=5, fontsize=8, frameon=False, loc="upper left")
    fig.savefig(figures / "holdout_forecasts.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), layout="constrained")
    first = panel.columns[0]
    axes[0].plot(panel.index, panel[first], color="#adc1d1", lw=.8, label="Daily")
    axes[0].plot(panel.index, panel[first].rolling(28).mean(), color="#009b83", lw=2, label="Trailing 28-day mean")
    axes[0].set_title(f"Observed sales: {first}", loc="left", fontweight="bold")
    for date, label in [(summary["holdout_start"], "Holdout")]:
        axes[0].axvline(pd.Timestamp(date), ls="--", color="#d99736", label=label)
    axes[0].legend(frameon=False)
    # EDA seasonal profile is restricted to the initial training period.
    train_end = pd.read_csv(report / "temporal_splits.csv").iloc[0].train_end
    early = panel.loc[:train_end]
    normalized = early.div(early.mean()).groupby(early.index.dayofweek).mean()
    for market in panel.columns:
        axes[1].plot(range(7), normalized[market], marker="o", label=market)
    axes[1].set_xticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    axes[1].set_ylabel("Sales / market daily mean")
    axes[1].set_title("Day-of-week profile, initial training data only", loc="left", fontweight="bold")
    axes[1].legend(frameon=False, ncol=3)
    fig.savefig(figures / "data_overview.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), layout="constrained")
    market_errors = champion.groupby("market").scaled_abs_error.mean().sort_values()
    axes[0].barh(market_errors.index, market_errors, color="#466586")
    axes[0].set_xlabel("Holdout MASE")
    axes[0].set_title("Error differs substantially by market", loc="left", fontweight="bold")
    interval = pd.read_csv(report / "interval_metrics.csv")
    xx = np.arange(len(interval))
    axes[1].bar(xx-.18, interval.coverage_80, width=.36, color="#12a58c", label="80% interval")
    axes[1].bar(xx+.18, interval.coverage_95, width=.36, color="#466586", label="95% interval")
    axes[1].axhline(.8, color="#12a58c", ls="--", lw=1)
    axes[1].axhline(.95, color="#466586", ls="--", lw=1)
    axes[1].set_xticks(xx, interval.market, rotation=25, ha="right")
    axes[1].set_ylim(0, 1.07)
    axes[1].set_title("Empirical coverage: 28 test days per market", loc="left", fontweight="bold")
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(figures / "error_diagnostics.png", dpi=180)
    plt.close(fig)

    scores = pd.read_csv(report / "holdout_metrics.csv").set_index("model")
    change = 1 - scores.loc["selected_champion", "WAPE"] / scores.loc["seasonal_naive", "WAPE"]
    market_scores = pd.read_csv(report / "holdout_by_market.csv")
    interval_scores = pd.read_csv(report / "interval_metrics.csv")
    provenance = json.loads((root / "data/processed/provenance.json").read_text())
    winners = market_scores.pivot(index="market", columns="model", values="WAPE")
    worse_naive = winners.index[winners.selected_champion > winners.seasonal_naive].tolist()
    worse_smooth = winners.index[winners.selected_champion > winners.weekday_mean_8w].tolist()
    difficult = winners.index[winners.selected_champion > 1].tolist()
    text = f"""# RetailCast — báo cáo kết quả thực nghiệm

## 1. Bài toán và phạm vi
Dự báo tổng số đơn vị hàng hóa bán ra mỗi ngày cho 5 thị trường, horizon 28 ngày.
Kết quả hỗ trợ thảo luận lập kế hoạch tải vận hành; không tự động chuyển thành mức đặt hàng tồn kho.
Nguồn: [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii), DOI 10.24432/C5CG6D, CC BY 4.0.
Dữ liệu lịch sử {summary['history_start']} → {summary['history_end']}; {provenance['raw_rows']:,} dòng gốc.
Đây là bài thực nghiệm trên dữ liệu lịch sử, không phải dự báo bán hàng hiện tại.

## 2. Thiết kế đánh giá
- 4 fold validation, mỗi fold dự báo cố định 28 ngày; huấn luyện mở rộng theo thời gian.
- Chọn thuật toán riêng cho từng thị trường bằng MASE trung bình trên 4 fold.
- 2 fold tiếp theo dùng riêng để hiệu chỉnh khoảng dự báo (56 sai số / thị trường).
- Holdout: {summary['holdout_start']} → {summary['holdout_end']}. Khóa lựa chọn trước khi chấm điểm.
- Nhãn huấn luyện phải có target_date trước cutoff; toàn bộ lag và rolling chỉ lấy dữ liệu đã biết tại origin.
- Không cập nhật bằng actual trong 28 ngày đang dự báo. Không dùng random split hoặc early stopping ngẫu nhiên.

## 3. Kết quả holdout
WAPE của bộ champion: **{scores.loc['selected_champion', 'WAPE']:.1%}**; seasonal naive: **{scores.loc['seasonal_naive', 'WAPE']:.1%}**.
Thay đổi tương đối WAPE so với seasonal naive: **{change:+.1%}** (số dương là giảm lỗi).
Đây là kết quả một cửa sổ 28 ngày; không chứng minh cải thiện ổn định qua mọi mùa.

{markdown_table(scores.reset_index(), ['WAPE', 'bias_pct'])}

WAPE cộng gộp bị chi phối bởi thị trường lớn. MASE trung bình cân bằng số quan sát giữa thị trường.
MASE dùng thang đo seasonal naive trong tập train; MASE < 1 không tự động chứng minh thắng seasonal naive trên test.

### Theo thị trường
{markdown_table(market_scores, ['WAPE', 'bias_pct'])}

Champion thua seasonal naive tại: **{', '.join(worse_naive) or 'không có'}**.
Champion thua baseline trung bình theo thứ tại: **{', '.join(worse_smooth) or 'không có'}**.
Các thị trường có WAPE champion vượt 100%: **{', '.join(difficult) or 'không có'}**.
Không diễn giải chỉ số tổng hợp tốt thành chất lượng tốt ở mọi thị trường. Đặc biệt, denominator MASE lớn do đơn đột biến trong train có thể làm MASE thấp dù dự báo trên một thị trường bán thưa vẫn yếu.
Lựa chọn thuật toán giữ nguyên sau khi thấy các kết quả này.

### Khoảng dự báo
{markdown_table(interval_scores, ['coverage_80', 'coverage_95'])}

Khoảng dùng sai số tuyệt đối và hiệu chỉnh thứ hạng ceil((n+1) × coverage), cắt cận dưới tại 0.
Không gọi đây là bảo đảm độ phủ: sai số có phụ thuộc thời gian, quy mô bán có thể thay đổi, và holdout gần mùa Giáng sinh.
Khoảng được gộp qua 28 horizon cho từng thị trường; không có bảo đảm riêng cho từng horizon hoặc tổng 28 ngày.
Không cộng các cận ngày để tuyên bố khoảng tin cậy cho tổng bán 28 ngày.

## 4. Kiểm tra chất lượng và giới hạn
- Loại hủy đơn, số lượng/giá không dương, mã không phải hàng hóa; giữ đơn lớn để không làm đẹp dữ liệu.
- Không bỏ giao dịch chỉ vì thiếu Customer ID: biến này không cần cho target tổng bán.
- Loại bản ghi trùng hoàn toàn theo giả định lỗi ghi nhận; không có line ID để xác nhận mọi trường hợp.
- Bỏ ngày cuối vì ghi nhận chưa trọn ngày. Ngày không có giao dịch hợp lệ được gán 0, có thể lẫn với mất dữ liệu.
- Thị trường được chọn bằng dữ liệu trước validation đầu tiên, tránh chọn theo kết quả tương lai.
- Chưa có giá/khuyến mại/tồn kho/lead time; không suy luận nhu cầu bị mất do hết hàng hoặc tác động nhân quả.
- Ít hơn 3 năm dữ liệu và chỉ 1 holdout: hạn chế đánh giá mùa vụ năm, cold start, structural breaks.
- Nhiều khách là wholesaler; đơn lớn tạo sai số đột biến. Các lỗi lớn được giữ trong largest_errors.csv.

## 5. Hướng nâng cấp hợp lý
Thêm tồn kho, lịch khuyến mại và lịch nghỉ theo thị trường; kiểm tra nhiều mùa holdout theo protocol mới đã định trước;
so sánh mô hình phân cấp SKU → thị trường nếu có master data; dùng block calibration hoặc rolling quantile với nhiều lịch sử hơn.
Chỉ đánh giá chính sách tồn kho khi có lead time và chi phí thiếu/dư hàng thực.

## 6. Tái lập
Xem README.md. File run_summary.json ghi cấu hình, phiên bản thư viện, SHA-256 dữ liệu và thời gian chạy.
Kết quả có thể lệch nhỏ giữa phiên bản BLAS/nền tảng. Chạy tests trước khi dùng để trình bày.

![Validation](figures/validation_leaderboard.png)
![Holdout](figures/holdout_forecasts.png)
![Diagnostics](figures/error_diagnostics.png)
"""
    (report / "BAO_CAO_KET_QUA.md").write_text(text, encoding="utf-8")
    payload = {
        "summary": summary,
        "history": [{"date": str(d.date()), **{m: float(panel.loc[d, m]) for m in panel.columns}} for d in panel.index],
        "holdout": json.loads(holdout.to_json(orient="records", date_format="iso")),
        "future": json.loads(future.to_json(orient="records", date_format="iso")),
        "market_scores": market_scores.to_dict(orient="records"),
        "validation": pd.read_csv(report / "validation_by_market.csv").to_dict(orient="records"),
        "interval_scores": interval_scores.to_dict(orient="records"),
    }
    template = (root / "src/retailcast/dashboard_template.html").read_text(encoding="utf-8")
    (root / "DEMO.html").write_text(template.replace("__PAYLOAD__", json.dumps(payload, allow_nan=False)), encoding="utf-8")
