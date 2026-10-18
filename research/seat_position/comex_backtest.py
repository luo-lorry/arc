#!/usr/bin/env python3
"""Trade COMEX silver and gold with the Shanghai silver event-day signal.

The seat and warrant signal is known at 15:00 Beijing, which is 03:00 or
02:00 New York time, before the 08:30 ET release. The fill used here is the
08:00 ET hourly open. One exit is that hour's close, about 09:00 ET. The
other is the 13:00 ET hour's close, about 14:00 ET.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from audit_event_rules import binom_p, bp, pct, ptxt, summarize, wilson
from trade_strategy import agreement_book, build, curve_stats, event_calls, money, num

ROOT = Path(__file__).resolve().parent
BARS = ROOT / "data" / "comex_event_bars.json"
OUT = ROOT / "product" / "COMEX_BACKTEST.md"
NY = ZoneInfo("America/New_York")
UA = {"User-Agent": "Mozilla/5.0"}
SPECS = {
    "SI": {"symbol": "SI=F", "name": "COMEX白银", "oz": 5000, "tick": 0.005},
    "GC": {"symbol": "GC=F", "name": "COMEX黄金", "oz": 100, "tick": 0.10},
}
# One tick each side, plus 2.50 dollars a side for a non-member commission.
COMMISSION = 5.0


def fetch_hourly(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    frames = []
    cursor = start
    while cursor < end:
        stop = min(cursor + timedelta(days=60), end)
        query = urllib.parse.urlencode(
            {
                "period1": int(cursor.timestamp()),
                "period2": int(stop.timestamp()),
                "interval": "60m",
                "includePrePost": "true",
            }
        )
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(symbol)}?{query}"
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode())
        result = (payload.get("chart") or {}).get("result") or [None]
        block = result[0]
        if block and block.get("timestamp"):
            quote = block["indicators"]["quote"][0]
            frames.append(
                pd.DataFrame(
                    {
                        "utc": pd.to_datetime(block["timestamp"], unit="s", utc=True),
                        "open": quote["open"],
                        "close": quote["close"],
                    }
                )
            )
        cursor = stop
    if not frames:
        raise RuntimeError(f"no hourly bars for {symbol}")
    out = pd.concat(frames, ignore_index=True).dropna(subset=["open", "close"])
    out = out.drop_duplicates("utc").sort_values("utc")
    local = out["utc"].dt.tz_convert(NY)
    out["ny_date"] = local.dt.date
    out["ny_hour"] = local.dt.hour
    return out


def event_prices(bars: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for day, chunk in bars.groupby("ny_date", sort=False):
        by_hour = {}
        for hour, hour_rows in chunk.groupby("ny_hour"):
            by_hour[int(hour)] = hour_rows.iloc[0]
        entry = by_hour.get(8)
        settle = by_hour.get(13)
        prior = by_hour.get(7)
        if entry is None or pd.isna(entry["open"]) or pd.isna(entry["close"]):
            continue
        gap = None
        if prior is not None and pd.notna(prior["close"]) and float(prior["close"]) != 0:
            gap = float(entry["open"]) / float(prior["close"]) - 1.0
        rows.append(
            {
                "day": day,
                "entry": float(entry["open"]),
                "exit_0900": float(entry["close"]),
                "exit_1400": None if settle is None or pd.isna(settle["close"]) else float(settle["close"]),
                "gap_into_0800": gap,
            }
        )
    return pd.DataFrame(rows)


def load_prices() -> dict[str, pd.DataFrame]:
    if BARS.exists():
        saved = json.loads(BARS.read_text())
        return {key: pd.DataFrame(rows) for key, rows in saved.items()}
    start = datetime(2024, 12, 15, tzinfo=timezone.utc)
    end = datetime(2026, 10, 18, tzinfo=timezone.utc)
    saved = {}
    frames = {}
    for key, spec in SPECS.items():
        hourly = fetch_hourly(spec["symbol"], start, end)
        prices = event_prices(hourly)
        prices["day"] = prices["day"].map(lambda value: value.isoformat())
        frames[key] = prices
        saved[key] = prices.to_dict(orient="records")
    BARS.parent.mkdir(parents=True, exist_ok=True)
    BARS.write_text(json.dumps(saved), encoding="utf-8")
    return frames


def signal_tables(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    warrant = event_calls(frame, "ag", "warrant_sign")[["day", "pred", "label", "w_chg5", "gs_all_chg5"]].copy()
    goldman = event_calls(frame, "ag", "gs_chg5")[["day", "pred", "label", "w_chg5", "gs_all_chg5"]].copy()
    either = agreement_book(frame, "either")
    keep = [column for column in ("day", "pred", "label", "w_chg5", "gs_all_chg5") if column in either.columns]
    return {"仓单单独": warrant, "高盛五日单独": goldman, "冲突则空仓": either[keep].copy()}


def join_book(signal: pd.DataFrame, prices: pd.DataFrame, market: str) -> pd.DataFrame:
    prices = prices.copy()
    prices["day"] = prices["day"].map(lambda value: datetime.fromisoformat(str(value)).date())
    merged = signal.merge(prices, on="day", how="left")
    spec = SPECS[market]
    merged["market"] = market
    merged["splice"] = merged["gap_into_0800"].abs() > 0.03
    for column, ret_name in (("exit_0900", "ret_0900"), ("exit_1400", "ret_1400")):
        merged[ret_name] = merged[column] / merged["entry"] - 1.0
        gross = merged["pred"] * (merged[column] - merged["entry"]) * spec["oz"]
        cost = spec["tick"] * spec["oz"] * 2 + COMMISSION
        merged[f"gross_{ret_name[-4:]}"] = gross
        merged[f"net_{ret_name[-4:]}"] = gross - cost
    return merged


def usable(rows: pd.DataFrame, ret: str) -> pd.DataFrame:
    out = rows[rows[ret].notna() & rows[ret].ne(0) & ~rows["splice"].fillna(False)].copy()
    out["ret_open"] = out[ret]
    out["date"] = pd.to_datetime(out["day"].map(lambda value: value.isoformat()))
    return out


def main() -> None:
    prices = load_prices()
    frame = build()
    signals = signal_tables(frame)
    lines = [
        "# COMEX 上的发布日策略",
        "",
        "信号和上期所那一手沪银是同一条。15:00 北京时间的会员排名和仓单日报，换算到纽约是凌晨 2 点或 3 点，早于当天 8:30 ET 的 CPI、非农、PPI、零售、GDP、PCE、ECI。成交改到 COMEX。",
        "",
        "进场价是美东 8:00 那根小时线的开盘价，这时数据还没公布。短持仓的出场价是这根小时线的收盘价，大约 9:00 ET，公布后约 30 分钟。长持仓的出场价是美东 13:00 那根小时线的收盘价，大约 14:00 ET，过了 13:30 的期货结算。一天只做一次。两条信号方向相反就空仓。8:00 开盘相对 7:00 收盘跳过 3% 的日子视为连续合约换月，不计入盈亏。",
        "",
        "行情来自 Yahoo Finance 的连续主力 `SI=F` 和 `GC=F`。白银一手 5,000 盎司，最小变动 0.005 美元/盎司，一跳 25 美元。黄金一手 100 盎司，最小变动 0.10 美元/盎司，一跳 10 美元。成本按买卖各让一跳，再加双边 5 美元佣金。这是非会员的一个整数假设，不是某一家期货公司的报价。",
        "",
        "事先规定的做法是：把沪银那一手原样搬到 COMEX 白银，冲突则空仓，8:00 进、9:00 出。COMEX 黄金用同一条沪银信号，是平行试验，不另选方向。",
        "",
    ]
    books = {}
    for market in ("SI", "GC"):
        for name, signal in signals.items():
            books[(market, name)] = join_book(signal, prices[market], market)

    lines.extend([
        "## 回测",
        "",
        "| 市场 | 做法 | 出场 | 手数 | 对 | 命中 | 95%下限 | 前半段 | 后半段 | 方向收益 | 毛利美元 | 成本后 | 最大回撤 | 最差一手 | 双侧随机概率 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ])

    def add_row(market: str, name: str, ret: str, label: str) -> None:
        rows = usable(books[(market, name)], ret)
        stats = summarize(rows, "ret_open") if len(rows) else {"n": 0, "hits": 0, "rate": None, "lo": None, "early": None, "late": None, "edge": None}
        suffix = ret[-4:]
        gross = curve_stats(rows[f"gross_{suffix}"]) if len(rows) else curve_stats(pd.Series(dtype=float))
        net = curve_stats(rows[f"net_{suffix}"]) if len(rows) else curve_stats(pd.Series(dtype=float))
        lines.append(
            f"| {SPECS[market]['name']} | {name} | {label} | {stats['n']} | {stats['hits']} | {pct(stats['rate'])} | {pct(stats['lo'])} | "
            f"{pct(stats['early'])} | {pct(stats['late'])} | {bp(stats['edge'])} | {money(gross['total'])} | {money(net['total'])} | "
            f"{money(net['dd'])} | {money(net['worst'])} | {ptxt(binom_p(stats['hits'], stats['n']))} |"
        )

    for market in ("SI", "GC"):
        for name in signals:
            add_row(market, name, "ret_0900", "9:00")
            add_row(market, name, "ret_1400", "14:00")

    chosen = usable(books[("SI", "冲突则空仓")], "ret_0900")
    chosen_long = usable(books[("SI", "冲突则空仓")], "ret_1400")
    stats = summarize(chosen, "ret_open") if len(chosen) else {"n": 0, "hits": 0, "rate": None, "lo": None, "hi": None, "edge": None}
    net = curve_stats(chosen["net_0900"]) if len(chosen) else curve_stats(pd.Series(dtype=float))
    gross = curve_stats(chosen["gross_0900"]) if len(chosen) else curve_stats(pd.Series(dtype=float))
    net_long = curve_stats(chosen_long["net_1400"]) if len(chosen_long) else curve_stats(pd.Series(dtype=float))
    missing = int(books[("SI", "冲突则空仓")]["entry"].isna().sum())
    spliced = int(books[("SI", "冲突则空仓")]["splice"].fillna(False).sum())
    lines.extend(["", "## 事先规定的那一手", ""])
    lines.append(
        f"COMEX 白银、冲突则空仓、8:00 进 9:00 出：{stats['hits']}/{stats['n']}，命中 {pct(stats['rate'])}，"
        f"95% 区间 {pct(stats['lo'])} 到 {pct(stats['hi'])}，方向收益 {bp(stats['edge'])}。"
        f"毛利 {money(gross['total'])} 美元，扣一跳和 5 美元佣金后 {money(net['total'])} 美元，最大回撤 {money(net['dd'])} 美元，"
        f"t 统计量 {num(net['t'])}。同一条持有到大约 14:00，成本后是 {money(net_long['total'])} 美元。"
        f"信号有、但 8:00 这根小时线缺失的有 {missing} 天，开盘跳空超过 3% 被剔除的有 {spliced} 天。"
    )
    lines.append("")
    lines.append("上期所夜盘那一手和这里不是同一段价格。夏令时上期所 21:00 已经在数据公布之后；COMEX 这手在 8:30 之前进场。命中率要分开看。")

    if len(chosen):
        lines.extend(["", "### 按年份和时令", ""])
        lines.append("| 切片 | 手数 | 对 | 命中 | 方向收益 | 成本后 |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        chosen = chosen.copy()
        chosen["year"] = pd.to_datetime(chosen["day"].map(lambda value: value.isoformat())).dt.year
        chosen["season"] = chosen["day"].map(lambda value: "夏令时" if value.month in (4, 5, 6, 7, 8, 9, 10) else "冬令时")
        # October is still EDT in the US through early November. Month 10 is mostly EDT, month 3 is mostly EST. Use the clock instead.
        from event_rules import release_clock

        chosen["season"] = chosen["day"].map(lambda value: "夏令时" if release_clock(value).hour < 21 else "冬令时")
        for label, part in (
            ("2025", chosen[chosen["year"] == 2025]),
            ("2026", chosen[chosen["year"] == 2026]),
            ("夏令时", chosen[chosen["season"] == "夏令时"]),
            ("冬令时", chosen[chosen["season"] == "冬令时"]),
        ):
            part_stats = summarize(part, "ret_open") if len(part) else {"n": 0, "hits": 0, "rate": None, "edge": None}
            part_net = curve_stats(part["net_0900"]) if len(part) else curve_stats(pd.Series(dtype=float))
            lines.append(
                f"| {label} | {part_stats['n']} | {part_stats['hits']} | {pct(part_stats['rate'])} | {bp(part_stats['edge'])} | {money(part_net['total'])} |"
            )
        top3 = float(chosen["net_0900"].sort_values(ascending=False).head(3).sum()) if len(chosen) else 0.0
        lines.append("")
        lines.append(f"8:00 到 9:00 这一手，成本后最大的三笔合计 {money(top3)} 美元。这一段是亏的，2025 和 2026 都亏。")
        lines.extend(["", "### 持有到大约 14:00", ""])
        lines.append("下面仍是冲突则空仓。9:00 和大约 14:00 是取行情之前写定的两种出场。")
        lines.append("")
        lines.append("| 市场 | 切片 | 手数 | 对 | 命中 | 方向收益 | 成本后 | 最大回撤 |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        from event_rules import release_clock

        for market in ("SI", "GC"):
            held = usable(books[(market, "冲突则空仓")], "ret_1400")
            held["year"] = pd.to_datetime(held["day"].map(lambda value: value.isoformat())).dt.year
            held["season"] = held["day"].map(lambda value: "夏令时" if release_clock(value).hour < 21 else "冬令时")
            slices = [
                ("全部", held),
                ("2025", held[held["year"] == 2025]),
                ("2026", held[held["year"] == 2026]),
                ("夏令时", held[held["season"] == "夏令时"]),
                ("冬令时", held[held["season"] == "冬令时"]),
            ]
            for label, part in slices:
                part_stats = summarize(part, "ret_open") if len(part) else {"n": 0, "hits": 0, "rate": None, "edge": None}
                part_net = curve_stats(part["net_1400"]) if len(part) else curve_stats(pd.Series(dtype=float))
                lines.append(
                    f"| {SPECS[market]['name']} | {label} | {part_stats['n']} | {part_stats['hits']} | {pct(part_stats['rate'])} | "
                    f"{bp(part_stats['edge'])} | {money(part_net['total'])} | {money(part_net['dd'])} |"
                )

    lines.extend(["", "## 每一次", ""])
    lines.append("| 发布日 | 发布 | 市场 | 做法 | 方向 | 8:00价 | 9:00价 | 8:00到9:00 | 毛利美元 | 成本后 | 约14:00收益 | 剔除 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    logs = []
    for market in ("SI", "GC"):
        rows = books[(market, "冲突则空仓")].copy()
        rows["market"] = market
        logs.append(rows)
    log = pd.concat(logs, ignore_index=True).sort_values(["day", "market"])
    for _, row in log.iterrows():
        if pd.isna(row["entry"]):
            move = ""
            gross_txt = ""
            net_txt = ""
            late = ""
            dropped = "无8:00行情"
            exit_txt = ""
        else:
            move = bp(row["ret_0900"])
            gross_txt = money(row["gross_0900"])
            net_txt = money(row["net_0900"])
            late = bp(row["ret_1400"])
            dropped = "换月跳空" if bool(row["splice"]) else ""
            exit_txt = "" if pd.isna(row["exit_0900"]) else f"{row['exit_0900']:.4f}"
        entry_txt = "" if pd.isna(row["entry"]) else f"{row['entry']:.4f}"
        lines.append(
            f"| {row['day'].isoformat()} | {row['label']} | {SPECS[row['market']]['name']} | 冲突则空仓 | "
            f"{'多' if row['pred'] > 0 else '空'} | {entry_txt} | {exit_txt} | {move} | {gross_txt} | {net_txt} | {late} | {dropped} |"
        )
    lines.append("")
    lines.append(
        f"数字由 `research/seat_position/comex_backtest.py` 从沪银信号和 Yahoo Finance 小时线算出。生成于 {datetime.now():%Y-%m-%d}。"
    )
    lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("chosen", stats.get("hits"), stats.get("n"), money(net.get("total")))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
