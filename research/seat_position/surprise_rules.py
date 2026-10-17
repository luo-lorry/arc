#!/usr/bin/env python3
"""Predict whether a US print is hot or cold from seats known before it.

A signal on trading day T uses only the disclosed net position through the
15:00 Beijing close of T. US 08:30 ET prints land at 20:30 or 21:30 Beijing.
An FOMC decision at 14:00 ET lands after 02:00 Beijing on the next calendar
day, so the same calendar day's close is still before the decision.

Hot means the released number is above the pre-release consensus. Cold means
it is below. An exact match is not scored. The mapped rule is fixed before
looking at hit rates: a higher gold or silver net, or a rise in that net,
predicts cold. The opposite mapping is reported beside it and is not treated
as a rule chosen in advance.
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
PANEL = ROOT / "product" / "zhongcai" / "precious_panel.json"
SURPRISES = ROOT / "data" / "macro_surprises.json"
OUT = ROOT / "product" / "SURPRISE_RULES.md"

SERIES_NAME = {
    "cpi_mom": "CPI环比",
    "cpi_core": "核心CPI环比",
    "cpi_yoy": "CPI同比",
    "ppi_mom": "PPI环比",
    "nfp": "非农",
    "fomc": "FOMC利率",
}
POOL = ("cpi_mom", "cpi_core", "ppi_mom", "nfp")
MIN_N = 6

SEATS = (
    ("gs", "gs_all", "高盛"),
    ("zc", "zc_near", "中财"),
)
FEATURES = (
    ("chg5", "五日差分", "今天的公布净持仓减去五个交易日前"),
    ("vs_ma5", "相对前五日均线", "今天的公布净持仓减去此前五个交易日的平均"),
    ("ma_delta", "均线五日变化", "含今天在内的五日均线，减去再往前五个交易日的均线"),
    ("chg1", "一日差分", "今天的公布净持仓减去上一个交易日"),
)
SIDES = (
    ("au", "沪金"),
    ("ag", "沪银"),
    ("both", "金银同向"),
)


def wilson_lower(hits: int, n: int, z: float = 1.96) -> float | None:
    if n <= 0:
        return None
    p = hits / n
    z2 = z * z
    denom = 1 + z2 / n
    centre = p + z2 / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return (centre - margin) / denom


def load_panel() -> pd.DataFrame:
    frame = pd.DataFrame(json.loads(PANEL.read_text()))
    frame["date"] = pd.to_datetime(frame["date"])
    for column in ("gs_all", "zc_near"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def seat_features(series: pd.Series) -> pd.DataFrame:
    values = series.astype(float)
    on_board = values.ne(0)
    average = values.rolling(5).mean()
    frame = pd.DataFrame(
        {
            "level": values,
            "chg1": values.diff(1),
            "chg5": values.diff(5),
            "vs_ma5": values - values.shift(1).rolling(5).mean(),
            "ma_delta": average - average.shift(5),
        }
    )
    frame["clean_chg1"] = on_board & on_board.shift(1).eq(True)
    frame["clean_chg5"] = on_board.rolling(6).min().eq(True)
    frame["clean_vs_ma5"] = on_board.rolling(6).min().eq(True)
    frame["clean_ma_delta"] = on_board.rolling(10).min().eq(True)
    frame["clean_level"] = on_board
    return frame


def build_feature_panel(panel: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for product, chunk in panel.groupby("product", sort=False):
        chunk = chunk.sort_values("date").reset_index(drop=True)
        for seat_key, column, _label in SEATS:
            features = seat_features(chunk[column])
            features.insert(0, "date", chunk["date"].dt.date)
            features.insert(1, "product", product)
            features.insert(2, "seat", seat_key)
            parts.append(features)
    return pd.concat(parts, ignore_index=True)


def outcome_of(actual: float, forecast: float) -> int:
    if actual > forecast:
        return 1
    if actual < forecast:
        return -1
    return 0


def predict(signal: float) -> int | None:
    if signal is None or pd.isna(signal) or signal == 0:
        return None
    return -1 if signal > 0 else 1


def latest_seat(panel: pd.DataFrame, day: date) -> date | None:
    dates = panel.loc[panel["date"].dt.date <= day, "date"]
    if dates.empty:
        return None
    return dates.max().date()


def signal_on(features: pd.DataFrame, day: date, product: str, seat: str, feature: str, clean: bool):
    row = features[
        (features["date"] == day) & (features["product"] == product) & (features["seat"] == seat)
    ]
    if row.empty:
        return None
    row = row.iloc[0]
    if clean and not bool(row[f"clean_{feature}"]):
        return None
    value = row[feature]
    if pd.isna(value):
        return None
    return float(value)


def iter_rules():
    for seat_key, _column, seat_label in SEATS:
        for side_key, side_label in SIDES:
            yield {
                "seat": seat_key,
                "side": side_key,
                "feature": "level",
                "clean": True,
                "name": f"{seat_label}{side_label}净持仓方向",
                "detail": "当天净持仓为正猜过冷，为负猜过热。不在前20名不喊。",
            }
            for feature_key, feature_label, _text in FEATURES:
                for clean, clean_label in ((False, "含进出榜"), (True, "窗口都在榜上")):
                    yield {
                        "seat": seat_key,
                        "side": side_key,
                        "feature": feature_key,
                        "clean": clean,
                        "name": f"{seat_label}{side_label}{feature_label}（{clean_label}）",
                        "detail": "",
                    }


def rule_signal(features: pd.DataFrame, day: date, rule: dict) -> float | None:
    def one(product: str):
        return signal_on(features, day, product, rule["seat"], rule["feature"], rule["clean"])

    if rule["side"] != "both":
        return one(rule["side"])
    gold = one("au")
    silver = one("ag")
    if predict(gold) is None or predict(silver) is None or predict(gold) != predict(silver):
        return None
    return float(gold) + float(silver)


def score(panel: pd.DataFrame, features: pd.DataFrame, events: list[dict]) -> pd.DataFrame:
    trading_days = set(panel["date"].dt.date)
    rows = []
    for event in events:
        if event.get("contested"):
            continue
        release = date.fromisoformat(event["date"])
        seat_day = latest_seat(panel, release)
        if seat_day is None:
            continue
        result = outcome_of(event["actual"], event["forecast"])
        for rule in iter_rules():
            signal = rule_signal(features, seat_day, rule)
            call = predict(signal)
            if call is None or result == 0:
                continue
            if event["series"] == "fomc":
                continue
            rows.append(
                {
                    "date": release,
                    "seat_day": seat_day,
                    "same_day": seat_day == release and release in trading_days,
                    "series": event["series"],
                    "pool": event["series"] in POOL,
                    "forecast": event["forecast"],
                    "actual": event["actual"],
                    "result": result,
                    "rule": rule["name"],
                    "seat": rule["seat"],
                    "side": rule["side"],
                    "feature": rule["feature"],
                    "clean": rule["clean"],
                    "signal": signal,
                    "call": call,
                    "hit": call == result,
                }
            )
    return pd.DataFrame(rows)


def half_rates(calls: pd.DataFrame) -> tuple[float | None, float | None, float | None]:
    dates = sorted(calls["date"].unique())
    if len(dates) < 2:
        return None, None, None
    early_dates = set(dates[: len(dates) // 2])
    early = calls[calls["date"].isin(early_dates)]
    late = calls[~calls["date"].isin(early_dates)]
    early_rate = early["hit"].mean() if len(early) else None
    late_rate = late["hit"].mean() if len(late) else None
    if early_rate is None or late_rate is None:
        return early_rate, late_rate, None
    return early_rate, late_rate, min(early_rate, late_rate)


def summarize(calls: pd.DataFrame, name: str) -> dict:
    n = len(calls)
    hits = int(calls["hit"].sum()) if n else 0
    early, late, worse = half_rates(calls) if n else (None, None, None)
    by_series = {}
    for series in POOL:
        part = calls[calls["series"] == series]
        by_series[series] = (int(part["hit"].sum()), len(part)) if len(part) else (0, 0)
    return {
        "name": name,
        "n": n,
        "hits": hits,
        "rate": hits / n if n else None,
        "wilson": wilson_lower(hits, n),
        "early": early,
        "late": late,
        "worse": worse,
        "gap": None if early is None or late is None else abs(early - late),
        "cold_calls": int((calls["call"] == -1).sum()) if n else 0,
        "days": calls["date"].nunique() if n else 0,
        "same_day": int(calls["same_day"].sum()) if n else 0,
        "by_series": by_series,
    }


def pct(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value * 100:.1f}%"


def series_cell(pair: tuple[int, int]) -> str:
    hits, n = pair
    if n == 0:
        return ""
    return f"{hits}/{n}"


def lots(value: float) -> str:
    rounded = round(value)
    if rounded == 0:
        return "0"
    return f"{rounded:+,.0f}"


def heat(result: int) -> str:
    return "过热" if result > 0 else "过冷"


def guess(call: int) -> str:
    return "过冷" if call < 0 else "过热"


def print_number(series: str, value: float) -> str:
    if series == "nfp":
        return f"{value:+,.0f}千人"
    if series == "fomc":
        return f"{value:.2f}%"
    return f"{value:.1f}%"


def focused_names() -> list[str]:
    names = []
    for feature_key, feature_label, _text in FEATURES:
        if feature_key == "chg1":
            continue
        for clean_label in ("含进出榜", "窗口都在榜上"):
            names.append(f"高盛沪金{feature_label}（{clean_label}）")
    names.append("高盛沪金净持仓方向")
    return names


def render(panel: pd.DataFrame, features: pd.DataFrame, events: list[dict], calls: pd.DataFrame) -> str:
    payload = json.loads(SURPRISES.read_text())
    scored_events = [event for event in events if not event.get("contested")]
    lines = [
        "# 用发布前的仓位猜数据冷热",
        "",
        "猜的是公布值相对预期偏高还是偏低，不是沪金、沪银下一交易日的涨跌。持仓只用上期所当天日盘 15:00 已经公布的排名。美国 8:30 ET 的数据落在北京时间 20:30 或 21:30。FOMC 的利率决定在美东 14:00，对应北京时间次日凌晨 2 点或 3 点，所以决定日前一天北京时间下午的排名也来得及。",
        "",
        "方向事先规定，没有先看命中率再挑选：净持仓变高，或者净持仓为正，猜过冷；变低或者为负，猜过热。过冷指公布值低于预期。黄金净多头通常对应更低的实际利率，而数据低于预期才会把实际利率往下压。反方向的命中率写在后面，那一列是看完同一样本才反过来的，不当成已经选定的规则。",
        "",
        "## 五日差分为什么会跳",
        "",
        "五日差分是今天的公布净持仓减去五个交易日前的公布净持仓。不在前 20 名时，这个数记成 0。席位掉出榜再回来，减出来的数就把旧仓位整段算进去，不是这五天里真实加减的手数。",
        "",
    ]

    gold = features[(features["product"] == "au") & (features["seat"] == "gs")].set_index("date")
    day = date(2026, 6, 3)
    window = panel[(panel["product"] == "au") & (panel["date"] >= "2026-05-27") & (panel["date"] <= "2026-06-03")]
    lines.append("2026-06-03 的沪金高盛就是这样。五个交易日前是 2026-05-27。")
    lines.append("")
    lines.append("| 交易日 | 高盛沪金公布净持仓 |")
    lines.append("| --- | --- |")
    for _, row in window.iterrows():
        lines.append(f"| {row['date'].date()} | {lots(row['gs_all'])} |")
    jump = gold.loc[day]
    lines.append("")
    lines.append(
        f"五日差分是 {lots(jump['chg5'])} 手，读成五天里净加了多头。中间三个交易日是 0，人不在榜上。当天重新出现时只有 {lots(jump['level'])} 手。"
        f"现在减去前五日均线仍是 {lots(jump['vs_ma5'])} 手，符号还是加多。今天的仓位整份拿来和过去比，这一跳还在。"
        f"含今天的五日均线相对再往前五日均线，变成 {lots(jump['ma_delta'])} 手，符号反过来，读成减多。均线会改判，不只是把同一个数缩小。"
    )
    lines.append("")
    lines.append(
        "所以滑动平均要分成两种。相对前五日均线，平滑的是比较基准，不是今天这一笔。均线自己的五日变化，才把今天摊进五天。还有一道闸：窗口里只要有一天不在前 20 名，这次不喊。这道闸去掉的是进出榜，不是榜内的大调仓。"
    )
    lines.append("")
    on_board_spike = gold.loc[date(2026, 6, 10)]
    lines.append(
        f"榜内大调仓的例子是 2026-06-10。高盛沪金前一天是 {lots(on_board_spike['level'] - on_board_spike['chg1'])} 手，当天变成 {lots(on_board_spike['level'])} 手，一日变化 {lots(on_board_spike['chg1'])}。"
        f"五日差分 {lots(on_board_spike['chg5'])}，相对前五日均线 {lots(on_board_spike['vs_ma5'])}，两个都还是这一跳。均线五日变化收到 {lots(on_board_spike['ma_delta'])}。这一天窗口里没有 0，进出榜那道闸不会把它滤掉。"
    )

    lines.extend(["", "## 冷热的定义和样本", ""])
    lines.append("| 数据 | 过热 | 过冷 | 相等时 |")
    lines.append("| --- | --- | --- | --- |")
    lines.append("| CPI环比、核心CPI环比、CPI同比、PPI环比 | 公布值高于预期 | 公布值低于预期 | 不记对错 |")
    lines.append("| 非农 | 新增就业高于预期 | 新增就业低于预期 | 不记对错 |")
    lines.append("| FOMC利率 | 目标上限高于会前预期 | 目标上限低于会前预期 | 不记对错 |")
    lines.append("")
    lines.append("预期和公布值来自 FX Blue 的经济日历。2026 年最近几次又对过 Investing.com 和 FrenzyCap。对不上的 2026-04-10 CPI 环比和同比排除。2025-12-18 没有 CPI 环比，政府停摆后 BLS 没有月度变化，只保留同比。非农用首次公布的就业人数，不用后来的修订。")
    lines.append("")
    lines.append("| 数据 | 发布次数 | 过热 | 过冷 | 与预期相同 | 来源对不上已排除 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for series in ("cpi_mom", "cpi_core", "cpi_yoy", "ppi_mom", "nfp", "fomc"):
        group = [event for event in events if event["series"] == series]
        kept = [event for event in group if not event.get("contested")]
        hot = sum(outcome_of(event["actual"], event["forecast"]) > 0 for event in kept)
        cold = sum(outcome_of(event["actual"], event["forecast"]) < 0 for event in kept)
        flat = sum(outcome_of(event["actual"], event["forecast"]) == 0 for event in kept)
        dropped = sum(bool(event.get("contested")) for event in group)
        lines.append(
            f"| {SERIES_NAME[series]} | {len(group)} | {hot} | {cold} | {flat} | {dropped} |"
        )
    fomc_kept = [event for event in scored_events if event["series"] == "fomc"]
    lines.append("")
    lines.append(
        f"FOMC 这 {len(fomc_kept)} 次，目标上限每一次都等于会前预期。利率本身没有过热，也没有过冷，仓位规则没有可记分的 FOMC 样本。声明措辞和点阵图没有一份公布前的一致预期，这里不另打分。"
    )
    pool_events = [event for event in scored_events if event["series"] in POOL]
    pool_hot = sum(outcome_of(event["actual"], event["forecast"]) > 0 for event in pool_events)
    pool_cold = sum(outcome_of(event["actual"], event["forecast"]) < 0 for event in pool_events)
    pool_flat = sum(outcome_of(event["actual"], event["forecast"]) == 0 for event in pool_events)
    surprise_n = pool_hot + pool_cold
    lines.append("")
    lines.append(
        f"主排序只用 CPI环比、核心CPI环比、PPI环比、非农。这四类里有 {len(pool_events)} 次发布，{pool_flat} 次与预期相同，剩下 {surprise_n} 次分出了冷热，其中过热 {pool_hot} 次、过冷 {pool_cold} 次。"
        "核心 CPI 和 CPI 环比常常在同一天，两次结果不是互相独立的样本。CPI同比放在后面单独看，不并进这张排序，避免同一个下午算两次。"
    )
    lines.append("")
    lines.append("高盛沪金、沪银的公布净持仓从 2026-05-26 才不是 0。中财当月加近月，沪金从 2026-03-26、沪银从 2026-03-31 才不是 0。2025 年的发布日这些席位不在前 20 名，规则不喊。不喊不是猜错。")
    lines.append("")
    lines.append("2025-09-11 有 CPI，环比 0.4% 对预期 0.3%，过热。当天高盛和中财的公布净持仓都是 0，任何一条规则都不喊。")
    oct_nfp = next(event for event in events if event["date"] == "2026-10-02" and event["series"] == "nfp")
    oct_seat = latest_seat(panel, date(2026, 10, 2))
    lines.append(
        f"2026-10-02 非农是 {print_number('nfp', oct_nfp['actual'])} 对预期 {print_number('nfp', oct_nfp['forecast'])}，过冷。上期所国庆休市，用的持仓日是 {oct_seat}，仍在公布之前。"
    )

    pool_calls = calls[calls["pool"]].copy()
    summaries = []
    for rule in iter_rules():
        part = pool_calls[pool_calls["rule"] == rule["name"]]
        if part.empty:
            continue
        summaries.append(summarize(part, rule["name"]))
    summaries.sort(
        key=lambda row: (
            -(row["wilson"] or -1),
            -(row["worse"] if row["worse"] is not None else -1),
            -row["n"],
            row["name"],
        )
    )

    by_name = {row["name"]: row for row in summaries}
    clean = by_name["高盛沪金五日差分（窗口都在榜上）"]
    raw = by_name["高盛沪金五日差分（含进出榜）"]
    versus = by_name["高盛沪金相对前五日均线（含进出榜）"]
    smoothed = by_name["高盛沪金均线五日变化（窗口都在榜上）"]
    lead = [
        "## 结果",
        "",
        (
            f"排第一的是高盛沪金五日差分，并且比较的六天都在前 20 名。{clean['n']} 次里对了 {clean['hits']} 次，"
            f"命中 {pct(clean['rate'])}。按发布日对半切开，两段都是 {pct(clean['worse'])}。按喊次把命中率往下压之后是 {pct(clean['wilson'])}。"
        ),
        "",
        (
            f"同一条五日差分如果把进出榜的 0 也算进去，是 {raw['hits']}/{raw['n']}，命中 {pct(raw['rate'])}，较差半段 {pct(raw['worse'])}。"
            f"现在的仓位减去前五日均线，是 {versus['hits']}/{versus['n']}，命中 {pct(versus['rate'])}，较差半段 {pct(versus['worse'])}。"
            f"均线自己的五日变化，并且十天都在榜上，是 {smoothed['hits']}/{smoothed['n']}，命中 {pct(smoothed['rate'])}，较差半段 {pct(smoothed['worse'])}。"
        ),
        "",
        "窗口都在榜上的五日差分，命中率和两段里较差的一段都高于现在减去前五日均线。均线自身的变化在十天都在榜上时，表面命中更高，喊次更少，压低后排在第二。",
        "",
        (
            "拆开看，这条第一名在 CPI环比是 {cpi}，核心CPI环比是 {core}，PPI环比是 {ppi}，非农是 {nfp}。"
            "样本集中在 2026 年 6 月到 10 月，前半段和后半段都在这几个月里面。"
        ).format(
            cpi=series_cell(clean["by_series"]["cpi_mom"]),
            core=series_cell(clean["by_series"]["cpi_core"]),
            ppi=series_cell(clean["by_series"]["ppi_mom"]),
            nfp=series_cell(clean["by_series"]["nfp"]),
        ),
        "",
        "FOMC 的目标利率 14 次都等于会前预期，没有过热或过冷可以记分。",
        "",
    ]
    lines[lines.index("## 五日差分为什么会跳"):lines.index("## 五日差分为什么会跳")] = lead

    lines.extend(["", "## 高盛沪金：差分和均线放在一起", ""])
    lines.append("这一张只回答差分和均线哪一种更稳。样本小的行也留在这里，不因为喊次少就藏起来。")
    lines.append("")
    lines.append("| 规则 | 喊次 | 对 | 命中 | 压低后 | 前半段 | 后半段 | 较差半段 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for name in focused_names():
        row = by_name.get(name)
        if row is None:
            lines.append(f"| {name} | 0 | 0 |  |  |  |  |  |")
            continue
        lines.append(
            f"| {name} | {row['n']} | {row['hits']} | {pct(row['rate'])} | {pct(row['wilson'])} | {pct(row['early'])} | {pct(row['late'])} | {pct(row['worse'])} |"
        )
    lines.append("")
    lines.append("压低后的命中率按喊次把表面命中率往下压。喊两次全对，压完大约三分之一。前半段、后半段按这条规则喊过的发布日对半切开。较差半段是两段里更低的命中率，用来看后一半还有没有前一半那么准。")

    main = [row for row in summaries if row["n"] >= MIN_N]
    small = [row for row in summaries if row["n"] < MIN_N]
    lines.extend(["", "## 全部规则的排序", ""])
    lines.append(
        f"排序先看压低后的命中率，再看较差半段，再看喊次。喊次少于 {MIN_N} 的不进主表。四类数据合在一起计。同一天的 CPI环比和核心CPI环比会各算一次。净持仓变高猜过冷，变低猜过热。前半段和后半段都落在席位上榜之后的几个月里。"
    )
    lines.append("")
    lines.append("| 排名 | 规则 | 喊次 | 对 | 命中 | 压低后 | 前半段 | 后半段 | 较差半段 | 落差 | 猜冷 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for rank, row in enumerate(main, start=1):
        lines.append(
            f"| {rank} | {row['name']} | {row['n']} | {row['hits']} | {pct(row['rate'])} | {pct(row['wilson'])} | "
            f"{pct(row['early'])} | {pct(row['late'])} | {pct(row['worse'])} | {pct(row['gap'])} | {row['cold_calls']} |"
        )
    if not main:
        lines.append("|  | 没有规则达到最少喊次 |  |  |  |  |  |  |  |  |  |")

    lines.extend(["", "## 主表规则在每类数据上的对错", ""])
    lines.append("| 排名 | 规则 | CPI环比 | 核心CPI | PPI环比 | 非农 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for rank, row in enumerate(main, start=1):
        lines.append(
            "| {rank} | {name} | {cpi} | {core} | {ppi} | {nfp} |".format(
                rank=rank,
                name=row["name"],
                cpi=series_cell(row["by_series"]["cpi_mom"]),
                core=series_cell(row["by_series"]["cpi_core"]),
                ppi=series_cell(row["by_series"]["ppi_mom"]),
                nfp=series_cell(row["by_series"]["nfp"]),
            )
        )

    lines.extend(["", "## 喊次不够的规则", ""])
    lines.append(f"下面这些喊次不到 {MIN_N}。命中率可以很高，一次分半之后往往一边没有样本，不拿来排序。")
    lines.append("")
    lines.append("| 规则 | 喊次 | 对 | 命中 | 压低后 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in small:
        lines.append(f"| {row['name']} | {row['n']} | {row['hits']} | {pct(row['rate'])} | {pct(row['wilson'])} |")

    lines.extend(["", "## 把方向反过来", ""])
    lines.append("下表用的是相反映射：净持仓变高改猜过热。这是同一批喊叫看完之后翻转的，只用来检查事先规定的方向是不是反了。它不另外产生新的仓位信息。")
    lines.append("")
    flipped = []
    for row in summaries:
        if row["n"] < MIN_N:
            continue
        hits = row["n"] - row["hits"]
        flipped.append((wilson_lower(hits, row["n"]), hits, row))
    flipped.sort(key=lambda item: (-(item[0] or -1), -item[1], item[2]["name"]))
    lines.append("| 排名 | 规则 | 喊次 | 反方向对 | 反方向命中 | 反方向压低后 | 原方向命中 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for rank, (lower, hits, row) in enumerate(flipped, start=1):
        lines.append(
            f"| {rank} | {row['name']} | {row['n']} | {hits} | {pct(hits / row['n'])} | {pct(lower)} | {pct(row['rate'])} |"
        )

    yoy = calls[calls["series"] == "cpi_yoy"]
    yoy_surprise = [
        event
        for event in scored_events
        if event["series"] == "cpi_yoy" and outcome_of(event["actual"], event["forecast"]) != 0
    ]
    lines.extend(["", "## CPI同比，单独记", ""])
    lines.append(
        f"同比和环比是同一个下午，不并进主排序。排除来源对不上的日子之后，同比分出冷热的有 {len(yoy_surprise)} 次。"
        "其中大部分在高盛、中财上榜之前，规则不喊。上榜之后能喊到的次数不超过 3，所以这里只列出喊次至少 2 的规则。"
    )
    lines.append("")
    lines.append("| 规则 | 喊次 | 对 | 命中 |")
    lines.append("| --- | --- | --- | --- |")
    yoy_rows = []
    for rule in iter_rules():
        part = yoy[yoy["rule"] == rule["name"]]
        if len(part) < 2:
            continue
        yoy_rows.append((int(part["hit"].sum()), len(part), rule["name"]))
    yoy_rows.sort(key=lambda item: (-item[0] / item[1], -item[1], item[2]))
    for hits, n, name in yoy_rows:
        lines.append(f"| {name} | {n} | {hits} | {pct(hits / n)} |")
    if not yoy_rows:
        lines.append("| 没有规则在 CPI同比上喊到 2 次 |  |  |  |")

    lines.extend(["", "## 排在前面的规则，每一次", ""])
    top = main[:3]
    if not top:
        lines.append("主表是空的。")
    for row in top:
        part = pool_calls[pool_calls["rule"] == row["name"]].sort_values(["date", "series"])
        lines.append(f"### {row['name']}")
        lines.append("")
        lines.append("| 发布日 | 持仓日 | 数据 | 信号 | 猜测 | 公布 | 预期 | 结果 |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for _, call in part.iterrows():
            mark = "对" if call["hit"] else "错"
            lines.append(
                f"| {call['date']} | {call['seat_day']} | {SERIES_NAME[call['series']]} | {lots(call['signal'])} | "
                f"{guess(call['call'])} | {print_number(call['series'], call['actual'])} | "
                f"{print_number(call['series'], call['forecast'])} | {mark}，实际{heat(call['result'])} |"
            )
        day_hit = part.groupby("date")["hit"].all()
        lines.append("")
        lines.append(
            f"这 {row['n']} 次落在 {len(day_hit)} 个发布日上。{int(day_hit.sum())} 天是当天每一条结果都对。"
        )
        lines.append("")

    stale = pool_calls[~pool_calls["same_day"]]
    lines.extend(["## 发布日交易所没开市", ""])
    if stale.empty:
        lines.append("主样本里的每一次喊叫，持仓日都是发布日当天。")
    else:
        lines.append(f"有 {stale['date'].nunique()} 个发布日、{len(stale)} 次喊叫用的是发布日之前最近一个交易日的持仓。这些持仓仍然早于公布。")
        lines.append("")
        shown = (
            stale.groupby(["date", "seat_day", "series"], as_index=False)
            .size()
            .sort_values("date")
        )
        lines.append("| 发布日 | 持仓日 | 数据 | 落到这一天的喊叫 |")
        lines.append("| --- | --- | --- | --- |")
        for _, row in shown.iterrows():
            lines.append(f"| {row['date']} | {row['seat_day']} | {SERIES_NAME[row['series']]} | {row['size']} |")
    lines.append("")
    lines.append("信号是 0、席位不在榜上，或者金银同向那条规则里两个品种方向相反，这一天就不喊。不喊不要补成过热或过冷。下一次 FOMC 是 2026-10-28，还没公布。前面 14 次利率都等于预期，这条规则在利率上没有可引用的命中率。")
    lines.append("")
    generated = datetime.now().strftime("%Y-%m-%d")
    lines.append(f"数字由 `research/seat_position/surprise_rules.py` 从持仓面板和 `data/macro_surprises.json` 算出。本页生成于 {generated}。")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    panel = load_panel()
    features = build_feature_panel(panel)
    events = json.loads(SURPRISES.read_text())["events"]
    calls = score(panel, features, events)
    text = render(panel, features, events, calls)
    OUT.write_text(text)
    pool = calls[calls["pool"]]
    print(f"pool calls {len(pool)} rules {pool['rule'].nunique()}")
    summary = (
        pool.groupby("rule")["hit"]
        .agg(["sum", "count"])
        .reset_index()
        .sort_values(["count", "sum"], ascending=False)
    )
    summary["rate"] = summary["sum"] / summary["count"]
    print(summary.head(20).to_string(index=False))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
