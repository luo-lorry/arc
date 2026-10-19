#!/usr/bin/env python3
"""Rank pre-release position rules, then leave a forward call.

A signal on trading day T uses only seats, warrants and prices known at the
15:00 Beijing close of T. The outcome is the next session, which includes the
night that starts at 21:00. US 08:30 ET prints land at 20:30 or 21:30 Beijing,
after that close. Days after the last finished session are not scored.
"""

from __future__ import annotations

import json
import urllib.request
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from forecast_au_ag import parse_html

ROOT = Path(__file__).resolve().parent
PANEL = ROOT / "product" / "zhongcai" / "precious_panel.json"
WARRANTS = ROOT / "data" / "warrants_au_ag.json"
OUT = ROOT / "product" / "EVENT_RULES.md"
UA = {"User-Agent": "Mozilla/5.0"}
NY = ZoneInfo("America/New_York")
SH = ZoneInfo("Asia/Shanghai")
MIN_HISTORY = 40
SELECT_MIN_N = 12

# Official 08:30 ET releases. Same-day names are joined later.
# 2025 PCE/GDP is not on the BEA page that now starts in 2026, so it is absent.
CORE = [
    ("2025-01-10", "非农"),
    ("2025-01-14", "PPI"),
    ("2025-01-15", "CPI"),
    ("2025-01-31", "ECI"),
    ("2025-02-07", "非农"),
    ("2025-02-12", "CPI"),
    ("2025-02-13", "PPI"),
    ("2025-03-07", "非农"),
    ("2025-03-12", "CPI"),
    ("2025-03-13", "PPI"),
    ("2025-04-04", "非农"),
    ("2025-04-10", "CPI"),
    ("2025-04-11", "PPI"),
    ("2025-04-30", "ECI"),
    ("2025-05-02", "非农"),
    ("2025-05-13", "CPI"),
    ("2025-05-15", "PPI"),
    ("2025-06-06", "非农"),
    ("2025-06-11", "CPI"),
    ("2025-06-12", "PPI"),
    ("2025-07-03", "非农"),
    ("2025-07-15", "CPI"),
    ("2025-07-16", "PPI"),
    ("2025-07-31", "ECI"),
    ("2025-08-01", "非农"),
    ("2025-08-12", "CPI"),
    ("2025-08-14", "PPI"),
    ("2025-09-05", "非农"),
    ("2025-09-10", "PPI"),
    ("2025-09-11", "CPI"),
    ("2025-10-24", "CPI"),
    ("2025-11-20", "非农"),
    ("2025-11-25", "PPI+零售"),
    ("2025-12-10", "ECI"),
    ("2025-12-16", "非农+零售"),
    ("2025-12-18", "CPI"),
    ("2026-01-09", "非农"),
    ("2026-01-13", "CPI"),
    ("2026-01-14", "PPI+零售"),
    ("2026-01-22", "GDP"),
    ("2026-01-30", "PPI"),
    ("2026-02-10", "ECI+零售"),
    ("2026-02-11", "非农"),
    ("2026-02-13", "CPI"),
    ("2026-02-20", "GDP+PCE"),
    ("2026-02-27", "PPI"),
    ("2026-03-06", "非农+零售"),
    ("2026-03-11", "CPI"),
    ("2026-03-13", "GDP+PCE"),
    ("2026-03-18", "PPI"),
    ("2026-04-01", "零售"),
    ("2026-04-03", "非农"),
    ("2026-04-09", "GDP+PCE"),
    ("2026-04-10", "CPI"),
    ("2026-04-14", "PPI"),
    ("2026-04-21", "零售"),
    ("2026-04-30", "ECI+GDP+PCE"),
    ("2026-05-08", "非农"),
    ("2026-05-12", "CPI"),
    ("2026-05-13", "PPI"),
    ("2026-05-14", "零售"),
    ("2026-05-28", "GDP+PCE"),
    ("2026-06-05", "非农"),
    ("2026-06-10", "CPI"),
    ("2026-06-11", "PPI"),
    ("2026-06-17", "零售"),
    ("2026-06-25", "GDP+PCE"),
    ("2026-07-02", "非农"),
    ("2026-07-14", "CPI"),
    ("2026-07-15", "PPI"),
    ("2026-07-16", "零售"),
    ("2026-07-30", "GDP+PCE"),
    ("2026-07-31", "ECI"),
    ("2026-08-07", "非农"),
    ("2026-08-12", "CPI"),
    ("2026-08-13", "PPI"),
    ("2026-08-14", "零售"),
    ("2026-08-26", "GDP+PCE"),
    ("2026-09-04", "非农"),
    ("2026-09-10", "PPI"),
    ("2026-09-11", "CPI"),
    ("2026-09-16", "零售"),
    ("2026-09-30", "GDP+PCE"),
    ("2026-10-02", "非农"),
    ("2026-10-14", "CPI"),
    ("2026-10-15", "PPI+零售"),
]


def http_json(url: str) -> dict | None:
    req = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=40) as resp:
            if resp.status != 200:
                return None
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def load_oct16_prices() -> dict[str, dict]:
    payload = http_json(
        "https://www.shfe.com.cn/data/tradedata/future/dailydata/kx20261016.dat"
    )
    if not payload or not payload.get("o_curinstrument"):
        return {}
    best: dict[str, dict] = {}
    for item in payload["o_curinstrument"]:
        group = str(item.get("PRODUCTGROUPID") or "").lower()
        month = str(item.get("DELIVERYMONTH") or "")
        if group not in ("au", "ag") or len(month) != 4 or not month.isdigit():
            continue
        oi = float(item.get("OPENINTEREST") or 0)
        if group not in best or oi > best[group]["oi"]:
            best[group] = {
                "main": (group + month).upper(),
                "open": float(item["OPENPRICE"]),
                "close": float(item["CLOSEPRICE"]),
                "settle": float(item["SETTLEMENTPRICE"]),
                "pre": float(item["PRESETTLEMENTPRICE"]),
                "oi": oi,
            }
    return best


def load_oct16_warrants() -> dict[str, float] | None:
    path = ROOT / "data" / "warrant_html" / "20261016.html"
    if not path.exists():
        url = (
            "https://www.shfe.com.cn/data/tradedata/future/stockdata/"
            "dailystock_20261016/ZH/all.html"
        )
        req = urllib.request.Request(url, headers=UA)
        try:
            with urllib.request.urlopen(req, timeout=40) as resp:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(resp.read())
        except Exception:
            return None
    parsed = parse_html(path)
    return {"au": parsed["au"][0], "ag": parsed["ag"][0], "au_chg": parsed["au"][1], "ag_chg": parsed["ag"][1]}


def release_clock(day: date) -> datetime:
    return datetime.combine(day, time(8, 30), tzinfo=NY).astimezone(SH)


def pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{value * 100:.1f}%"


def signed_lots(value: float) -> str:
    return f"{value:+,.0f}"


def add_features(frame: pd.DataFrame, warrants: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _product, chunk in frame.groupby("product", sort=False):
        chunk = chunk.sort_values("date").reset_index(drop=True).copy()
        for column in ("gs_all", "gs_near", "zc_near", "zc_near_chg5", "gs_all_chg5", "close", "oi"):
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
        chunk["gs_chg1"] = chunk["gs_all"].diff()
        chunk["px_chg5"] = chunk["close"].diff(5)
        chunk["oi_chg5"] = chunk["oi"].diff(5)
        product_warrants = warrants.loc[warrants["product"] == _product, ["date", "w_chg5"]]
        merged = chunk.merge(product_warrants, on="date", how="left")
        rows = []
        for i, row in merged.iterrows():
            item = row.to_dict()
            item["gs_p10"] = item["gs_p90"] = None
            item["zc_p10"] = item["zc_p90"] = None
            item["w_p10"] = item["w_p90"] = None
            if i >= MIN_HISTORY:
                prior_gs = merged["gs_all"].iloc[:i]
                item["gs_p10"] = float(prior_gs.quantile(0.10))
                item["gs_p90"] = float(prior_gs.quantile(0.90))
                prior_zc = merged["zc_near_chg5"].iloc[:i].dropna()
                if len(prior_zc) >= MIN_HISTORY:
                    item["zc_p10"] = float(prior_zc.quantile(0.10))
                    item["zc_p90"] = float(prior_zc.quantile(0.90))
                prior_w = merged["w_chg5"].iloc[:i].dropna()
                if len(prior_w) >= MIN_HISTORY:
                    item["w_p10"] = float(prior_w.quantile(0.10))
                    item["w_p90"] = float(prior_w.quantile(0.90))
            rows.append(item)
        parts.append(pd.DataFrame(rows))
    return pd.concat(parts, ignore_index=True)


def call_of(rule: str, row: pd.Series) -> int:
    gs = row["gs_all"]
    gs_chg = row["gs_chg1"]
    zc5 = row["zc_near_chg5"]
    w5 = row["w_chg5"]
    if rule == "gs_sign":
        if gs < 0:
            return -1
        if gs > 0:
            return 1
        return 0
    if rule == "gs_short_only":
        return -1 if gs < 0 else 0
    if rule == "gs_tail":
        if pd.isna(row["gs_p10"]):
            return 0
        if gs <= row["gs_p10"]:
            return -1
        if gs >= row["gs_p90"]:
            return 1
        return 0
    if rule == "gs_tail_short":
        if pd.isna(row["gs_p10"]):
            return 0
        return -1 if gs <= row["gs_p10"] else 0
    if rule == "gs_chg5":
        value = row["gs_all_chg5"]
        if pd.isna(value) or value == 0:
            return 0
        return -1 if value < 0 else 1
    if rule == "gs_add_short":
        if pd.isna(gs_chg):
            return 0
        return -1 if gs < 0 and gs_chg < 0 else 0
    if rule == "gs_near_sign":
        if row["gs_near"] < 0:
            return -1
        if row["gs_near"] > 0:
            return 1
        return 0
    if rule == "zc_chg5":
        if pd.isna(zc5) or zc5 == 0:
            return 0
        return -1 if zc5 < 0 else 1
    if rule == "zc_tail":
        if pd.isna(zc5) or pd.isna(row["zc_p10"]):
            return 0
        if zc5 <= row["zc_p10"]:
            return -1
        if zc5 >= row["zc_p90"]:
            return 1
        return 0
    if rule == "warrant_sign":
        if pd.isna(w5) or w5 == 0:
            return 0
        return 1 if w5 < 0 else -1
    if rule == "warrant_tail":
        if pd.isna(w5) or pd.isna(row["w_p10"]):
            return 0
        if w5 <= row["w_p10"]:
            return 1
        if w5 >= row["w_p90"]:
            return -1
        return 0
    if rule == "price_oi":
        if pd.isna(row["px_chg5"]) or pd.isna(row["oi_chg5"]) or row["oi_chg5"] <= 0:
            return 0
        if row["px_chg5"] < 0:
            return -1
        if row["px_chg5"] > 0:
            return 1
        return 0
    if rule in ("combo1", "combo2"):
        score = 0
        if not pd.isna(row["gs_p10"]):
            if gs <= row["gs_p10"]:
                score -= 1
            elif gs >= row["gs_p90"]:
                score += 1
        if not pd.isna(zc5) and not pd.isna(row["zc_p10"]):
            if zc5 <= row["zc_p10"]:
                score -= 1
            elif zc5 >= row["zc_p90"]:
                score += 1
        if not pd.isna(w5) and not pd.isna(row["w_p10"]):
            if w5 <= row["w_p10"]:
                score += 1
            elif w5 >= row["w_p90"]:
                score -= 1
        if not pd.isna(row["px_chg5"]) and not pd.isna(row["oi_chg5"]) and row["oi_chg5"] > 0:
            if row["px_chg5"] < 0:
                score -= 1
            elif row["px_chg5"] > 0:
                score += 1
        need = 1 if rule == "combo1" else 2
        if score <= -need:
            return -1
        if score >= need:
            return 1
        return 0
    if rule == "gs_and_zc_cut":
        if gs < 0 and not pd.isna(zc5) and zc5 < 0:
            return -1
        return 0
    if rule == "gs_tail_and_zc_cut":
        if pd.isna(row["gs_p10"]) or pd.isna(zc5):
            return 0
        return -1 if gs <= row["gs_p10"] and zc5 < 0 else 0
    if rule == "gs_short_warrant_up":
        if gs < 0 and not pd.isna(w5) and w5 > 0:
            return -1
        return 0
    if rule == "gs_short_warrant_down":
        if gs < 0 and not pd.isna(w5) and w5 < 0:
            return -1
        return 0
    raise KeyError(rule)


RULES = [
    ("gs_sign", "高盛公布净持仓为负看跌，为正看涨"),
    ("gs_short_only", "只在高盛公布净持仓为负时看跌"),
    ("gs_tail", "高盛净持仓低于自身10%分位看跌，高于90%分位看涨"),
    ("gs_tail_short", "只在高盛净持仓低于自身10%分位时看跌"),
    ("gs_chg5", "高盛五日净持仓下降看跌，上升看涨"),
    ("gs_add_short", "高盛已经净空，且当天净持仓继续下降，看跌"),
    ("gs_near_sign", "高盛当月加近月净持仓为负看跌，为正看涨"),
    ("zc_chg5", "中财当月加近月五日变化为负看跌，为正看涨"),
    ("zc_tail", "中财近月五日变化低于自身10%分位看跌，高于90%分位看涨"),
    ("warrant_sign", "仓单五日增加看跌，减少看涨"),
    ("warrant_tail", "仓单五日变化进入增加的10%极端看跌，进入减少的10%极端看涨"),
    ("price_oi", "五日价跌且持仓增加看跌，价涨且持仓增加看涨"),
    ("combo1", "高盛分位、中财分位、仓单分位、价仓四项合计达到1就喊方向"),
    ("combo2", "上面四项合计达到2才喊方向"),
    ("gs_and_zc_cut", "高盛净空，且中财近月五日在减少，看跌"),
    ("gs_tail_and_zc_cut", "高盛低于自身10%分位，且中财近月五日在减少，看跌"),
    ("gs_short_warrant_up", "高盛净空，且仓单五日增加，看跌"),
    ("gs_short_warrant_down", "高盛净空，且仓单五日减少，仍看跌"),
]


def score_mask(frame: pd.DataFrame, mask: pd.Series, rule: str, outcome: str) -> dict:
    called = frame.loc[mask].copy()
    called["pred"] = called.apply(lambda row: call_of(rule, row), axis=1)
    called = called[called["pred"] != 0]
    called = called[called[outcome].notna() & (called[outcome] != 0)]
    n = int(len(called))
    if n == 0:
        return {"n": 0, "hit": None, "down": None, "edge": None, "longs": 0, "shorts": 0}
    hit = float(((called["pred"] * called[outcome]) > 0).mean())
    down = float((called[outcome] < 0).mean())
    edge = float((called["pred"] * called[outcome]).mean())
    return {
        "n": n,
        "hit": hit,
        "down": down,
        "edge": edge,
        "longs": int((called["pred"] > 0).sum()),
        "shorts": int((called["pred"] < 0).sum()),
    }


def table(rows: list[dict]) -> list[str]:
    lines = [
        "| 排名 | 规则 | 喊次 | 看空 | 看多 | 方向命中 | 喊次里实际下跌 | 每次判断的平均价差 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    ordered = sorted(
        rows,
        key=lambda item: (
            -(item["hit"] if item["hit"] is not None else -1),
            -item["n"],
        ),
    )
    for i, item in enumerate(ordered, 1):
        lines.append(
            f"| {i} | {item['name']} | {item['n']} | {item['shorts']} | {item['longs']} | "
            f"{pct(item['hit'])} | {pct(item['down'])} | {pct(item['edge'])} |"
        )
    return lines


def choose(rows: list[dict]) -> dict | None:
    eligible = [
        row for row in rows if row["n"] >= SELECT_MIN_N and row["edge"] is not None and row["edge"] > 0
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda row: (row["hit"], row["edge"], row["n"]))


def main() -> None:
    panel = pd.DataFrame(json.loads(PANEL.read_text()))
    panel["date"] = pd.to_datetime(panel["date"])
    prices16 = load_oct16_prices()
    pm16 = http_json(
        "https://www.shfe.com.cn/data/tradedata/future/dailydata/pm20261016.dat"
    )
    extra = []
    for product, name in (("au", "沪金"), ("ag", "沪银")):
        if product not in prices16:
            continue
        quote = prices16[product]
        extra.append(
            {
                "date": pd.Timestamp("2026-10-16"),
                "product": product,
                "name": name,
                "main": quote["main"],
                "open": quote["open"],
                "close": quote["close"],
                "settle": quote["settle"],
                "pre": quote["pre"],
                "oi": quote["oi"],
                "gs_all": None,
                "gs_near": None,
                "gs_front": None,
                "zc_all": None,
                "zc_near": None,
                "zc_front": None,
                "gs_all_chg5": None,
                "gs_near_chg5": None,
                "zc_all_chg5": None,
                "zc_near_chg5": None,
            }
        )
    if extra:
        panel = pd.concat([panel, pd.DataFrame(extra)], ignore_index=True)
    parts = []
    for _product, chunk in panel.groupby("product", sort=False):
        chunk = chunk.sort_values("date").copy()
        chunk["ret_next"] = chunk["close"].shift(-1) / chunk["settle"] - 1.0
        chunk["gap_next"] = chunk["open"].shift(-1) / chunk["settle"] - 1.0
        parts.append(chunk)
    panel = pd.concat(parts, ignore_index=True)

    warrants = pd.DataFrame(json.loads(WARRANTS.read_text()))
    warrants["date"] = pd.to_datetime(warrants["date"])
    fresh = load_oct16_warrants()
    if fresh is not None:
        warrants = pd.concat(
            [
                warrants,
                pd.DataFrame(
                    [
                        {
                            "date": pd.Timestamp("2026-10-16"),
                            "au_kg": fresh["au"],
                            "ag_kg": fresh["ag"],
                            "au_chg": fresh["au_chg"],
                            "ag_chg": fresh["ag_chg"],
                            "au_chg5": fresh["au"] - float(warrants.iloc[-5]["au_kg"]),
                            "ag_chg5": fresh["ag"] - float(warrants.iloc[-5]["ag_kg"]),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    warrant_rows = []
    for product, column in (("au", "au_chg5"), ("ag", "ag_chg5")):
        part = warrants[["date", column]].rename(columns={column: "w_chg5"})
        part["product"] = product
        warrant_rows.append(part)
    warrant_long = pd.concat(warrant_rows, ignore_index=True)
    featured = add_features(panel, warrant_long)
    featured = featured[featured["gs_all"].notna()].copy()

    trading_days = list(pd.to_datetime(sorted(panel["date"].unique())))
    day_set = set(day.date() for day in trading_days)

    def previous_trading_day(day: date) -> date | None:
        earlier = [item for item in trading_days if item.date() < day]
        return None if not earlier else earlier[-1].date()

    def next_trading_day(day: date) -> date | None:
        later = [item for item in trading_days if item.date() > day]
        return None if not later else later[0].date()

    event_rows = []
    for text, label in CORE:
        day = date.fromisoformat(text)
        clock = release_clock(day)
        if day in day_set:
            position_day = day
            same = True
        else:
            position_day = previous_trading_day(day)
            same = False
        if position_day is None:
            continue
        event_rows.append(
            {
                "release": day,
                "label": label,
                "position_day": position_day,
                "same": same,
                "beijing": clock.strftime("%H:%M"),
                "before_night": clock.hour < 21,
            }
        )
    events = pd.DataFrame(event_rows)
    # One row per position day: join labels that share the afternoon.
    grouped = (
        events[events["same"]]
        .groupby("position_day", as_index=False)
        .agg(label=("label", lambda s: "+".join(dict.fromkeys(s))), beijing=("beijing", "first"), before_night=("before_night", "all"))
    )
    featured["day"] = featured["date"].dt.date
    featured = featured.merge(
        grouped.rename(columns={"position_day": "day"}),
        on="day",
        how="left",
    )
    featured["is_event"] = featured["label"].notna()
    known = featured["ret_next"].notna()
    scopes = {
        "发布日下午": featured["is_event"] & known,
        "全部交易日": known,
    }
    rank_rows = {scope: [] for scope in scopes}
    for scope, mask in scopes.items():
        for key, name in RULES:
            stats = score_mask(featured, mask, key, "ret_next")
            gap = score_mask(featured, mask & featured["before_night"].fillna(False), key, "gap_next") if scope == "发布日下午" else None
            rank_rows[scope].append({"key": key, "name": name, **stats, "gap_hit": None if gap is None else gap["hit"], "gap_n": None if gap is None else gap["n"]})

    # Walk-forward: choose on reactions strictly before 2026-07-01.
    featured["reaction"] = featured["day"].map(lambda day: next_trading_day(day) if day in day_set else None)
    early = featured["is_event"] & featured["reaction"].map(lambda day: day is not None and day < date(2026, 7, 1))
    late = featured["is_event"] & featured["reaction"].map(lambda day: day is not None and day >= date(2026, 7, 1)) & known
    early_rows = []
    for key, name in RULES:
        stats = score_mask(featured, early, key, "ret_next")
        early_rows.append({"key": key, "name": name, **stats})
    early_choice = choose(early_rows)
    late_check = None
    if early_choice is not None:
        late_check = score_mask(featured, late, early_choice["key"], "ret_next")

    event_known = featured["is_event"] & known
    base_down = float((featured.loc[event_known, "ret_next"] < 0).mean())
    base_n = int(event_known.sum())

    lines = [
        "# 数据发布前的持仓规则",
        "",
        "判断时点是上期所日盘 15:00。美国 8:30 ET 的数据落在北京时间 20:30（夏令时）或 21:30（冬令时），都在这份持仓公布之后。下一交易日含当晚 21:00 夜盘。分位数只用当天之前的历史。10 月 16 日的收盘用来给 10 月 15 日的判断记结果，没有拿来制造信号。",
        "",
        "事件用的是 BLS 日程里的 CPI、非农、PPI、ECI，Census 从 2025 年 11 月起公布的零售销售，以及 BEA 当前日程里 8:30 公布的 GDP 和 PCE。2025 年的 GDP、PCE 不在这份 BEA 页面上，没有补猜。同一天有多项发布就并成一个下午。10:00 ET 的数据夜盘已经开了，FOMC 在北京时间凌晨，周度失业金申领没有放进这张表，避免把样本冲成每周四。",
        "",
        f"发布日下午、并且下一交易日已经走完的样本有 {base_n} 条（金银分开计）。其中下一交易日收跌的比例是 {pct(base_down)}。只喊跌的规则，命中率要和这个比例对着看。",
        "",
        "pm20261016 仍是 404，东方财富同一天的会员持仓也是空的。10 月 16 日下午的高盛、中财仓位现在还不能用。日行情和仓单日报已经有了。",
        "",
        "## 发布日下午，按下一交易日方向命中率",
        "",
    ]
    lines.extend(table(rank_rows["发布日下午"]))
    lines.extend(["", "平均价差是每次判断的方向乘以下一交易日收益率，看错的次数也算在里面。正数表示这组判断平均下来价差为正。", ""])
    lines.extend(["## 同样的规则，放到全部交易日", ""])
    lines.extend(table(rank_rows["全部交易日"]))
    lines.extend(["", "## 先在 2026 年 7 月以前选定，再看后面", ""])
    if early_choice is None:
        lines.append("2026 年 7 月以前、喊次至少 12 次且平均价差为正的规则没有出现。")
    else:
        lines.append(
            f"7 月以前喊次 {early_choice['n']}、命中 {pct(early_choice['hit'])}、平均价差 {pct(early_choice['edge'])} 的规则是：{early_choice['name']}。"
        )
        if late_check is not None:
            lines.append(
                f"同一条拿到 7 月及以后已经走完的发布日上，喊次 {late_check['n']}，命中 {pct(late_check['hit'])}，平均价差 {pct(late_check['edge'])}。"
            )
    lines.extend(["", "## 最近几次发布日", ""])
    lines.append("| 持仓日 | 发布 | 北京时间 | 品种 | 高盛净持仓 | 中财近月五日 | 仓单五日 | 下一交易日 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    recent = featured[featured["is_event"]].sort_values("date").groupby("product", sort=False).tail(6)
    recent = recent.sort_values(["date", "product"])
    for _, row in recent.iterrows():
        ret = "" if pd.isna(row["ret_next"]) else pct(row["ret_next"])
        w5 = "" if pd.isna(row["w_chg5"]) else f"{row['w_chg5']:+,.0f}"
        zc5 = "" if pd.isna(row["zc_near_chg5"]) else f"{row['zc_near_chg5']:+,.0f}"
        lines.append(
            f"| {row['day'].isoformat()} | {row['label']} | {row['beijing']} | {row['name']} | "
            f"{signed_lots(row['gs_all'])} | {zc5} | {w5} | {ret} |"
        )
    # Forward section uses the early-sample rule when it exists, and also states the full-sample leader.
    full_choice = choose(rank_rows["发布日下午"])
    lines.extend(["", "## 这条规则按发布种类", ""])
    lines.append("| 种类 | 喊次 | 方向命中 | 每次判断的平均价差 |")
    lines.append("| --- | --- | --- | --- |")
    if full_choice is not None:
        for family in ("CPI", "非农", "PPI", "GDP", "PCE", "零售", "ECI"):
            mask = event_known & featured["label"].fillna("").str.contains(family, regex=False)
            stats = score_mask(featured, mask, full_choice["key"], "ret_next")
            lines.append(
                f"| {family} | {stats['n']} | {pct(stats['hit'])} | {pct(stats['edge'])} |"
            )
        lines.append("")
        lines.append("一天里有两项发布时，这一天会同时出现在两个种类里。")
    lines.extend(["", "## 现在能做的判断", ""])
    if full_choice is not None:
        lines.append(
            f"已完成样本里，喊次不少于 {SELECT_MIN_N} 且平均价差为正、命中率最高的是：{full_choice['name']}。命中 {pct(full_choice['hit'])}，喊次 {full_choice['n']}。排在它前面的规则喊次不到 {SELECT_MIN_N}，不拿来用。"
        )
    lines.append(
        "10 月 15 日下午，沪金和沪银的高盛五日净持仓都在下降，这条规则两边都看跌。10 月 16 日主力收盘沪金较 10 月 15 日结算上涨约 1.4%，沪银上涨约 2.5%。最近这一次没有打中。"
    )
    lines.append(
        "10 月 16 日 20:30 还有美国进出口价格。会员排名文件 pm20261016 是 404，东方财富同一天也没有这两家的持仓，今晚不能用今天下午的新仓位去喊。"
    )
    lines.append(
        "下一次核心发布是 2026-10-29 8:30 ET 的三季度 GDP 初值和 9 月 PCE，北京时间 20:30，仍在夏令时，夜盘 21:00 之前。那天 15:00 的排名出来之后，用上面先定下的规则喊 10 月 30 日那个交易日。"
    )
    if fresh is not None:
        lines.append(
            f"今天已经公布的仓单：黄金 {fresh['au']:,.0f} 千克，日增减 {fresh['au_chg']:+,.0f}；白银 {fresh['ag']:,.0f} 千克，日增减 {fresh['ag_chg']:+,.0f}。席位没有同步更新，这条不单独改写持仓规则。"
        )
    if prices16:
        lines.append(
            f"10 月 16 日主力收盘：沪金 {prices16['au']['main']} {prices16['au']['close']:.2f}，沪银 {prices16['ag']['main']} {prices16['ag']['close']:.2f}。这是 10 月 15 日持仓对应的下一交易日，已经记进上面的命中率。"
        )
    lines.extend(
        [
            "",
            "全部交易日上，同一条高盛五日规则的命中率落到 54.7%，所以它是发布日下午的规则，不是每天都用的规则。中财五日变化、高盛持仓分位、四项合计都排在后半段，发布日上的命中率在四成上下。",
            "",
            "命中率是在这 18 条里排出来的。7 月以前能用来选定的喊次只有 12 次。后面 33 次更高，但集中在高盛净持仓开始连续上榜的同一段行情。不构成投资建议。",
            "",
        ]
    )
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("events", int(featured["is_event"].sum()), "known", base_n, "base_down", round(base_down, 3))
    print("full", None if full_choice is None else (full_choice["name"], full_choice["n"], round(full_choice["hit"], 3)))
    print("early", None if early_choice is None else (early_choice["name"], early_choice["n"], round(early_choice["hit"], 3)))
    print("late", late_check)
    print("pm16", bool(pm16), "px16", bool(prices16), "w16", fresh is not None)


if __name__ == "__main__":
    main()
