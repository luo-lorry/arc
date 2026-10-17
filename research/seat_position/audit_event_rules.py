#!/usr/bin/env python3
"""Correct the event-afternoon hit rates.

The earlier table pooled gold and silver and scored the next close against
today's settlement. Settlement is not a fill. This script scores the next
session's close against that session's open, which is the 21:00 night open
on contracts that have a night session, and it drops a day when the main
contract code changes. Gold and silver are ranked separately.
"""

from __future__ import annotations

import json
import math
import urllib.request
from datetime import date, datetime

import pandas as pd

from event_rules import CORE, PANEL, ROOT, RULES, WARRANTS, add_features, call_of

OUT = ROOT / "product" / "AUDIT_EVENT_RULES.md"
UA = {"User-Agent": "Mozilla/5.0"}
MIN_N = 10


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float] | tuple[None, None]:
    if n <= 0:
        return None, None
    p = hits / n
    z2 = z * z
    centre = p + z2 / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    denom = 1 + z2 / n
    return (centre - margin) / denom, (centre + margin) / denom


def pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{value * 100:.1f}%"


def bp(value: float | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{value * 10000:+.1f}bp"


def binom_p(hits: int, n: int) -> float | None:
    """Two-sided exact test of a fair coin. Not adjusted for how many rules were tried."""
    if n <= 0:
        return None
    log_target = math.lgamma(n + 1) - math.lgamma(hits + 1) - math.lgamma(n - hits + 1)
    total = 0.0
    for k in range(n + 1):
        log_ways = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
        if log_ways <= log_target + 1e-9:
            total += math.exp(log_ways - n * math.log(2))
    return min(1.0, total)


def ptxt(value: float | None) -> str:
    if value is None:
        return ""
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def fetch_oct16() -> dict[str, dict]:
    url = "https://www.shfe.com.cn/data/tradedata/future/dailydata/kx20261016.dat"
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=40) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    best: dict[str, dict] = {}
    for item in payload.get("o_curinstrument") or []:
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
                "oi": oi,
            }
    if set(best) != {"au", "ag"}:
        raise RuntimeError("kx20261016 did not contain both main contracts")
    return best


def build() -> pd.DataFrame:
    panel = pd.DataFrame(json.loads(PANEL.read_text()))
    panel["date"] = pd.to_datetime(panel["date"])
    try:
        quotes = fetch_oct16()
    except Exception:
        quotes = {
            "au": {"main": "AU2612", "open": 896.04, "close": 901.82, "settle": 898.8, "oi": 222956.0},
            "ag": {"main": "AG2612", "open": 13763.0, "close": 14029.0, "settle": 13889.0, "oi": 282779.0},
        }
    extra = []
    for product, name in (("au", "沪金"), ("ag", "沪银")):
        quote = quotes[product]
        extra.append(
            {
                "date": pd.Timestamp("2026-10-16"),
                "product": product,
                "name": name,
                "main": quote["main"],
                "open": quote["open"],
                "close": quote["close"],
                "settle": quote["settle"],
                "oi": quote["oi"],
                "gs_all": None,
                "gs_near": None,
                "zc_near": None,
                "gs_all_chg5": None,
                "zc_near_chg5": None,
            }
        )
    panel = pd.concat([panel, pd.DataFrame(extra)], ignore_index=True)
    parts = []
    for _product, chunk in panel.groupby("product", sort=False):
        chunk = chunk.sort_values("date").copy()
        chunk["ret_settle"] = chunk["close"].shift(-1) / chunk["settle"] - 1.0
        chunk["ret_open"] = chunk["close"].shift(-1) / chunk["open"].shift(-1) - 1.0
        chunk["roll"] = chunk["main"] != chunk["main"].shift(-1)
        disclosed = pd.to_numeric(chunk["gs_all"], errors="coerce")
        on_board = disclosed.notna() & disclosed.ne(0)
        chunk["gs_clean"] = on_board.rolling(6).min().eq(True)
        parts.append(chunk)
    panel = pd.concat(parts, ignore_index=True)
    warrants = pd.DataFrame(json.loads(WARRANTS.read_text()))
    warrants["date"] = pd.to_datetime(warrants["date"])
    long_rows = []
    for product, column in (("au", "au_chg5"), ("ag", "ag_chg5")):
        part = warrants[["date", column]].rename(columns={column: "w_chg5"})
        part["product"] = product
        long_rows.append(part)
    featured = add_features(panel, pd.concat(long_rows, ignore_index=True))
    labels: dict[date, list[str]] = {}
    for text, label in CORE:
        labels.setdefault(date.fromisoformat(text), []).append(label)
    trading = set(featured["date"].dt.date)
    rows = []
    for day, names in labels.items():
        if day not in trading:
            continue
        rows.append({"day": day, "label": "+".join(dict.fromkeys(names))})
    featured["day"] = featured["date"].dt.date
    featured = featured.merge(pd.DataFrame(rows), on="day", how="left")
    featured["is_event"] = featured["label"].notna()
    return featured


def called_frame(frame: pd.DataFrame, rule: str, product: str, outcome: str) -> pd.DataFrame:
    mask = (
        frame["is_event"]
        & frame["product"].eq(product)
        & frame[outcome].notna()
        & frame[outcome].ne(0)
    )
    if outcome == "ret_open":
        mask = mask & ~frame["roll"].fillna(False)
    rows = frame.loc[mask].copy()
    rows["pred"] = rows.apply(lambda row: call_of(rule, row), axis=1)
    rows = rows[rows["pred"] != 0]
    return rows


def summarize(rows: pd.DataFrame, outcome: str) -> dict:
    n = int(len(rows))
    if n == 0:
        return {"n": 0, "hits": 0, "rate": None, "lo": None, "hi": None, "early": None, "late": None, "edge": None}
    hit_flag = rows["pred"] * rows[outcome] > 0
    hits = int(hit_flag.sum())
    ordered = rows.sort_values("date")
    mid = n // 2
    early = float((ordered.iloc[:mid]["pred"] * ordered.iloc[:mid][outcome] > 0).mean()) if mid else None
    late = float((ordered.iloc[mid:]["pred"] * ordered.iloc[mid:][outcome] > 0).mean()) if n - mid else None
    lo, hi = wilson(hits, n)
    edge = float((rows["pred"] * rows[outcome]).mean())
    return {
        "n": n,
        "hits": hits,
        "rate": hits / n,
        "lo": lo,
        "hi": hi,
        "early": early,
        "late": late,
        "edge": edge,
    }


def main() -> None:
    frame = build()
    products = (("au", "沪金"), ("ag", "沪银"))
    rank_rows = []
    for product, product_name in products:
        for key, name in RULES:
            stats = summarize(called_frame(frame, key, product, "ret_open"), "ret_open")
            rank_rows.append({"product": product_name, "name": name, "key": key, **stats})
        clean = called_frame(frame, "gs_chg5", product, "ret_open")
        clean = clean[clean["gs_clean"].fillna(False)]
        stats = summarize(clean, "ret_open")
        rank_rows.append({"product": product_name, "name": "高盛五日差分，且窗口六天都在榜上", "key": "gs_clean", **stats})

    old_bits = []
    new_bits = []
    for product, product_name in products:
        old = called_frame(frame, "gs_chg5", product, "ret_settle")
        new = called_frame(frame, "gs_chg5", product, "ret_open")
        old_bits.append(summarize(old, "ret_settle") | {"product": product_name})
        new_bits.append(summarize(new, "ret_open") | {"product": product_name})

    def cell(stats: dict) -> str:
        if stats["n"] == 0:
            return "0"
        return f"{stats['hits']}/{stats['n']}"

    calls = pd.concat(
        [called_frame(frame, "gs_chg5", product, "ret_open") for product, _name in products],
        ignore_index=True,
    )
    agree_n = agree_hits = split_n = single_n = single_hits = mixed_same = 0
    for _day, grp in calls.groupby("day"):
        if len(grp) == 1:
            single_n += 1
            row = grp.iloc[0]
            single_hits += int(row["pred"] * row["ret_open"] > 0)
            continue
        same_sign = grp["pred"].nunique() == 1
        hits = grp["pred"] * grp["ret_open"] > 0
        if same_sign and bool(hits.all()):
            agree_n += 1
            agree_hits += 1
        elif same_sign and bool((~hits).all()):
            agree_n += 1
        elif same_sign:
            mixed_same += 1
        else:
            split_n += 1
    agree_lo, agree_hi = wilson(agree_hits, agree_n)

    dust = calls[(calls["product"] == "au") & (calls["ret_open"].abs() < 0.0005)]
    dust_note = "、".join(
        f"{row.day.isoformat()} {bp(row.ret_open)}" for row in dust.sort_values("day").itertuples()
    )
    gold_keep = calls[(calls["product"] == "au") & (calls["ret_open"].abs() >= 0.0005)]
    gold_dust_hits = int((gold_keep["pred"] * gold_keep["ret_open"] > 0).sum())

    def day_quote(product: str, day: str) -> pd.Series:
        hit = frame[(frame["product"] == product) & (frame["date"] == pd.Timestamp(day))]
        return hit.iloc[0]

    au15, au16 = day_quote("au", "2026-10-15"), day_quote("au", "2026-10-16")
    ag15, ag16 = day_quote("ag", "2026-10-15"), day_quote("ag", "2026-10-16")

    lines = [
        "# 发布日规则的复核",
        "",
        "上一张表里「高盛五日净持仓下降看跌，上升看涨」是 45 次、命中 68.9%。这个数字按原代码可以复算出来。计分用的是当日结算价到下一交易日收盘，沪金和沪银各算一次，零售、GDP、PCE、ECI 和 CPI、非农、PPI 放在同一张表里。",
        "",
        "修正后的计分是：信号仍是当日公布净持仓减去五个交易日前的公布净持仓，下降看跌、上升看涨，变化为 0 不喊。结果改成下一交易日收盘除以该日开盘再减 1。有夜盘的合约，日行情开盘价是前一晚 21:00 的第一笔。主力合约代码变了的日子去掉。沪金、沪银分开报。同一下午有两项发布时，总喊次里只记一次。未上榜的席位在原始数据里记为 0，和真正的零持仓分不开；另有一列要求这五天变化的六个交易日都在榜上。",
        "",
        "## 45 次、68.9% 怎么来的",
        "",
        "| 口径 | 沪金 | 沪银 | 合计 | 命中 |",
        "| --- | --- | --- | --- | --- |",
    ]
    old_hits = sum(item["hits"] for item in old_bits)
    old_n = sum(item["n"] for item in old_bits)
    new_hits = sum(item["hits"] for item in new_bits)
    new_n = sum(item["n"] for item in new_bits)
    lines.append(
        f"| 结算价到下一收盘，换月也保留 | {cell(old_bits[0])} | {cell(old_bits[1])} | {old_hits}/{old_n} | {pct(old_hits / old_n)} |"
    )
    lines.append(
        f"| 夜盘开盘到下一收盘，去掉换月 | {cell(new_bits[0])} | {cell(new_bits[1])} | {new_hits}/{new_n} | {pct(new_hits / new_n)} |"
    )
    new_lo, new_hi = wilson(new_hits, new_n)
    lines.append("")
    lines.append(
        f"合计 {new_hits}/{new_n} 的 95% 区间是 {pct(new_lo)} 到 {pct(new_hi)}，双侧随机概率 {ptxt(binom_p(new_hits, new_n))}。"
        "区间盖住 50%。沪金和沪银经常是同一个下午，两次结果不是独立样本，把它们相加会把同一条消息算成更多证据。"
    )
    lines.append("")
    lines.append("| 品种 | 喊次 | 对 | 命中 | 95%区间 | 前半段 | 后半段 | 平均方向收益 | 双侧随机概率 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for item in new_bits:
        lines.append(
            f"| {item['product']} | {item['n']} | {item['hits']} | {pct(item['rate'])} | "
            f"{pct(item['lo'])}–{pct(item['hi'])} | {pct(item['early'])} | {pct(item['late'])} | {bp(item['edge'])} | "
            f"{ptxt(binom_p(item['hits'], item['n']))} |"
        )
    lines.append("")
    lines.append(
        f"前半段、后半段是按日期排序后对半切开，奇数次时后半段多一次。"
        f"沪金 {cell(new_bits[0])}，方向收益 {bp(new_bits[0]['edge'])}，后半段 {pct(new_bits[0]['late'])}。"
        f"沪银 {cell(new_bits[1])}，方向收益 {bp(new_bits[1]['edge'])}，95% 下限 {pct(new_bits[1]['lo'])}，后半段 {pct(new_bits[1]['late'])}。"
        "沪银这 20 次都在 2026 年 5 月高盛开始上榜之后。双侧随机概率没有把同时试过的规则数扣掉。"
    )
    lines.append("")
    lines.append(
        f"同一下午两个品种方向相同、并且两边同对或同错的有 {agree_n} 个下午，其中同对 {agree_hits} 个、同错 {agree_n - agree_hits} 个，"
        f"95% 区间 {pct(agree_lo)} 到 {pct(agree_hi)}。"
        f"方向相同但一边对、一边错的有 {mixed_same} 个下午。方向相反的有 {split_n} 个下午。"
        f"只喊了一个品种的有 {single_n} 次，打中 {single_hits} 次。"
        "把同对同错的下午收成一次之后，区间仍然盖住 50%。"
    )
    lines.append("")
    lines.append(
        f"开盘到收盘不到 5bp 的沪金有 {len(dust)} 次（{dust_note}）。这两次仍按符号计分。"
        f"去掉之后沪金是 {gold_dust_hits}/{len(gold_keep)}。"
        "原表按发布种类把 GDP 和 PCE 各写了一行，那两行是同一批喊叫。总的 45 次没有把这一天算两遍，种类表算了两遍。"
    )
    lines.append("")
    au_vs = au16["close"] / au15["settle"] - 1.0
    ag_vs = ag16["close"] / ag15["settle"] - 1.0
    au_open = au16["close"] / au16["open"] - 1.0
    ag_open = ag16["close"] / ag16["open"] - 1.0
    lines.append(
        f"10 月 15 日两边的五日净持仓都在下降，规则看跌。"
        f"10 月 16 日上期所日行情，沪金 {au16['main']} 开盘 {au16['open']:.2f}、收盘 {au16['close']:.2f}，"
        f"相对 10 月 15 日结算 {au15['settle']:.2f} 上涨 {au_vs * 100:.4f}%，相对夜盘开盘上涨 {au_open * 100:.4f}%。"
        f"沪银 {ag16['main']} 开盘 {ag16['open']:.0f}、收盘 {ag16['close']:.0f}，"
        f"相对结算 {ag15['settle']:.0f} 上涨 {ag_vs * 100:.4f}%，相对夜盘开盘上涨 {ag_open * 100:.4f}%。"
        "按结算价和按夜盘开盘价，这一次都是错的。"
    )

    ranked = [row for row in rank_rows if row["n"] >= MIN_N]
    ranked.sort(key=lambda row: (-(row["rate"] or -1), -row["n"], row["product"], row["name"]))
    by_floor = sorted(
        ranked,
        key=lambda row: (
            -(row["lo"] or -1),
            -(min(row["early"] or 0, row["late"] or 0)),
            -row["n"],
            row["name"],
        ),
    )
    floor_rank = {id(row): rank for rank, row in enumerate(by_floor, start=1)}
    lines.extend(["", "## 按夜盘开盘到收盘的命中率排序", ""])
    lines.append(
        f"每条规则仍是原来的方向：高盛五日净持仓下降看跌、上升看涨；仓单五日增加看跌、减少看涨。只收喊次不少于 {MIN_N} 的格子。表按命中率排序。「按下限」是同一批格子按 95% 下限、较差半段、喊次重排后的名次。下限低于 50%，或者后半段掉到 50% 以下，这条命中率还站不住。"
    )
    lines.append("")
    lines.append("| 排名 | 按下限 | 品种 | 规则 | 喊次 | 对 | 命中 | 95%下限 | 前半段 | 后半段 | 平均方向收益 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for rank, row in enumerate(ranked, start=1):
        lines.append(
            f"| {rank} | {floor_rank[id(row)]} | {row['product']} | {row['name']} | {row['n']} | {row['hits']} | {pct(row['rate'])} | "
            f"{pct(row['lo'])} | {pct(row['early'])} | {pct(row['late'])} | {bp(row['edge'])} |"
        )

    hold = [
        row for row in ranked
        if row["lo"] is not None and row["lo"] > 0.5 and row["early"] is not None and row["early"] >= 0.5 and row["late"] >= 0.5
    ]
    lines.append("")
    if hold:
        names = "、".join(f"{row['product']}{row['name']}（{row['hits']}/{row['n']}）" for row in hold)
        lines.append(f"95% 下限高于 50%、并且前后半段都不低于 50% 的，只有：{names}。")
    else:
        lines.append("没有任何一条同时满足 95% 下限高于 50%、前后半段都不低于 50%。")
    lines.append("这些格子来自同一批发布日上的十多条规则。单看其中一个低于 5% 的随机概率，没有把同时试过的规则数算进去。")

    # Event-conditional candidates. Same sign as the pooled rules, plus the opposite warrant sign.
    lines.extend(["", "## 按发布种类切开", ""])
    lines.append("种类表里，标签含有该种类的下午才进入这一行。GDP+PCE 进入 GDP 这一行，这张表没有再单列 PCE。JOLTS 通常在美东 10:00，夏令时夜盘 21:00 已经开过一小时，不放进这张表。")
    lines.append("")
    lines.append("| 种类 | 品种 | 规则 | 喊次 | 对 | 命中 | 95%下限 | 前半段 | 后半段 | 平均方向收益 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    slices = []
    specs = [
        ("PPI", "au", "沪金", "warrant_sign", "仓单五日增加看跌"),
        ("CPI", "ag", "沪银", "warrant_sign", "仓单五日增加看跌"),
        ("CPI", "ag", "沪银", "warrant_up", "仓单五日增加看涨"),
        ("CPI", "ag", "沪银", "oi_up", "总持仓量五日增加看涨"),
        ("非农", "ag", "沪银", "zc_chg5", "中财五日下降看跌，上升看涨"),
        ("GDP", "au", "沪金", "px_up", "五日价格动量"),
        ("CPI", "au", "沪金", "gs_chg5", "高盛五日差分"),
        ("CPI", "ag", "沪银", "gs_chg5", "高盛五日差分"),
        ("非农", "au", "沪金", "gs_chg5", "高盛五日差分"),
        ("非农", "ag", "沪银", "gs_chg5", "高盛五日差分"),
        ("PPI", "au", "沪金", "gs_chg5", "高盛五日差分"),
        ("PPI", "ag", "沪银", "gs_chg5", "高盛五日差分"),
    ]
    for family, product, product_name, key, name in specs:
        rows = frame[
            frame["is_event"]
            & frame["product"].eq(product)
            & frame["label"].fillna("").str.contains(family, regex=False)
            & frame["ret_open"].notna()
            & frame["ret_open"].ne(0)
            & ~frame["roll"].fillna(False)
        ].copy()
        if key == "warrant_up":
            rows["pred"] = rows["w_chg5"].map(lambda value: 0 if pd.isna(value) or value == 0 else (1 if value > 0 else -1))
        elif key == "oi_up":
            rows["pred"] = rows["oi_chg5"].map(lambda value: 0 if pd.isna(value) or value == 0 else (1 if value > 0 else -1))
        elif key == "px_up":
            rows["pred"] = rows["px_chg5"].map(lambda value: 0 if pd.isna(value) or value == 0 else (1 if value > 0 else -1))
        else:
            rows["pred"] = rows.apply(lambda row: call_of(key, row), axis=1)
        rows = rows[rows["pred"] != 0]
        stats = summarize(rows, "ret_open")
        slices.append({"family": family, "product": product_name, "name": name, **stats})
    slices.sort(key=lambda row: (-(row["rate"] or -1), -row["n"], row["family"]))
    for row in slices:
        lines.append(
            f"| {row['family']} | {row['product']} | {row['name']} | {row['n']} | {row['hits']} | {pct(row['rate'])} | "
            f"{pct(row['lo'])} | {pct(row['early'])} | {pct(row['late'])} | {bp(row['edge'])} |"
        )

    def sliced(family: str, product: str, name: str) -> dict:
        for row in slices:
            if row["family"] == family and row["product"] == product and row["name"] == name:
                return row
        raise KeyError((family, product, name))

    cpi_w = sliced("CPI", "沪银", "仓单五日增加看跌")
    ppi_w = sliced("PPI", "沪金", "仓单五日增加看跌")
    cpi_w_up = sliced("CPI", "沪银", "仓单五日增加看涨")
    cpi_oi = sliced("CPI", "沪银", "总持仓量五日增加看涨")
    nfp_zc = sliced("非农", "沪银", "中财五日下降看跌，上升看涨")
    gdp_px = sliced("GDP", "沪金", "五日价格动量")
    lines.append("")
    lines.append(
        f"CPI 日沪银仓单增加看跌是 {cpi_w['hits']}/{cpi_w['n']}，命中 {pct(cpi_w['rate'])}，"
        f"前后半段 {pct(cpi_w['early'])} 和 {pct(cpi_w['late'])}，95% 下限 {pct(cpi_w['lo'])}，方向收益 {bp(cpi_w['edge'])}。"
        f"PPI 日沪金仓单增加看跌是 {ppi_w['hits']}/{ppi_w['n']}，命中 {pct(ppi_w['rate'])}，95% 下限 {pct(ppi_w['lo'])}，方向收益 {bp(ppi_w['edge'])}。"
        f"仓单增加改成看涨之后，CPI 日沪银是 {cpi_w_up['hits']}/{cpi_w_up['n']}，命中 {pct(cpi_w_up['rate'])}。"
        f"CPI 日沪银总持仓量五日增加看涨是 {cpi_oi['hits']}/{cpi_oi['n']}。"
        f"非农日沪银中财五日是 {nfp_zc['hits']}/{nfp_zc['n']}。"
        f"GDP 日沪金五日价格动量是 {gdp_px['hits']}/{gdp_px['n']}，方向收益 {bp(gdp_px['edge'])}。"
    )
    warrant_all = next(row for row in ranked if row["product"] == "沪银" and row["key"] == "warrant_sign")
    lines.append("")
    lines.append(
        f"按命中率，第一是{ranked[0]['product']}{ranked[0]['name']}。"
        f"按 95% 下限，第一是{by_floor[0]['product']}{by_floor[0]['name']}（{by_floor[0]['hits']}/{by_floor[0]['n']}，下限 {pct(by_floor[0]['lo'])}），"
        f"第二是{by_floor[1]['product']}{by_floor[1]['name']}（{by_floor[1]['hits']}/{by_floor[1]['n']}，下限 {pct(by_floor[1]['lo'])}）。"
        f"CPI 日沪银仓单是 {cpi_w['hits']}/{cpi_w['n']}，下限 {pct(cpi_w['lo'])}。"
        f"同一条仓单规则放到全部发布日上是 {warrant_all['hits']}/{warrant_all['n']}，下限 {pct(warrant_all['lo'])}，前后半段 {pct(warrant_all['early'])} 和 {pct(warrant_all['late'])}。"
    )
    lines.append("")
    lines.append(
        "本面板从 2025-01-02 到 2026-10-16。下面把另一份复核里的计数和本面板并排放上。另一份复核的分子分母在这份数据里对不上，表里留着，避免把两套样本写成同一条结果。"
    )
    lines.append("")
    lines.append("| 规则 | 另一份复核 | 本面板 |")
    lines.append("| --- | --- | --- |")
    lines.append(f"| 高盛五日，沪金 | 9/15，60.0%，方向收益约 -21bp | {cell(new_bits[0])}，{pct(new_bits[0]['rate'])}，{bp(new_bits[0]['edge'])} |")
    lines.append(f"| 高盛五日，沪银 | 6/12，50.0%，方向收益约 +68.5bp | {cell(new_bits[1])}，{pct(new_bits[1]['rate'])}，{bp(new_bits[1]['edge'])} |")
    lines.append(
        f"| PPI 日沪金仓单五日增加看跌 | 20/30，66.7%，前后半 80.0% 和 53.3% | "
        f"{ppi_w['hits']}/{ppi_w['n']}，{pct(ppi_w['rate'])}，前后半 {pct(ppi_w['early'])} 和 {pct(ppi_w['late'])} |"
    )
    lines.append(
        f"| CPI 日沪银仓单五日增加看跌 | 38/58，65.5%，95% 下限 52.7% | "
        f"{cpi_w['hits']}/{cpi_w['n']}，{pct(cpi_w['rate'])}，95% 下限 {pct(cpi_w['lo'])} |"
    )
    lines.append("| JOLTS 日沪银五日价格动量 | 35/54，64.8% | 未列入。JOLTS 通常在美东 10:00，夏令时是北京时间 22:00，夜盘 21:00 已经开过一小时 |")
    lines.append(
        f"| CPI 日沪银总持仓量五日增加看涨 | 38/61，62.3% | {cpi_oi['hits']}/{cpi_oi['n']}，{pct(cpi_oi['rate'])}，95% 下限 {pct(cpi_oi['lo'])} |"
    )
    lines.append(
        f"| 非农日沪银中财五日 | 23/37，62.2%，95% 下限 46.1% | {nfp_zc['hits']}/{nfp_zc['n']}，{pct(nfp_zc['rate'])} |"
    )
    lines.append(
        f"| GDP 日沪金五日价格动量 | 38/62，61.3%，夜盘方向收益约 +2bp | {gdp_px['hits']}/{gdp_px['n']}，{pct(gdp_px['rate'])}，{bp(gdp_px['edge'])} |"
    )
    lines.append("")
    lines.append(
        "中财沪银近月在这份面板里很多日子是 0。非农日、五日变化不为 0 的喊次只有上面这一行，到不了 37 次。"
        "仓单和价格从 2025-01-02 起算，CPI 日沪银仓单到不了 58 次。更长历史上的 38/58 要用更早的仓单才能复算，这里没有。"
    )

    lines.extend(["", "## 高盛五日差分，夜盘开盘口径的每一次", ""])
    lines.append("| 发布日 | 品种 | 发布 | 五日变化 | 方向 | 开盘到收盘 | 结果 | 换月已剔除 | 窗口都在榜 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    detail = frame[frame["is_event"] & frame["gs_all_chg5"].notna() & frame["gs_all_chg5"].ne(0) & frame["ret_open"].notna()].copy()
    detail = detail.sort_values(["day", "product"])
    for _, row in detail.iterrows():
        pred = -1 if row["gs_all_chg5"] < 0 else 1
        kept = not bool(row["roll"]) and row["ret_open"] != 0
        result = ""
        if kept:
            result = "对" if pred * row["ret_open"] > 0 else "错"
        lines.append(
            f"| {row['day'].isoformat()} | {row['name']} | {row['label']} | {row['gs_all_chg5']:+,.0f} | "
            f"{'看跌' if pred < 0 else '看涨'} | {bp(row['ret_open'])} | {result} | "
            f"{'是' if row['roll'] else ''} | {'是' if row['gs_clean'] else ''} |"
        )
    lines.append("")
    lines.append(
        f"数字由 `research/seat_position/audit_event_rules.py` 从持仓面板、仓单和上期所 kx20261016 算出。生成于 {datetime.now():%Y-%m-%d}。"
    )
    lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"old {old_hits}/{old_n} new {new_hits}/{new_n}")
    print("top", ranked[0]["product"], ranked[0]["name"], ranked[0]["hits"], ranked[0]["n"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
