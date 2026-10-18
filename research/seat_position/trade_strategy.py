#!/usr/bin/env python3
"""Full corrected leaderboard, and an event-day silver book.

Signals use only the 15:00 Beijing ranking and the same day's warrant report.
The fill is the next session's open, which is the 21:00 night open, and the
exit is that session's close. A day whose main-contract code changes is skipped.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from audit_event_rules import binom_p, bp, build, called_frame, pct, ptxt, summarize, wilson
from event_rules import RULES, release_clock

OUT_PATH = __import__("pathlib").Path(__file__).resolve().parent / "product" / "TRADE_STRATEGY.md"
AG_KG = 15
# Two published SHFE schedules, each side, plus one tick (1 yuan/kg) each side.
# June and December contracts, and other months once they enter the pre-delivery window, use the higher open fee.
FEE_LOW = 0.00001  # 万分之0.1 per side, ordinary non-6/12 month
FEE_HIGH_OPEN = 0.00005  # 万分之0.5
FEE_HIGH_CLOSE = 0.00025  # 万分之2.5 close-today, the elevated schedule in the 2026-01-07 notice


RULE_TEXT = [
    (
        "gs_sign",
        "高盛公布净持仓为负看跌，为正看涨",
        "上期所当日会员排名里，高盛在该品种前 20 名多头手数减去前 20 名空头手数，记为 gs_all。小于 0 看跌，大于 0 看涨，等于 0 不喊。未上榜也记成 0，所以这一条把「不在榜上」和「多空刚好抵消」看成同一种情况。",
    ),
    (
        "gs_short_only",
        "只在高盛公布净持仓为负时看跌",
        "同一份 gs_all。只有小于 0 才看跌，大于 0 或等于 0 都不喊。它是单边规则，行情上涨时没有对冲的看涨喊次。",
    ),
    (
        "gs_tail",
        "高盛净持仓低于自身10%分位看跌，高于90%分位看涨",
        "分位数只用当天之前的 gs_all，至少要有 40 个交易日。当天小于等于 10% 分位看跌，大于等于 90% 分位看涨。历史里大量的 0 会把 10% 分位压在 0 附近，于是当天 gs_all 也是 0 时同样满足「小于等于 10% 分位」，空榜日会被喊成看跌。",
    ),
    (
        "gs_tail_short",
        "只在高盛净持仓低于自身10%分位时看跌",
        "和上一条用同一个 10% 分位，只保留看跌的一侧。空榜日记成 0 时，同样会落入这一侧。",
    ),
    (
        "gs_chg5",
        "高盛五日净持仓下降看跌，上升看涨",
        "gs_all_chg5 = 今日 gs_all − 五个交易日前的 gs_all。小于 0 看跌，大于 0 看涨，差分为 0 不喊。窗口两端只要有一天是空榜的 0，这个差分就是进出榜造成的跳变，不是五天里的真实加减仓。",
    ),
    (
        "gs_add_short",
        "高盛已经净空，且当天净持仓继续下降，看跌",
        "gs_all < 0，并且 gs_all 相对前一个交易日的差分也小于 0，才看跌。两个条件缺一个就不喊。用的是一天的变化，不是五日变化。",
    ),
    (
        "gs_near_sign",
        "高盛当月加近月净持仓为负看跌，为正看涨",
        "只加总高盛在当月合约和近月合约上、并且出现在前 20 名里的多空差，记为 gs_near。符号规则和全月净持仓相同。远月上的仓位不进这个数。",
    ),
    (
        "zc_chg5",
        "中财当月加近月五日变化为负看跌，为正看涨",
        "中财期货在当月加近月上的可见净持仓，今日减去五个交易日前，记为 zc_near_chg5。小于 0 看跌，大于 0 看涨。中财在这份面板里很多日子是 0，差分为 0 的日子不喊。",
    ),
    (
        "zc_tail",
        "中财近月五日变化低于自身10%分位看跌，高于90%分位看涨",
        "分位数用的是当天之前的 zc_near_chg5，至少 40 个非空观察。五日变化落到自身 10% 分位或更低就看跌，落到 90% 分位或更高就看涨。中间的变化不喊。",
    ),
    (
        "warrant_sign",
        "仓单五日增加看跌，减少看涨",
        "仓单日报里该品种的库存千克数，今日减去五个交易日前，记为 w_chg5。增加看跌，减少看涨，变化为 0 不喊。黄金用 au_kg，白银用 ag_kg。这条把库存增加理解成可交割供给变多。",
    ),
    (
        "warrant_tail",
        "仓单五日变化进入增加的10%极端看跌，进入减少的10%极端看涨",
        "用当天之前的 w_chg5 做 10% 和 90% 分位。五日变化大于等于 90% 分位，也就是增加特别多，看跌。小于等于 10% 分位，也就是减少特别多，看涨。普通增减不喊。",
    ),
    (
        "price_oi",
        "五日价跌且持仓增加看跌，价涨且持仓增加看涨",
        "主力收盘价的五日差分 px_chg5，主力持仓量的五日差分 oi_chg5。持仓量没有增加就不喊。持仓量增加且价格下跌看跌，持仓量增加且价格上涨看涨。价格用的是收盘价，不是夜盘开盘价。",
    ),
    (
        "combo1",
        "高盛分位、中财分位、仓单分位、价仓四项合计达到1就喊方向",
        "四项各贡献 −1、0 或 +1，加总后绝对值达到 1 就按合计的符号喊。高盛净持仓落在低分位减 1、高分位加 1。中财五日变化低分位减 1、高分位加 1。仓单五日变化低分位加 1、高分位减 1，库存大减算正分。价跌且持仓增加减 1，价涨且持仓增加加 1。",
    ),
    (
        "combo2",
        "上面四项合计达到2才喊方向",
        "和上一条是同一套四项打分。合计小于等于 −2 才看跌，大于等于 +2 才看涨。只有一项有方向时不喊。",
    ),
    (
        "gs_and_zc_cut",
        "高盛净空，且中财近月五日在减少，看跌",
        "gs_all < 0，并且 zc_near_chg5 < 0，两个都成立才看跌。没有看涨的一侧。",
    ),
    (
        "gs_tail_and_zc_cut",
        "高盛低于自身10%分位，且中财近月五日在减少，看跌",
        "gs_all 小于等于自身 10% 分位，并且 zc_near_chg5 < 0，才看跌。分位数同样把空榜的 0 算进历史。",
    ),
    (
        "gs_short_warrant_up",
        "高盛净空，且仓单五日增加，看跌",
        "gs_all < 0，并且 w_chg5 > 0，才看跌。",
    ),
    (
        "gs_short_warrant_down",
        "高盛净空，且仓单五日减少，仍看跌",
        "gs_all < 0，并且 w_chg5 < 0，才看跌。仓单在减少，这条仍然只因为高盛净空而看跌。",
    ),
    (
        "gs_clean",
        "高盛五日差分，且窗口六天都在榜上",
        "方向和五日差分相同，但今日往前数六个交易日的 gs_all 必须都不是 0。空榜造成的跳变不喊。榜内某一天仓位大幅变化，这一条仍然会喊。",
    ),
]


def pred_of(rule: str, row: pd.Series) -> int:
    if rule == "gs_clean":
        if not bool(row["gs_clean"]):
            return 0
        rule = "gs_chg5"
    from event_rules import call_of

    return call_of(rule, row)


def event_calls(frame: pd.DataFrame, product: str, rule: str) -> pd.DataFrame:
    if rule == "gs_clean":
        rows = called_frame(frame, "gs_chg5", product, "ret_open")
        return rows[rows["gs_clean"].fillna(False)].copy()
    return called_frame(frame, rule, product, "ret_open")


def everyday_calls(frame: pd.DataFrame, product: str, rule: str) -> pd.DataFrame:
    mask = frame["product"].eq(product) & frame["ret_open"].notna() & frame["ret_open"].ne(0) & ~frame["roll"].fillna(False)
    rows = frame.loc[mask].copy()
    rows["pred"] = rows.apply(lambda row: pred_of(rule, row), axis=1)
    return rows[rows["pred"] != 0].copy()


def yuan_cost(open_px: float, schedule: str) -> float:
    notional = open_px * AG_KG
    slip = 2 * AG_KG  # one tick each side, tick is 1 yuan per kg
    if schedule == "low":
        fee = notional * FEE_LOW * 2
    else:
        fee = notional * FEE_HIGH_OPEN + notional * FEE_HIGH_CLOSE
    return fee + slip


def attach_fills(frame: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Entry is the next bar's open; exit is the next bar's close."""
    nxt = frame[["date", "product", "open", "close", "main"]].copy()
    nxt["date"] = nxt["date"] - pd.Timedelta(0)
    pieces = []
    for product, chunk in frame.groupby("product", sort=False):
        chunk = chunk.sort_values("date").copy()
        chunk["entry"] = chunk["open"].shift(-1)
        chunk["exit"] = chunk["close"].shift(-1)
        chunk["exit_main"] = chunk["main"].shift(-1)
        pieces.append(chunk[["date", "product", "entry", "exit", "exit_main"]])
    fills = pd.concat(pieces, ignore_index=True)
    merged = rows.merge(fills, on=["date", "product"], how="left")
    merged["gross_yuan"] = merged["pred"] * (merged["exit"] - merged["entry"]) * AG_KG
    merged["cost_low"] = merged["entry"].map(lambda px: yuan_cost(px, "low") if pd.notna(px) else pd.NA)
    merged["cost_high"] = merged["entry"].map(lambda px: yuan_cost(px, "high") if pd.notna(px) else pd.NA)
    merged["net_low"] = merged["gross_yuan"] - merged["cost_low"]
    merged["net_high"] = merged["gross_yuan"] - merged["cost_high"]
    merged["season"] = merged["day"].map(lambda day: "夏令时" if release_clock(day).hour < 21 else "冬令时")
    return merged


def curve_stats(pnl: pd.Series) -> dict:
    values = pd.to_numeric(pnl, errors="coerce").dropna()
    n = int(len(values))
    if n == 0:
        return {"n": 0, "total": 0.0, "mean": None, "t": None, "dd": None, "worst": None}
    equity = pd.concat([pd.Series([0.0]), values.cumsum()], ignore_index=True)
    drawdown = float((equity - equity.cummax()).min())
    std = float(values.std(ddof=1)) if n > 1 else float("nan")
    t_stat = None if not std or pd.isna(std) or std == 0 else float(values.mean() / std * (n ** 0.5))
    return {
        "n": n,
        "total": float(values.sum()),
        "mean": float(values.mean()),
        "t": t_stat,
        "dd": drawdown,
        "worst": float(values.min()),
    }


def money(value: float | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{value:,.0f}"


def num(value: float | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{value:.2f}"


def agreement_book(frame: pd.DataFrame, mode: str) -> pd.DataFrame:
    warrant = event_calls(frame, "ag", "warrant_sign").set_index("day", drop=False)
    goldman = event_calls(frame, "ag", "gs_chg5").set_index("day", drop=False)
    rows = []
    for day in sorted(set(warrant.index) | set(goldman.index)):
        w_pred = int(warrant.at[day, "pred"]) if day in warrant.index else 0
        g_pred = int(goldman.at[day, "pred"]) if day in goldman.index else 0
        if mode == "agree":
            pred = w_pred if w_pred != 0 and w_pred == g_pred else 0
        elif mode == "either":
            if w_pred != 0 and g_pred != 0 and w_pred != g_pred:
                pred = 0
            else:
                pred = w_pred or g_pred
        else:
            raise KeyError(mode)
        if pred == 0:
            continue
        base = warrant.loc[day] if day in warrant.index else goldman.loc[day]
        item = base.to_dict()
        item["pred"] = pred
        item["product"] = "ag"
        if day in warrant.index:
            item["w_chg5"] = warrant.at[day, "w_chg5"]
        if day in goldman.index:
            item["gs_all_chg5"] = goldman.at[day, "gs_all_chg5"]
        rows.append(item)
    if not rows:
        return pd.DataFrame(columns=["day", "pred", "ret_open", "date", "product"])
    return pd.DataFrame(rows)


def main() -> None:
    frame = build()
    products = (("au", "沪金"), ("ag", "沪银"))
    rank_rows = []
    for product, product_name in products:
        for key, _name in RULES:
            stats = summarize(event_calls(frame, product, key), "ret_open")
            rank_rows.append({"product": product_name, "key": key, **stats})
        stats = summarize(event_calls(frame, product, "gs_clean"), "ret_open")
        rank_rows.append({"product": product_name, "key": "gs_clean", **stats})
    names = {key: name for key, name, _text in RULE_TEXT}
    for row in rank_rows:
        row["name"] = names[row["key"]]
    ranked = [row for row in rank_rows if row["n"] > 0]
    ranked.sort(key=lambda row: (-(row["rate"] or -1), -row["n"], row["product"], row["name"]))
    by_floor = sorted(
        [row for row in ranked if row["n"] >= 10],
        key=lambda row: (-(row["lo"] or -1), -(min(row["early"] or 0, row["late"] or 0)), -row["n"], row["name"]),
    )
    floor_rank = {id(row): i for i, row in enumerate(by_floor, start=1)}

    books = {
        "仓单单独": attach_fills(frame, event_calls(frame, "ag", "warrant_sign")),
        "高盛五日单独": attach_fills(frame, event_calls(frame, "ag", "gs_chg5")),
        "两条同向才做": attach_fills(frame, agreement_book(frame, "agree")),
        "有一边就做，冲突不做": attach_fills(frame, agreement_book(frame, "either")),
    }
    everyday = {
        "仓单，全部交易日": attach_fills(frame, everyday_calls(frame, "ag", "warrant_sign")),
        "高盛五日，全部交易日": attach_fills(frame, everyday_calls(frame, "ag", "gs_chg5")),
    }

    lines = [
        "# 完整排行榜和沪银发布日策略",
        "",
        "计分和复核用的是同一套。信号只用发布日当天 15:00 已经出来的上期所会员排名，以及同一天的仓单日报。结果是下一交易日收盘相对该日开盘。有夜盘的合约，开盘价是前一晚 21:00 的第一笔。主力合约代码变了的日子去掉。沪金、沪银分开。同一下午有两项美国数据时只记一次。",
        "",
        "排名里的命中率来自 19 条规则、两个品种。下面用来做交易的，只留下同时满足三件事的格子：喊次不少于 10，95% 下限高于 50%，前半段和后半段都不低于 50%。满足的是沪银高盛五日，和沪银仓单五日增加看跌、减少看涨。",
        "",
        "## 数据",
        "",
        "会员排名是上期所 `pm` 日文件里的前 20 名多头、前 20 名空头。高盛全月净持仓 gs_all 等于可见多头手数减去可见空头手数。高盛当月加近月 gs_near、中财当月加近月 zc_near 用同样的减法，只保留当月和近月。不在前 20 名就记 0，文件里看不出这是真的零还是没上榜。五日变化是今日这个数减去五个交易日前的同一个数。",
        "",
        "仓单是上期所仓单日报的千克数。黄金用 au_kg，白银用 ag_kg。w_chg5 是今日千克数减去五个交易日前的千克数。",
        "",
        "价格和持仓量取当天持仓量最大的合约。px_chg5 是主力收盘价的五日差分，oi_chg5 是主力持仓量的五日差分。分位数只用当天之前的历史，不满 40 个交易日不喊。发布日名单是 BLS 的 CPI、非农、PPI、ECI，Census 从 2025 年 11 月起的零售销售，以及 BEA 页面上 2026 年 8:30 ET 的 GDP 和 PCE。",
        "",
        "样本从 2025-01-02 到 2026-10-16。10 月 16 日的开盘、收盘用来给 10 月 15 日的信号记结果，不拿来制造 10 月 16 日的信号。",
        "",
        "## 完整排行榜",
        "",
        "先排喊次不少于 10 的格子，按命中率从高到低。按下限是同一批格子的另一个名次：先看 95% 下限，再看较差的半段，再看喊次。半段按日期排序后对半切，奇数次时后半段多一次。方向收益是每次判断的方向乘以开盘到收盘的收益率，看错的次数也算进去。喊次不到 10 的格子放在后面，一次喊叫打中就是 100%，不拿来排第一。",
        "",
        "| 排名 | 按下限 | 品种 | 规则 | 喊次 | 对 | 命中 | 95%下限 | 前半段 | 后半段 | 平均方向收益 | 双侧随机概率 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    main_board = [row for row in ranked if row["n"] >= 10]
    short_board = [row for row in ranked if row["n"] < 10]
    for rank, row in enumerate(main_board, start=1):
        lines.append(
            f"| {rank} | {floor_rank[id(row)]} | {row['product']} | {row['name']} | {row['n']} | {row['hits']} | {pct(row['rate'])} | "
            f"{pct(row['lo'])} | {pct(row['early'])} | {pct(row['late'])} | {bp(row['edge'])} | {ptxt(binom_p(row['hits'], row['n']))} |"
        )
    lines.extend([
        "",
        "喊次不到 10 的格子：",
        "",
        "| 品种 | 规则 | 喊次 | 对 | 命中 | 95%下限 | 前半段 | 后半段 | 平均方向收益 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ])
    for row in short_board:
        lines.append(
            f"| {row['product']} | {row['name']} | {row['n']} | {row['hits']} | {pct(row['rate'])} | "
            f"{pct(row['lo'])} | {pct(row['early'])} | {pct(row['late'])} | {bp(row['edge'])} |"
        )
    lines.append("")
    lines.append(
        "双侧随机概率按每一次喊叫独立、公平硬币来算，没有把 19 条规则乘两个品种的尝试次数扣掉。沪金和沪银经常是同一个下午，两行并不是两份独立证据。"
    )
    hold = [
        row for row in ranked
        if row["n"] >= 10 and row["lo"] is not None and row["lo"] > 0.5 and (row["early"] or 0) >= 0.5 and (row["late"] or 0) >= 0.5
    ]
    lines.append("")
    if hold:
        text = "、".join(f"{row['product']}{row['name']}（{row['hits']}/{row['n']}，{bp(row['edge'])}）" for row in hold)
        lines.append(f"喊次不少于 10、95% 下限高于 50%、前后半段都不低于 50% 的，是：{text}。")
    lines.append("")
    lines.append("## 每一条规则")
    lines.append("")
    for key, name, text in RULE_TEXT:
        lines.append(f"### {name}")
        lines.append("")
        lines.append(text)
        lines.append("")
        bits = []
        for row in rank_rows:
            if row["key"] != key:
                continue
            if row["n"] == 0:
                bits.append(f"{row['product']}在发布日上没有喊过。")
            else:
                bits.append(
                    f"{row['product']} {row['hits']}/{row['n']}，命中 {pct(row['rate'])}，95% 区间 {pct(row['lo'])}–{pct(row['hi'])}，"
                    f"前半段 {pct(row['early'])}，后半段 {pct(row['late'])}，方向收益 {bp(row['edge'])}。"
                )
        lines.append("".join(bits))
        lines.append("")

    lines.extend([
        "## 交易策略",
        "",
        "只做沪银主力，一次一手。合约乘数是 15 千克/手，报价是元/千克。美国 8:30 ET 的发布日如果当天上期所有日盘，就在 15:00 的排名和仓单日报出来之后算信号。21:00 夜盘开盘成交，这个交易日的 15:00 收盘平仓。从夜盘持有到当天下午收盘，在上期所的交易日里属于平今。主力合约代码相对信号日发生变化就不做。",
        "",
        "仓单规则：白银仓单五日千克数增加则做空，减少则做多，变化为 0 不做。高盛规则：沪银高盛全月公布净持仓的五日变化为负则做空，为正则做多，差分为 0 不做。",
        "",
        "两条可以单独做，也可以合在一起。合在一起时有两种做法。同向才做：两条都喊了并且方向一样，做 1 手；缺一条，或者方向相反，就不做。有一边就做：只有一条喊时做那一条，两条方向相反时不做，避免把冲突信号叠成两手。",
        "",
        "夏令时美国 8:30 对应北京时间 20:30，21:00 的开盘已经是数据出来之后约 30 分钟。冬令时对应北京时间 21:30，21:00 的开盘还在数据出来之前。两段都按同一张开盘到收盘的表记账，但夏令时这笔赚的是公布之后的漂移，冬令时这笔把公布本身包进持仓。",
        "",
        "成本按两档算。低档用非 6 月、12 月合约的开平各万分之 0.1，再加买卖各 1 个最小变动价位，白银最小变动是 1 元/千克，一手 15 千克，滑点 30 元。高档用 2026 年 1 月 7 日上期所通知里写过的那一档：开仓万分之 0.5，平今万分之 2.5，再加同样的 30 元滑点。期货公司在交易所费用上的加收没有算进去。这不是投资建议，下面的盈亏是这段历史里按这个规则会留下的手数。",
        "",
        "## 回测",
        "",
        "| 做法 | 手数 | 命中 | 95%下限 | 前半段 | 后半段 | 方向收益 | 毛利元 | 低档成本后 | 高档成本后 | 低档最大回撤 | 低档最差一手 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ])

    def add_book_row(title: str, rows: pd.DataFrame) -> None:
        stats = summarize(rows, "ret_open") if len(rows) else {"n": 0, "hits": 0, "rate": None, "lo": None, "early": None, "late": None, "edge": None}
        low = curve_stats(rows["net_low"]) if len(rows) else curve_stats(pd.Series(dtype=float))
        high = curve_stats(rows["net_high"]) if len(rows) else curve_stats(pd.Series(dtype=float))
        gross = curve_stats(rows["gross_yuan"]) if len(rows) else curve_stats(pd.Series(dtype=float))
        lines.append(
            f"| {title} | {stats['n']} | {pct(stats['rate'])} | {pct(stats['lo'])} | {pct(stats['early'])} | {pct(stats['late'])} | {bp(stats['edge'])} | "
            f"{money(gross['total'])} | {money(low['total'])} | {money(high['total'])} | {money(low['dd'])} | {money(low['worst'])} |"
        )

    for title, rows in books.items():
        add_book_row(title, rows)
    lines.append("")
    lines.append("全部交易日用同一条规则、同一个开盘到收盘，用来看这条优势是不是只集中在发布日。")
    lines.append("")
    lines.append("| 做法 | 手数 | 命中 | 方向收益 | 毛利元 | 低档成本后 | 高档成本后 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for title, rows in everyday.items():
        stats = summarize(rows, "ret_open")
        low = curve_stats(rows["net_low"])
        high = curve_stats(rows["net_high"])
        gross = curve_stats(rows["gross_yuan"])
        lines.append(
            f"| {title} | {stats['n']} | {pct(stats['rate'])} | {bp(stats['edge'])} | {money(gross['total'])} | {money(low['total'])} | {money(high['total'])} |"
        )

    lines.extend(["", "### 发布日按夏令时和冬令时拆开", ""])
    lines.append("| 做法 | 时令 | 手数 | 命中 | 方向收益 | 低档成本后 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for title, rows in books.items():
        for season in ("夏令时", "冬令时"):
            part = rows[rows["season"] == season]
            stats = summarize(part, "ret_open") if len(part) else {"n": 0, "rate": None, "edge": None}
            low = curve_stats(part["net_low"]) if len(part) else curve_stats(pd.Series(dtype=float))
            lines.append(f"| {title} | {season} | {stats['n']} | {pct(stats['rate'])} | {bp(stats['edge'])} | {money(low['total'])} |")

    lines.extend(["", "### 仓单规则按年份", ""])
    warrant_rows = books["仓单单独"]
    lines.append("| 年份 | 手数 | 命中 | 方向收益 | 低档成本后 | 高档成本后 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for year in (2025, 2026):
        part = warrant_rows[warrant_rows["date"].dt.year == year]
        stats = summarize(part, "ret_open") if len(part) else {"n": 0, "rate": None, "edge": None}
        low = curve_stats(part["net_low"]) if len(part) else curve_stats(pd.Series(dtype=float))
        high = curve_stats(part["net_high"]) if len(part) else curve_stats(pd.Series(dtype=float))
        lines.append(f"| {year} | {stats['n']} | {pct(stats['rate'])} | {bp(stats['edge'])} | {money(low['total'])} | {money(high['total'])} |")

    chosen = books["有一边就做，冲突不做"]
    lines.extend(["", "### 采用的做法", ""])
    warrant_hits = int((books["仓单单独"]["pred"] * books["仓单单独"]["ret_open"] > 0).sum())
    removed_n = len(books["仓单单独"]) - len(chosen)
    removed_hits = warrant_hits - int((chosen["pred"] * chosen["ret_open"] > 0).sum())
    top3 = float(chosen["net_low"].sort_values(ascending=False).head(3).sum())
    low = curve_stats(chosen["net_low"])
    high = curve_stats(chosen["net_high"])
    stats = summarize(chosen, "ret_open")
    lines.append(
        "往前做，用「有一边就做，冲突不做」，每次一手沪银主力，夜盘开盘进，当日 15:00 出。"
        "在这段样本里，高盛五日有信号的 20 个下午，仓单五日也都有信号。高盛没有单独开过仓，它只在和仓单方向相反时把那天空出来。"
        f"这样的冲突有 {removed_n} 个下午，其中仓单原本文对了 {removed_hits} 次、错了 {removed_n - removed_hits} 次，跳过之后从仓单单独的 49/79 变成 {stats['hits']}/{stats['n']}。"
    )
    lines.append("")
    lines.append(
        f"该做法命中 {pct(stats['rate'])}，方向收益 {bp(stats['edge'])}，"
        f"低档成本后合计 {money(low['total'])} 元，高档成本后合计 {money(high['total'])} 元，"
        f"低档最大回撤 {money(low['dd'])} 元。t 统计量低档为 {num(low['t'])}，高档为 {num(high['t'])}。"
        f"低档盈亏里最大的三笔合计 {money(top3)} 元。2025 年还没有可用的高盛五日信号，那一年就是仓单单独，低档成本后接近打平，高档成本后是亏的。"
        "规则是在全部发布日上挑出来的，这段盈亏不是挑规则之前留出来的考试样本。仓单放到全部交易日，方向收益只剩 +3.3bp，高档成本后接近 0，所以不要在没有发布的日子用同一条规则。"
        "10 月 14 日和 10 月 15 日两条方向相反，按这个做法空仓。14 日夜盘开盘到收盘沪银下跌，高盛看跌会对；15 日上涨，仓单看涨会对。"
    )
    lines.append("")
    lines.append("下一次核心发布是 2026-10-29 的 GDP 和 PCE，北京时间 20:30。那天 15:00 的 `pm` 文件和仓单日报出来之后，按上面的顺序决定做不做、做哪一边。10 月 16 日的排名还没有，现在没有可以下的单。")
    lines.append("")
    lines.append("| 发布日 | 发布 | 时令 | 仓单五日 | 高盛五日 | 方向 | 开盘到收盘 | 毛利元 | 低档后 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    log = chosen.sort_values("date")
    for _, row in log.iterrows():
        w5 = "" if pd.isna(row.get("w_chg5")) else f"{row['w_chg5']:+,.0f}"
        g5 = "" if pd.isna(row.get("gs_all_chg5")) else f"{row['gs_all_chg5']:+,.0f}"
        lines.append(
            f"| {row['day'].isoformat()} | {row['label']} | {row['season']} | {w5} | {g5} | "
            f"{'多' if row['pred'] > 0 else '空'} | {bp(row['ret_open'])} | {money(row['gross_yuan'])} | {money(row['net_low'])} |"
        )
    lines.append("")
    lines.append(f"数字由 `research/seat_position/trade_strategy.py` 算出。生成于 {datetime.now():%Y-%m-%d}。")
    lines.append("")
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("ranked", len(ranked), "hold", [(row["product"], row["key"], row["hits"], row["n"]) for row in hold])
    for title, rows in books.items():
        print(title, summarize(rows, "ret_open")["hits"], summarize(rows, "ret_open")["n"], money(curve_stats(rows["net_low"])["total"]))
    print("wrote", OUT_PATH)


if __name__ == "__main__":
    main()
