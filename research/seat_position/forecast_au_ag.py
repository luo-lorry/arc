#!/usr/bin/env python3
"""Forward gold and silver direction from today's seats and warrants.

Information cutoff is the 2026-10-15 day-session close. The night session
that starts at 21:00 belongs to the next trading day and is not an input.
The four score lines below were fixed before the percentile results were
read. Returns are not used to set a threshold or a weight.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
HTML_DIR = ROOT / "data" / "warrant_html"
DAT_DIR = ROOT / "data" / "warrant_dat"
COMPACT = ROOT / "data" / "shfe_compact"
PANEL = ROOT / "product" / "zhongcai" / "precious_panel.json"
OUT_MD = ROOT / "product" / "FORECAST_AU_AG.md"
OUT_JSON = ROOT / "data" / "warrants_au_ag.json"

CUTOFF = date(2026, 10, 15)
HORIZON = [
    date(2026, 10, 16),
    date(2026, 10, 19),
    date(2026, 10, 20),
    date(2026, 10, 21),
    date(2026, 10, 22),
]
PRODUCTS = {"au": "沪金", "ag": "沪银"}
# SHFE gold is 1,000 grams a lot. SHFE silver is 15 kilograms a lot.
KG_PER_LOT = {"au": 1.0, "ag": 15.0}


def parse_dat(path: Path) -> dict:
    payload = json.loads(path.read_text())
    out = {}
    for product in ("au", "ag"):
        rows = [row for row in payload["o_cursor"] if row["VARID"] == product]
        totals = [row for row in rows if str(row["ROWSTATUS"]) == "2"]
        if totals:
            row = totals[-1]
            out[product] = (float(row["WRTWGHTS"]), float(row["WRTCHANGE"]))
        else:
            warehouses = [row for row in rows if str(row["ROWSTATUS"]) == "0"]
            out[product] = (
                sum(float(row["WRTWGHTS"]) for row in warehouses),
                sum(float(row["WRTCHANGE"]) for row in warehouses),
            )
    return out


def _cells(row_html: str) -> list[str]:
    cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row_html, flags=re.S)
    values = [re.sub(r"<[^>]+>", "", cell).strip() for cell in cells]
    return [value for value in values if value]


def parse_html(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    current = None
    buckets = {"au": [], "ag": []}
    for row_html in re.findall(r"<tr[^>]*>(.*?)</tr>", text, flags=re.S):
        values = _cells(row_html)
        if not values:
            continue
        if any(value.startswith("单位：") for value in values):
            head = values[0]
            if head.startswith("黄金"):
                current = "au"
            elif head.startswith("白银"):
                current = "ag"
            else:
                current = None
            continue
        if current is None or values[0] in ("地区", "仓库"):
            continue
        numbers = []
        for value in values:
            token = value.replace(",", "").replace("+", "")
            if re.fullmatch(r"-?\d+(?:\.\d+)?", token):
                numbers.append(float(token))
        if len(numbers) < 2:
            continue
        buckets[current].append((values[0], numbers[-2], numbers[-1]))
    out = {}
    for product, items in buckets.items():
        totals = [item for item in items if item[0] == "总计"]
        if totals:
            out[product] = (
                sum(item[1] for item in totals),
                sum(item[2] for item in totals),
            )
        else:
            warehouses = [
                item for item in items if item[0] not in ("合计", "总计", "小计")
            ]
            out[product] = (
                sum(item[1] for item in warehouses),
                sum(item[2] for item in warehouses),
            )
    return out


def load_warrants(trading_days: list[str]) -> pd.DataFrame:
    rows = []
    for day in trading_days:
        ymd = day.replace("-", "")
        dat_path = DAT_DIR / f"{ymd}.dat"
        html_path = HTML_DIR / f"{ymd}.html"
        if dat_path.exists():
            parsed = parse_dat(dat_path)
            source = "dat"
        elif html_path.exists():
            parsed = parse_html(html_path)
            source = "html"
        else:
            raise FileNotFoundError(day)
        rows.append(
            {
                "date": day,
                "source": source,
                "au_kg": parsed["au"][0],
                "au_chg": parsed["au"][1],
                "ag_kg": parsed["ag"][0],
                "ag_chg": parsed["ag"][1],
            }
        )
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"])
    for product in ("au", "ag"):
        diff = frame[f"{product}_kg"].diff()
        mismatch = (diff - frame[f"{product}_chg"]).abs() > 1
        mismatch.iloc[0] = False
        if mismatch.any():
            bad = frame.loc[mismatch, "date"].dt.strftime("%Y-%m-%d").tolist()
            raise RuntimeError(f"{product} warrant change does not match the stock difference: {bad[:5]}")
        frame[f"{product}_chg5"] = frame[f"{product}_kg"].diff(5)
    return frame


def contract_month(code: str) -> tuple[int, int]:
    return 2000 + int(code[-4:-2]), int(code[-2:])


def book_roles(quotes: dict, product: str, day: date) -> tuple[dict[str, str], str]:
    pool = [(code, quote.get("oi") or 0) for code, quote in quotes.items() if code.startswith(product)]
    main = max(pool, key=lambda item: item[1])[0]
    roles = {}
    for code in quotes:
        if not code.startswith(product):
            continue
        year, month = contract_month(code)
        if (year, month) < (day.year, day.month):
            roles[code] = "已过交割月"
        elif (year, month) == (day.year, day.month):
            roles[code] = "当月"
        elif contract_month(code) <= contract_month(main):
            roles[code] = "近月"
        else:
            roles[code] = "远月"
    if roles.get(main) == "当月":
        later = [code for code, role in roles.items() if role == "远月" and quotes[code].get("oi")]
        if later:
            roles[min(later, key=contract_month)] = "近月"
    return roles, main


def side_sign(side: str) -> int:
    return 1 if side == "long" else -1


def fmt_lots(value: float) -> str:
    return f"{value:,.0f}"


def fmt_kg(value: float) -> str:
    return f"{value:,.0f}"


def fmt_pct(rank: float) -> str:
    return f"{rank * 100:.1f}%"


def point_from_tail(value: float, low: float, high: float, low_is_bearish: bool) -> int:
    if value <= low:
        return -1 if low_is_bearish else 1
    if value >= high:
        return 1 if low_is_bearish else -1
    return 0


def price_oi_point(price_change: float, oi_change: float) -> int:
    if oi_change <= 0:
        return 0
    if price_change < 0:
        return -1
    if price_change > 0:
        return 1
    return 0


def verdict(score: int) -> str:
    if score <= -2:
        return "偏空"
    if score == -1:
        return "温和偏空"
    if score == 0:
        return "无明确方向"
    if score == 1:
        return "温和偏多"
    return "偏多"


def rank_of(prior: pd.Series, value: float) -> tuple[float, float, float, int]:
    sample = prior.dropna()
    return (
        float((sample <= value).mean()),
        float(sample.quantile(0.10)),
        float(sample.quantile(0.90)),
        int(sample.shape[0]),
    )


def load_seats(day: date) -> dict:
    path = COMPACT / f"{day.strftime('%Y%m%d')}.json"
    return json.loads(path.read_text())


def member_rows(payload: dict, day: date, product: str, member: str) -> list[dict]:
    roles, _main = book_roles(payload["quotes"], product, day)
    rows = []
    for seat in payload["seats"]:
        if seat["member"] != member or not seat["code"].startswith(product):
            continue
        if seat["side"] not in ("long", "short"):
            continue
        rows.append(
            {
                "code": seat["code"].upper(),
                "role": roles.get(seat["code"], ""),
                "side": "多" if seat["side"] == "long" else "空",
                "rank": int(seat["rank"]),
                "qty": seat["qty"],
                "chg": seat["chg"],
                "signed": side_sign(seat["side"]) * seat["qty"],
            }
        )
    rows.sort(key=lambda item: (item["code"], item["side"]))
    return rows


def write_report(result: dict) -> None:
    horizon = "、".join(day.isoformat() for day in HORIZON)
    lines = [
        "# 沪金、沪银未来五个交易日的方向",
        "",
        f"信息截止：{CUTOFF.isoformat()} 日盘收盘。用的是这一天的上期所会员持仓排名，和同一天的仓单日报。当天 21:00 才开始的夜盘属于 {HORIZON[0].isoformat()}，没有拿来当输入。",
        "",
        f"预测窗口：{horizon}。这是截止日之后的五个交易日，中间隔的是周末。今晚夜盘算第一天。窗口里没有新的美国 CPI，下一次是 2026-11-10。10 月 14 日已经公布的 CPI，第一段价格反应在 10 月 15 日日盘里，不再重复计算。",
        "",
        "这是方向，不是目标价。四项分数在看分位数之前写死，没有用涨跌幅去调权重。不构成投资建议。",
        "",
        "## 结论",
        "",
        f"沪银主力，未来五个交易日 **{result['ag']['verdict']}**。四项合计 {result['ag']['score']:+d}。",
        "",
        f"沪金主力，同一窗口 **{result['au']['verdict']}**。四项合计 {result['au']['score']:+d}。",
        "",
        "空的一边来自高盛期货已经公布的净持仓：白银是 2025 年以来的新低，黄金也在最空的一档。白银中财的当月加近月合计在减少，减的是 AG2611，主力 AG2612 上的多单还在。白银仓单这五天在下降，降幅停在过去 13% 的位置，没有进到事先定下的 10% 极端档，库存本身仍高，所以这一项是 0。黄金仓单在样本高位，五日变化为 0。",
        "",
        "## 规则",
        "",
        "每一项只取 −1、0、+1。过去样本都停在 2026-10-14，当天的数只用来打分，不放进自己的阈值。分位是样本的线性分位。位置用「过去小于等于今天这个数的比例」。",
        "",
        "1. 高盛期货公布净持仓，全部合约，多单减空单。低于过去 10% 分位记 −1，高于 90% 分位记 +1。没进前 20 的一侧不在这个数里。",
        "2. 中财期货当月加近月的公布净持仓，五个交易日的变化。低于过去 10% 分位记 −1，高于 90% 分位记 +1。远月多单单列，不进这一项。",
        "3. 仓单，五个交易日的千克变化。减少落到过去 10% 分位记 +1，增加落到 90% 分位记 −1。仓单下降读成可交割库存变紧，仓单上升读成供给变多。",
        "4. 主力合约过去五个交易日的价格和持仓。价格下跌且持仓增加记 −1，价格上涨且持仓增加记 +1，持仓没有增加记 0。",
        "",
        "合计 ≤ −2 为偏空，−1 为温和偏空，0 为无明确方向，+1 为温和偏多，≥ +2 为偏多。",
        "",
        "当月是交割月份等于交易日所在月份的合约。近月是从当月数到持仓最大的合约，含两端。再往后是远月。",
        "",
        "## 打分",
        "",
        "| 品种 | 高盛净持仓 | 中财近月五日变化 | 仓单五日变化 | 价与仓 | 合计 | 方向 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for product in ("ag", "au"):
        row = result[product]
        lines.append(
            f"| {PRODUCTS[product]} | {row['gs_point']:+d} | {row['zc_point']:+d} | "
            f"{row['warrant_point']:+d} | {row['oi_point']:+d} | {row['score']:+d} | {row['verdict']} |"
        )
    lines.extend(["", "## 今天手里有的数", ""])
    lines.extend(result["detail_lines"])
    lines.extend(
        [
            "",
            "## 这套分数没有做的事",
            "",
            "没有用 10 月 15 日之后的价格。面板里这一天的下一日收益是空的。没有把历史上「信号出现之后的涨跌」再拟合回来改今天的符号。仓单「增减」和相邻交易日库存差逐日对过，430 个交易日都对齐。",
            "",
            "公布持仓不是全账户。高盛和中财在多数交易日进不了前 20，那些日子的公布净持仓按 0 计，所以 10% 分位常常就是 0。低于这条线，意思是净空已经上榜，并且处在 2025 年以来公布数的左侧，不是整个席位的真实库存。",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def detail_for(product: str, seats_now: dict, seats_then: dict, warrant: pd.Series, prior_w: pd.DataFrame, panel_today: pd.Series, panel_prior: pd.DataFrame, panel_then: pd.Series) -> list[str]:
    name = PRODUCTS[product]
    gs_rank, gs_p10, gs_p90, gs_n = rank_of(panel_prior["gs_all"], panel_today["gs_all"])
    zc_rank, zc_p10, zc_p90, zc_n = rank_of(panel_prior["zc_near_chg5"], panel_today["zc_near_chg5"])
    w_rank, w_p10, w_p90, w_n = rank_of(prior_w[f"{product}_chg5"], warrant[f"{product}_chg5"])
    level_rank, _level_p10, level_p90, level_n = rank_of(prior_w[f"{product}_kg"], warrant[f"{product}_kg"])
    day_rank, day_p10, _day_p90, day_n = rank_of(prior_w[f"{product}_chg"], warrant[f"{product}_chg"])
    kg_lot = KG_PER_LOT[product]
    lines = [
        f"### {name}",
        "",
        (
            f"高盛公布净持仓 {fmt_lots(panel_today['gs_all'])} 手，其中当月加近月 {fmt_lots(panel_today['gs_near'])} 手。"
            f"在 {gs_n} 个更早的交易日里，小于等于这个数的比例是 {fmt_pct(gs_rank)}。"
            f"10% 分位是 {fmt_lots(gs_p10)} 手，90% 分位是 {fmt_lots(gs_p90)} 手。"
        ),
        "",
        (
            f"中财当月加近月五日变化 {fmt_lots(panel_today['zc_near_chg5'])} 手。"
            f"过去 {zc_n} 次五日变化里，小于等于这次的比例是 {fmt_pct(zc_rank)}。"
            f"10% 分位是 {fmt_lots(zc_p10)} 手，90% 分位是 {fmt_lots(zc_p90)} 手。"
            f"全部合约的公布净持仓是 {fmt_lots(panel_today['zc_all'])} 手，远月包含在这个数里，不另计分。"
        ),
        "",
        (
            f"仓单 {fmt_kg(warrant[f'{product}_kg'])} 千克，折合 {fmt_lots(warrant[f'{product}_kg'] / kg_lot)} 手。"
            f"过去 {level_n} 天里，库存不低于今天的日子只占 {fmt_pct(1 - level_rank)}，今天处在 {fmt_pct(level_rank)} 的位置，90% 分位是 {fmt_kg(level_p90)} 千克。"
            f"今日增减 {fmt_kg(warrant[f'{product}_chg'])} 千克，这个单日变化在过去的位置是 {fmt_pct(day_rank)}，10% 分位是 {fmt_kg(day_p10)} 千克。"
            f"计分用的是五日变化 {fmt_kg(warrant[f'{product}_chg5'])} 千克，位置 {fmt_pct(w_rank)}，10% 分位 {fmt_kg(w_p10)} 千克，90% 分位 {fmt_kg(w_p90)} 千克，样本 {w_n}。"
        ),
        "",
        (
            f"主力 {panel_today['main']} 收盘 {panel_today['close']:,.2f}，持仓 {fmt_lots(panel_today['oi'])} 手。"
            f"五个交易日前（{panel_then['date'].date().isoformat()}，{panel_then['main']}）收盘 {panel_then['close']:,.2f}，持仓 {fmt_lots(panel_then['oi'])} 手。"
            f"价格变化 {panel_today['close'] - panel_then['close']:+,.2f}，持仓变化 {panel_today['oi'] - panel_then['oi']:+,.0f} 手。"
        ),
        "",
        "10 月 15 日上榜明细：",
        "",
        "| 席位 | 合约 | 角色 | 方向 | 排名 | 手数 | 当日变化 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    day = CUTOFF
    for member in ("高盛期货", "中财期货"):
        rows = member_rows(seats_now, day, product, member)
        if not rows:
            lines.append(f"| {member} |  |  |  |  | 不在榜 |  |")
            continue
        for row in rows:
            if row["side"] == "多" and row["role"] == "远月" and member == "中财期货":
                pass
            lines.append(
                f"| {member} | {row['code']} | {row['role']} | {row['side']} | {row['rank']} | "
                f"{fmt_lots(row['qty'])} | {row['chg']:+,.0f} |"
            )
    lines.append("")
    lines.append(f"五个交易日前（{panel_then['date'].date().isoformat()}）同一席位的上榜合约：")
    lines.append("")
    lines.append("| 席位 | 合约 | 角色 | 方向 | 手数 |")
    lines.append("| --- | --- | --- | --- | --- |")
    then_day = panel_then["date"].date()
    for member in ("高盛期货", "中财期货"):
        rows = [row for row in member_rows(seats_then, then_day, product, member) if row["side"] in ("多", "空")]
        # volume rows are not in member_rows; only long/short were stored. Good.
        position_rows = [row for row in rows if row["side"] in ("多", "空")]
        if not position_rows:
            lines.append(f"| {member} |  |  |  | 不在榜 |")
        for row in position_rows:
            lines.append(
                f"| {member} | {row['code']} | {row['role']} | {row['side']} | {fmt_lots(row['qty'])} |"
            )
    lines.append("")
    return lines


def main() -> None:
    panel = pd.DataFrame(json.loads(PANEL.read_text()))
    panel["date"] = pd.to_datetime(panel["date"])
    if panel["date"].max().date() != CUTOFF:
        raise RuntimeError(f"panel ends at {panel['date'].max().date()}, not the cutoff")
    if panel.loc[panel["date"].dt.date == CUTOFF, "ret_next"].notna().any():
        raise RuntimeError("cutoff day already has a next-session return")
    trading_days = sorted(panel["date"].dt.strftime("%Y-%m-%d").unique())
    warrants = load_warrants(trading_days)
    if warrants["date"].max().date() != CUTOFF:
        raise RuntimeError("warrants do not reach the cutoff")
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    serial = warrants.copy()
    serial["date"] = serial["date"].dt.strftime("%Y-%m-%d")
    OUT_JSON.write_text(json.dumps(serial.to_dict(orient="records"), ensure_ascii=False), encoding="utf-8")

    seats_now = load_seats(CUTOFF)
    result = {}
    detail_lines: list[str] = []
    for product in ("ag", "au"):
        sub = panel[panel["product"] == product].sort_values("date").reset_index(drop=True)
        today = sub[sub["date"].dt.date == CUTOFF].iloc[0]
        prior = sub[sub["date"].dt.date < CUTOFF]
        then = sub.iloc[sub.index[sub["date"].dt.date == CUTOFF][0] - 5]
        w_today = warrants[warrants["date"].dt.date == CUTOFF].iloc[0]
        w_prior = warrants[warrants["date"].dt.date < CUTOFF]
        gs_rank, gs_p10, gs_p90, _gs_n = rank_of(prior["gs_all"], today["gs_all"])
        zc_rank, zc_p10, zc_p90, _zc_n = rank_of(prior["zc_near_chg5"], today["zc_near_chg5"])
        w_rank, w_p10, w_p90, _w_n = rank_of(w_prior[f"{product}_chg5"], w_today[f"{product}_chg5"])
        gs_point = point_from_tail(today["gs_all"], gs_p10, gs_p90, True)
        zc_point = point_from_tail(today["zc_near_chg5"], zc_p10, zc_p90, True)
        warrant_point = point_from_tail(w_today[f"{product}_chg5"], w_p10, w_p90, False)
        oi_point = price_oi_point(today["close"] - then["close"], today["oi"] - then["oi"])
        score = gs_point + zc_point + warrant_point + oi_point
        result[product] = {
            "score": score,
            "verdict": verdict(score),
            "gs_point": gs_point,
            "zc_point": zc_point,
            "warrant_point": warrant_point,
            "oi_point": oi_point,
            "gs_rank": gs_rank,
            "zc_rank": zc_rank,
            "w_rank": w_rank,
        }
        seats_then = load_seats(then["date"].date())
        detail_lines.extend(
            detail_for(product, seats_now, seats_then, w_today, w_prior, today, prior, then)
        )
    # Composition notes that keep the score readable. These sentences describe
    # the same rows already scored; they do not add a fifth point.
    detail_lines.extend(composition_notes(seats_now))
    result["detail_lines"] = detail_lines
    write_report(result)
    print(json.dumps({k: result[k] for k in ("ag", "au")}, ensure_ascii=False, indent=2))


def composition_notes(seats_now: dict) -> list[str]:
    return [
        "### 分数里需要单独看清的地方",
        "",
        "白银的中财减仓，减的是 AG2611，不是主力 AG2612。10 月 8 日 AG2611 多单 2,786 手，10 月 15 日 777 手。AG2612 多单同期从 6,621 手到 6,977 手，五日是增加的，10 月 15 日当天还加了 228 手。当月 AG2610 只剩空单 2 手。近月合计因此是减少，但主力合约上的中财多单还在。远月 AG2702、AG2704 仍是多单榜的前两名，不进入近五日的计分。",
        "",
        "白银仓单 10 月 15 日一天减少 45,656 千克：中工美供应链 −24,195，深圳威豹 −11,286，外运华东虹桥 −10,175。这一天的减幅比过去 97% 的交易日都大。计分用的五日变化是 −75,097 千克，还没有低过过去的 10% 分位，所以仓单这一项是 0，不是 +1。库存本身仍在样本的高位。这是席位偏空旁边唯一方向相反的信息，它没有强到改符号。",
        "",
        "黄金仓单是「上期所指定交割金库」116,028 千克，当日变化 0，五日变化也是 0。库存靠近这段样本的最高点，但本周没有新的注册。中财在 AU2612 和当月合约都不在榜，多单在 AU2702、AU2704、AU2708，五日几乎没动。黄金的 −1 只来自高盛在 AU2612 的空单。",
        "",
        "白银主力最近两个交易日是价跌仓增：10 月 13 日持仓 284,275 手、收盘 14,743，10 月 15 日持仓 293,325 手、收盘 13,660。五日合计持仓仍是下降的，按事先定下的五日规则，价仓这一项记 0，不因为最后两天再加一个空分。",
        "",
    ]


if __name__ == "__main__":
    main()
