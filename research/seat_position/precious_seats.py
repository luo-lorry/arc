#!/usr/bin/env python3
"""Zhongcai and Goldman seats on SHFE gold and silver, nearby months only.

The chart labeled 净持仓量 matches the sum of lots the exchange actually
prints: visible long minus visible short, across every gold or silver
contract. A side outside the top 20 is omitted from that sum. It is not
filled in as zero, and it is not the member's whole book.

Nearby book, fixed before the results are read:
  当月 = the contract whose delivery month is the trading day's month.
  近月 = every contract from 当月 through the main contract (largest open
  interest), inclusive. Far months beyond the main contract stay in a
  separate column.

The rule being tested, also fixed first: after 高盛期货's disclosed net
position is more short than its own trailing 120 sessions (below the 10th
percentile), the next main-contract session falls more often than usual.
The position is the day-session ranking. The next session includes the
night that starts at 21:00.
"""

from __future__ import annotations

import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
COMPACT = ROOT / "data" / "shfe_compact"
OUT = ROOT / "product" / "zhongcai"
UA = {"User-Agent": "Mozilla/5.0"}
FONT = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
MEMBERS = ("中财期货", "高盛期货")
PRODUCTS = {"au": "沪金", "ag": "沪银"}

# BLS schedule, 08:30 US Eastern. In October that is 20:30 Beijing, half an
# hour before the Shanghai night session.
CPI_DATES = [
    date(2026, 1, 13),
    date(2026, 2, 13),
    date(2026, 3, 11),
    date(2026, 4, 10),
    date(2026, 5, 12),
    date(2026, 6, 10),
    date(2026, 7, 14),
    date(2026, 8, 12),
    date(2026, 9, 11),
    date(2026, 10, 14),
    date(2026, 11, 10),
    date(2026, 12, 10),
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


def num(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def ymd_of(day: date) -> str:
    return day.strftime("%Y%m%d")


def pull_day(day: date) -> dict | None:
    ymd = ymd_of(day)
    path = COMPACT / f"{ymd}.json"
    if path.exists():
        return json.loads(path.read_text())
    pm = http_json(
        "https://www.shfe.com.cn/data/tradedata/future/dailydata/"
        f"pm{ymd}.dat"
    )
    kx = http_json(
        "https://www.shfe.com.cn/data/tradedata/future/dailydata/"
        f"kx{ymd}.dat"
    )
    if not pm or not pm.get("o_cursor") or not kx or not kx.get("o_curinstrument"):
        return None
    quotes = {}
    for item in kx["o_curinstrument"]:
        group = str(item.get("PRODUCTGROUPID") or "").lower()
        month = str(item.get("DELIVERYMONTH") or "")
        if group not in PRODUCTS or not month.isdigit() or len(month) != 4:
            continue
        quotes[group + month] = {
            "open": num(item.get("OPENPRICE")),
            "high": num(item.get("HIGHESTPRICE")),
            "low": num(item.get("LOWESTPRICE")),
            "close": num(item.get("CLOSEPRICE")),
            "settle": num(item.get("SETTLEMENTPRICE")),
            "pre": num(item.get("PRESETTLEMENTPRICE")),
            "oi": num(item.get("OPENINTEREST")),
            "volume": num(item.get("VOLUME")),
        }
    seats = []
    cuts: dict[str, dict] = {}
    for row in pm["o_cursor"]:
        code = str(row.get("INSTRUMENTID") or "").lower()
        if code.endswith("all") or code[:2] not in PRODUCTS:
            continue
        rank = int(row.get("RANK") or 0)
        if rank < 1 or rank > 20:
            continue
        slot = cuts.setdefault(
            code,
            {"long_n": 0, "long_cut": None, "short_n": 0, "short_cut": None},
        )
        if str(row.get("PARTICIPANTABBR2") or "").strip() and rank >= slot["long_n"]:
            slot["long_n"] = rank
            slot["long_cut"] = num(row.get("CJ2"))
        if str(row.get("PARTICIPANTABBR3") or "").strip() and rank >= slot["short_n"]:
            slot["short_n"] = rank
            slot["short_cut"] = num(row.get("CJ3"))
        for side, name_key, qty_key, chg_key in (
            ("volume", "PARTICIPANTABBR1", "CJ1", "CJ1_CHG"),
            ("long", "PARTICIPANTABBR2", "CJ2", "CJ2_CHG"),
            ("short", "PARTICIPANTABBR3", "CJ3", "CJ3_CHG"),
        ):
            name = str(row.get(name_key) or "")
            if name not in MEMBERS:
                continue
            seats.append(
                {
                    "member": name,
                    "code": code,
                    "side": side,
                    "rank": rank,
                    "qty": num(row.get(qty_key)),
                    "chg": num(row.get(chg_key)),
                }
            )
    payload = {"date": day.isoformat(), "quotes": quotes, "seats": seats, "cuts": cuts}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload


def download(start: date, end: date) -> list[dict]:
    days = []
    cur = start
    while cur <= end:
        if cur.weekday() < 5:
            days.append(cur)
        cur += timedelta(days=1)
    found = []
    done = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(pull_day, day): day for day in days}
        for fut in as_completed(futures):
            done += 1
            try:
                payload = fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"fail {futures[fut]} {exc}")
                continue
            if payload:
                found.append(payload)
            if done % 40 == 0 or done == len(days):
                print(f"pulled {done}/{len(days)} trading-day slots, kept {len(found)}")
    found.sort(key=lambda item: item["date"])
    return found


def contract_month(code: str) -> tuple[int, int]:
    return 2000 + int(code[-4:-2]), int(code[-2:])


def main_code(quotes: dict, product: str) -> str | None:
    pool = [
        (code, quote["oi"])
        for code, quote in quotes.items()
        if code.startswith(product) and quote.get("oi")
    ]
    if not pool:
        return None
    return max(pool, key=lambda item: item[1])[0]


def book_codes(quotes: dict, product: str, day: date) -> dict[str, str]:
    """Map contract -> 当月 / 近月 / 远月 / 已过交割月."""
    main = main_code(quotes, product)
    roles = {}
    for code in quotes:
        if not code.startswith(product):
            continue
        year, month = contract_month(code)
        if (year, month) < (day.year, day.month):
            roles[code] = "已过交割月"
        elif (year, month) == (day.year, day.month):
            roles[code] = "当月"
        elif main and contract_month(code) <= contract_month(main):
            roles[code] = "近月"
        else:
            roles[code] = "远月"
    if main and roles.get(main) == "当月":
        # The liquid contract is the current month. The next listed month
        # with open interest is the nearby one.
        later = [
            code
            for code, role in roles.items()
            if role == "远月" and quotes[code].get("oi")
        ]
        if later:
            roles[min(later, key=contract_month)] = "近月"
    return roles


def seat_table(payload: dict) -> pd.DataFrame:
    day = date.fromisoformat(payload["date"])
    roles = {}
    for product in PRODUCTS:
        roles.update(book_codes(payload["quotes"], product, day))
    rows = []
    for seat in payload["seats"]:
        if seat["side"] == "volume":
            continue
        quote = payload["quotes"].get(seat["code"], {})
        cut = payload["cuts"].get(seat["code"], {})
        rows.append(
            {
                "date": payload["date"],
                "member": seat["member"],
                "code": seat["code"].upper(),
                "product": seat["code"][:2],
                "role": roles.get(seat["code"], ""),
                "side": seat["side"],
                "rank": seat["rank"],
                "qty": seat["qty"],
                "chg": seat["chg"],
                "oi": quote.get("oi"),
                "close": quote.get("close"),
                "settle": quote.get("settle"),
                "long_cut": cut.get("long_cut"),
                "short_cut": cut.get("short_cut"),
                "long_n": cut.get("long_n"),
                "short_n": cut.get("short_n"),
            }
        )
    columns = [
        "date", "member", "code", "product", "role", "side", "rank", "qty", "chg",
        "oi", "close", "settle", "long_cut", "short_cut", "long_n", "short_n",
    ]
    return pd.DataFrame(rows, columns=columns)


def disclosed_net(frame: pd.DataFrame, member: str, product: str, roles: set[str] | None) -> float:
    if frame.empty or "member" not in frame.columns:
        return 0.0
    sub = frame[(frame["member"] == member) & (frame["product"] == product)]
    if roles is not None:
        sub = sub[sub["role"].isin(roles)]
    if sub.empty:
        return 0.0
    long_qty = sub.loc[sub["side"] == "long", "qty"].sum()
    short_qty = sub.loc[sub["side"] == "short", "qty"].sum()
    return float(long_qty - short_qty)


def daily_panel(payloads: list[dict]) -> pd.DataFrame:
    rows = []
    frames = []
    for payload in payloads:
        frame = seat_table(payload)
        frames.append(frame)
        day = date.fromisoformat(payload["date"])
        for product, name in PRODUCTS.items():
            main = main_code(payload["quotes"], product)
            quote = payload["quotes"].get(main or "", {})
            row = {
                "date": payload["date"],
                "product": product,
                "name": name,
                "main": None if main is None else main.upper(),
                "open": quote.get("open"),
                "close": quote.get("close"),
                "settle": quote.get("settle"),
                "pre": quote.get("pre"),
                "oi": quote.get("oi"),
            }
            for member, prefix in (("高盛期货", "gs"), ("中财期货", "zc")):
                row[f"{prefix}_all"] = disclosed_net(frame, member, product, None)
                row[f"{prefix}_near"] = disclosed_net(frame, member, product, {"当月", "近月"})
                row[f"{prefix}_front"] = disclosed_net(frame, member, product, {"当月"})
            rows.append(row)
        if not frame.empty:
            pass
    panel = pd.DataFrame(rows).sort_values(["product", "date"])
    parts = []
    for _product, chunk in panel.groupby("product", sort=False):
        chunk = chunk.copy()
        chunk["ret_next"] = chunk["close"].shift(-1) / chunk["settle"] - 1.0
        chunk["gap_next"] = chunk["open"].shift(-1) / chunk["settle"] - 1.0
        chunk["ret_same"] = chunk["close"] / chunk["pre"] - 1.0
        for col in ("gs_all", "gs_near", "zc_all", "zc_near"):
            chunk[f"{col}_chg5"] = chunk[col] - chunk[col].shift(5)
        parts.append(chunk)
    panel = pd.concat(parts, ignore_index=True)
    seats = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return panel, seats


def rank_corr(left: pd.Series, right: pd.Series) -> float | None:
    pair = pd.DataFrame({"l": left, "r": right}).dropna()
    if len(pair) < 30:
        return None
    return float(pair["l"].rank().corr(pair["r"].rank()))


def extreme_test(panel: pd.DataFrame, product: str, column: str) -> dict:
    """Trailing 120-day 10th percentile. The threshold uses only earlier days."""
    sub = panel[panel["product"] == product].sort_values("date").reset_index(drop=True)
    flags = []
    for i in range(len(sub)):
        if i < 120:
            flags.append(False)
            continue
        window = sub.iloc[i - 120 : i][column]
        threshold = float(window.quantile(0.10))
        flags.append(bool(sub.iloc[i][column] < threshold))
    sub["extreme_short"] = flags
    extreme = sub[sub["extreme_short"] & sub["ret_next"].notna()]
    base = sub.iloc[120:]
    base = base[base["ret_next"].notna() & ~base["extreme_short"]]
    def pack(frame: pd.DataFrame) -> dict:
        rets = frame["ret_next"].dropna()
        gaps = frame["gap_next"].dropna()
        return {
            "n": int(len(rets)),
            "mean_ret": None if rets.empty else float(rets.mean()),
            "median_ret": None if rets.empty else float(rets.median()),
            "down_rate": None if rets.empty else float((rets < 0).mean()),
            "mean_gap": None if gaps.empty else float(gaps.mean()),
            "gap_down_rate": None if gaps.empty else float((gaps < 0).mean()),
        }
    return {
        "product": product,
        "column": column,
        "extreme": pack(extreme),
        "baseline": pack(base),
        "corr_level_next": rank_corr(sub[column], sub["ret_next"]),
        "corr_chg5_next": rank_corr(sub[f"{column}_chg5"], sub["ret_next"]),
        "corr_chg5_same": rank_corr(sub[f"{column}_chg5"], sub["ret_same"]),
    }


def history_stats(panel: pd.DataFrame, product: str, column: str, asof: str) -> dict:
    sub = panel[(panel["product"] == product) & (panel["date"] <= asof)].sort_values("date")
    prior = sub[sub["date"] < asof][column]
    current = float(sub.iloc[-1][column]) if not sub.empty else None
    shorts = prior[prior < 0]
    return {
        "current": current,
        "prior_min": None if prior.empty else float(prior.min()),
        "prior_p10": None if prior.empty else float(prior.quantile(0.10)),
        "prior_median_short": None if shorts.empty else float(shorts.median()),
        "days": int(len(prior)),
        "short_days": int((prior < 0).sum()),
        "deeper_than_min": current is not None and not prior.empty and current < float(prior.min()),
    }


def cpi_events(panel: pd.DataFrame, trading_days: list[date]) -> list[dict]:
    events = []
    for cpi in CPI_DATES:
        position_days = [day for day in trading_days if day <= cpi]
        reaction_days = [day for day in trading_days if day > cpi]
        if not position_days or not reaction_days:
            continue
        pos_day = position_days[-1]
        react_day = reaction_days[0]
        events.append({"cpi": cpi.isoformat(), "position_day": pos_day.isoformat(), "reaction_day": react_day.isoformat()})
    out = []
    for event in events:
        for product in PRODUCTS:
            row = panel[(panel["product"] == product) & (panel["date"] == event["position_day"])]
            nxt = panel[(panel["product"] == product) & (panel["date"] == event["reaction_day"])]
            if row.empty:
                continue
            rec = row.iloc[0]
            reaction = None if nxt.empty else nxt.iloc[0]
            # Gap of the reaction day is open / previous settle. Previous settle
            # is the reaction day's own `pre` field.
            gap = None
            full = None
            if reaction is not None and reaction["pre"] and reaction["open"] and reaction["close"]:
                gap = reaction["open"] / reaction["pre"] - 1.0
                full = reaction["close"] / reaction["pre"] - 1.0
            out.append(
                {
                    **event,
                    "product": product,
                    "gs_all": rec["gs_all"],
                    "gs_near": rec["gs_near"],
                    "zc_near": rec["zc_near"],
                    "main": rec["main"],
                    "gap": gap,
                    "full": full,
                }
            )
    return out


def lots(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    return f"{int(round(value)):,}"


def pct(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    return f"{value * 100:+.2f}%"


def write_report(panel: pd.DataFrame, seats: pd.DataFrame, tests: list[dict], events: list[dict], path: Path) -> None:
    latest = panel["date"].max()
    day = date.fromisoformat(latest)
    lines = [
        "# 中财期货、高盛期货在沪金沪银当月和近月",
        "",
        f"最新交易日 {latest}。持仓是该日日盘结束后的会员排名，不含当天 21:00 以后的夜盘。",
        "",
        "图上的「净持仓量」等于各合约已公布多单减去已公布空单。没进前 20 的那一侧不出现在这个数里。10 月 13 日白银高盛空单 4,743（AG2612）加 942（AG2702）等于图上的 −5,685；黄金空单 911（AU2612）等于图上的 −911。",
        "",
        "当月是交割月份等于交易日所在月的合约。近月是从当月数到持仓最大的合约，含两端。再往后的月份单独标成远月。",
        "",
        f"## {latest} 合约明细",
        "",
    ]
    latest_seats = seats[seats["date"] == latest].copy()
    for member in MEMBERS:
        lines.append(f"### {member}")
        lines.append("")
        lines.append("| 合约 | 品种 | 位置 | 方向 | 手数 | 排名 | 当日增减 | 该合约持仓 |")
        lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: |")
        chunk = latest_seats[latest_seats["member"] == member].sort_values(["product", "code", "side"])
        if chunk.empty:
            lines.append("| — | — | — | 当天金银榜上没有这个名字 | — | — | — | — |")
        for rec in chunk.itertuples(index=False):
            lines.append(
                f"| {rec.code} | {PRODUCTS[rec.product]} | {rec.role} | "
                f"{'多单' if rec.side == 'long' else '空单'} | {lots(rec.qty)} | {int(rec.rank)} | "
                f"{int(round(rec.chg)):+,} | {lots(rec.oi)} |"
            )
        lines.append("")
        for product, name in PRODUCTS.items():
            sub = panel[(panel["date"] == latest) & (panel["product"] == product)].iloc[0]
            prefix = "gs" if member == "高盛期货" else "zc"
            lines.append(
                f"- {name}公布净持仓：全部合约 {lots(sub[f'{prefix}_all'])} 手，"
                f"当月 {lots(sub[f'{prefix}_front'])} 手，当月加近月 {lots(sub[f'{prefix}_near'])} 手。"
                f"主力 {sub['main']}。"
            )
        lines.append("")
    lines.append("## 高盛空单和它自己的历史相比")
    lines.append("")
    lines.append("比较用的是全部合约公布净持仓，因为图上的 −5,685 和 −911 是这个口径。样本从有排名文件的第一天算到最新交易日的前一天。")
    lines.append("")
    lines.append("| 品种 | 最新公布净持仓 | 此前最空 | 此前空头中位数 | 此前 10% 分位 | 空头天数 | 是否新低 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for product, name in PRODUCTS.items():
        stats = history_stats(panel, product, "gs_all", latest)
        lines.append(
            f"| {name} | {lots(stats['current'])} | {lots(stats['prior_min'])} | "
            f"{lots(stats['prior_median_short'])} | {lots(stats['prior_p10'])} | "
            f"{stats['short_days']}/{stats['days']} | "
            f"{'是' if stats['deeper_than_min'] else '否'} |"
        )
    lines.append("")
    lines.append("空头中位数是此前所有净持仓为负的交易日的中位数，用来代表常出现的试探规模。10% 分位更空，接近它过去很少到达的区域。")
    lines.append("")
    lines.append("## 这条规律有没有用")
    lines.append("")
    lines.append("规则在看结果之前定好：高盛公布净持仓低于它自己过去 120 个交易日的 10% 分位，叫极端做空。看的是下一个交易日主力合约收盘相对上一交易日结算的涨跌，以及下一交易日开盘相对上一结算的跳空。过去 120 天不含当天。")
    lines.append("")
    for test in tests:
        name = PRODUCTS[test["product"]]
        ext = test["extreme"]
        base = test["baseline"]
        lines.append(f"### {name}，{test['column']}")
        lines.append("")
        lines.append(
            f"极端做空 {ext['n']} 次，下一日平均 {pct(ext['mean_ret'])}，"
            f"中位数 {pct(ext['median_ret'])}，下跌占比 {pct(ext['down_rate']).replace('+', '')}，"
            f"跳空平均 {pct(ext['mean_gap'])}。"
        )
        lines.append(
            f"同一时期其余交易日 {base['n']} 次，下一日平均 {pct(base['mean_ret'])}，"
            f"下跌占比 {pct(base['down_rate']).replace('+', '')}，跳空平均 {pct(base['mean_gap'])}。"
        )
        def corr(value):
            return "—" if value is None else f"{value:+.2f}"

        lines.append(
            f"净持仓与下一日涨跌的秩相关 {corr(test['corr_level_next'])}。"
            f"5 日净持仓变化与下一日涨跌的秩相关 {corr(test['corr_chg5_next'])}，"
            f"与当日涨跌的秩相关 {corr(test['corr_chg5_same'])}。"
        )
        lines.append("")
    lines.append("高盛在多数交易日并没有净空头寸上榜，所以整段样本的 10% 分位常常是 0。「极端做空」在这张表里大体就是公布净持仓为负，不是一个很深的历史分位。这些日子常常连在同一段行情里，不能当成互相独立的几十次试验。秩相关接近 0，净持仓的高低和下一日涨跌之间没有一条稳定的直线关系。")
    lines.append("")
    lines.append("## CPI 发布日前后")
    lines.append("")
    lines.append("美国 CPI 在美东 8:30 公布，对应北京时间 20:30 或 21:30，落在上期所夜盘 21:00 之前或刚开盘。持仓用公布日当天或之前最近一个日盘的排名。价格反应用下一个交易日：开盘相对前结算是夜盘跳空，收盘相对前结算含进白天。")
    lines.append("")
    lines.append("| CPI日 | 持仓日 | 反应日 | 品种 | 高盛全部净持仓 | 高盛当月+近月 | 中财当月+近月 | 跳空 | 全日 |")
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for event in events:
        if event["reaction_day"] > latest and event["gap"] is None:
            continue
        lines.append(
            f"| {event['cpi']} | {event['position_day']} | {event['reaction_day']} | "
            f"{PRODUCTS[event['product']]} | {lots(event['gs_all'])} | {lots(event['gs_near'])} | "
            f"{lots(event['zc_near'])} | {pct(event['gap'])} | {pct(event['full'])} |"
        )
    lines.append("")
    lines.append("持仓排名只记录这个席位已公布的多空。不构成投资建议。")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def font(size: int):
    return ImageFont.truetype(FONT, size)


def render_net(panel: pd.DataFrame, path: Path) -> None:
    width, height = 1400, 760
    image = Image.new("RGB", (width, height), (7, 18, 34))
    draw = ImageDraw.Draw(image)
    draw.text((36, 24), "高盛期货公布净持仓  多单减空单，未进前20的一侧不计入", font=font(32), fill=(236, 242, 248))
    colors = {"au": (232, 168, 74), "ag": (46, 196, 150)}
    for index, product in enumerate(("au", "ag")):
        sub = panel[panel["product"] == product].sort_values("date")
        values = sub["gs_all"].tolist()
        if not values:
            continue
        x0, y0 = 40, 90 + index * 320
        draw.rounded_rectangle((x0, y0, width - 40, y0 + 290), radius=12, fill=(12, 32, 56))
        draw.text((x0 + 20, y0 + 12), PRODUCTS[product], font=font(26), fill=(236, 242, 248))
        lo, hi = min(values), max(values)
        if lo == hi:
            hi = lo + 1
        left, right = x0 + 24, width - 70
        top, bottom = y0 + 56, y0 + 250
        zero_y = bottom - (bottom - top) * ((0 - lo) / (hi - lo))
        draw.line((left, zero_y, right, zero_y), fill=(80, 100, 120), width=1)
        points = []
        for i, value in enumerate(values):
            x = left + (right - left) * (i / max(len(values) - 1, 1))
            y = bottom - (bottom - top) * ((value - lo) / (hi - lo))
            points.append((x, y))
        if len(points) >= 2:
            draw.line(points, fill=colors[product], width=2)
        last = values[-1]
        draw.text((x0 + 140, y0 + 16), f"最新 {int(round(last)):+,} 手", font=font(22), fill=colors[product])
        draw.text((x0 + 20, y0 + 258), str(sub.iloc[0]["date"]), font=font(16), fill=(154, 176, 196))
        end = str(sub.iloc[-1]["date"])
        face = font(16)
        draw.text((width - 70 - draw.textlength(end, font=face), y0 + 258), end, font=face, fill=(154, 176, 196))
    image.save(path, "PNG")


def main():
    today = datetime.now(ZoneInfo("Asia/Shanghai")).date()
    end = today
    # Walk back if today's file is not published yet.
    while end.weekday() >= 5:
        end -= timedelta(days=1)
    payloads = download(date(2025, 1, 2), end)
    # Drop a trailing day that has seats but the next return is what we need;
    # keep every day that actually downloaded.
    if not payloads:
        raise RuntimeError("no SHFE files")
    panel, seats = daily_panel(payloads)
    tests = []
    for product in PRODUCTS:
        tests.append(extreme_test(panel, product, "gs_all"))
        tests.append(extreme_test(panel, product, "gs_near"))
    trading_days = [date.fromisoformat(item) for item in sorted(panel["date"].unique())]
    events = cpi_events(panel, trading_days)
    OUT.mkdir(parents=True, exist_ok=True)
    report = ROOT / "product" / "GOLDMAN_AU_AG.md"
    write_report(panel, seats, tests, events, report)
    panel.to_json(OUT / "precious_panel.json", orient="records", force_ascii=False)
    render_net(panel, OUT / "goldman_net.png")
    latest = panel["date"].max()
    print("latest", latest, "days", panel["date"].nunique())
    for product in PRODUCTS:
        row = panel[(panel["date"] == latest) & (panel["product"] == product)].iloc[0]
        print(product, "gs", row["gs_all"], row["gs_near"], "zc", row["zc_all"], row["zc_near"], "main", row["main"])
    for test in tests:
        print(test["product"], test["column"], "extreme", test["extreme"], "base", test["baseline"]["mean_ret"], "corr", test["corr_level_next"], test["corr_chg5_next"])


if __name__ == "__main__":
    main()
