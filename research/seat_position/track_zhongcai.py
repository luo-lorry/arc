#!/usr/bin/env python3
"""Track 中财期货 on SHFE gold and silver, by delivery month.

The exchange publishes, for each busy contract, the top 20 members by long
position, short position, and volume. A blank cell means 中财期货 was outside
that top 20. It does not mean the seat was flat. Quiet months have a quote
and no member ranking at all.

Numbers come from the SHFE daily files pm{yyyymmdd}.dat and kx{yyyymmdd}.dat.
The page is the day-session ranking. The night session that opens at 21:00
belongs to the next trading day's file.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "shfe"
OUT = ROOT / "product" / "zhongcai"
MEMBER = "中财期货"
SHANGHAI = ZoneInfo("Asia/Shanghai")
UA = {"User-Agent": "Mozilla/5.0"}
FONT_PATH = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"

BG = (7, 18, 34)
CARD = (12, 32, 56)
WHITE = (236, 242, 248)
MUTED = (154, 176, 196)
TEAL = (46, 196, 150)
RED = (255, 107, 107)
GREEN = (46, 196, 124)
AMBER = (232, 168, 74)


def http_json(url: str) -> dict | None:
    req = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=40) as resp:
            if resp.status != 200:
                return None
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def load_day(ymd: str) -> tuple[dict | None, dict | None]:
    DATA.mkdir(parents=True, exist_ok=True)
    pm_path = DATA / f"pm{ymd}.json"
    kx_path = DATA / f"kx{ymd}.json"
    if pm_path.exists():
        pm = json.loads(pm_path.read_text())
    else:
        pm = http_json(
            "https://www.shfe.com.cn/data/tradedata/future/dailydata/"
            f"pm{ymd}.dat"
        )
        if pm and pm.get("o_cursor"):
            pm_path.write_text(json.dumps(pm, ensure_ascii=False))
        else:
            pm = None
    if kx_path.exists():
        kx = json.loads(kx_path.read_text())
    else:
        kx = http_json(
            "https://www.shfe.com.cn/data/tradedata/future/dailydata/"
            f"kx{ymd}.dat"
        )
        if kx and kx.get("o_curinstrument"):
            kx_path.write_text(json.dumps(kx, ensure_ascii=False))
        else:
            kx = None
    return pm, kx


def num(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def contract_month(code: str) -> tuple[int, int]:
    return 2000 + int(code[-4:-2]), int(code[-2:])


def month_label(code: str) -> str:
    year, month = contract_month(code)
    return f"{year}年{month}月"


def quotes_of(kx: dict) -> dict[str, dict]:
    out = {}
    for item in kx.get("o_curinstrument") or []:
        group = str(item.get("PRODUCTGROUPID") or "").lower()
        month = str(item.get("DELIVERYMONTH") or "")
        if group not in {"au", "ag"} or not month.isdigit() or len(month) != 4:
            continue
        code = group + month
        out[code] = {
            "close": num(item.get("CLOSEPRICE")),
            "settle": num(item.get("SETTLEMENTPRICE")),
            "oi": num(item.get("OPENINTEREST")),
            "oi_chg": num(item.get("OPENINTERESTCHG")),
            "volume": num(item.get("VOLUME")),
        }
    return out


def seat_of(pm: dict) -> dict[str, dict]:
    """Per contract: 中财期货 long/short/volume, plus the 20th-place cutoff."""
    grouped: dict[str, list] = {}
    for row in pm.get("o_cursor") or []:
        code = str(row.get("INSTRUMENTID") or "").lower()
        if code.startswith("au") or code.startswith("ag"):
            if code.endswith("all"):
                continue
            grouped.setdefault(code, []).append(row)
    out = {}
    for code, rows in grouped.items():
        seat = {
            "ranked": True,
            "long": None,
            "long_chg": None,
            "long_rank": None,
            "short": None,
            "short_chg": None,
            "short_rank": None,
            "vol": None,
            "vol_chg": None,
            "vol_rank": None,
            "long_cut": None,
            "short_cut": None,
            "long_board": 0,
            "short_board": 0,
            "vol_board": 0,
        }
        for row in rows:
            rank = int(row.get("RANK") or 0)
            if rank < 1 or rank > 20:
                continue
            if str(row.get("PARTICIPANTABBR1") or "").strip() and rank >= seat["vol_board"]:
                seat["vol_board"] = rank
            if str(row.get("PARTICIPANTABBR2") or "").strip() and rank >= seat["long_board"]:
                seat["long_board"] = rank
                seat["long_cut"] = num(row.get("CJ2"))
            if str(row.get("PARTICIPANTABBR3") or "").strip() and rank >= seat["short_board"]:
                seat["short_board"] = rank
                seat["short_cut"] = num(row.get("CJ3"))
            sides = (
                ("vol", "PARTICIPANTABBR1", "CJ1", "CJ1_CHG"),
                ("long", "PARTICIPANTABBR2", "CJ2", "CJ2_CHG"),
                ("short", "PARTICIPANTABBR3", "CJ3", "CJ3_CHG"),
            )
            for side, name_key, qty_key, chg_key in sides:
                if str(row.get(name_key) or "") != MEMBER:
                    continue
                seat[side] = num(row.get(qty_key))
                seat[f"{side}_chg"] = num(row.get(chg_key))
                seat[f"{side}_rank"] = rank
        out[code] = seat
    return out


def build_day(ymd: str, pm: dict, kx: dict) -> list[dict]:
    day = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}"
    asof = date(int(ymd[:4]), int(ymd[4:6]), int(ymd[6:]))
    quotes = quotes_of(kx) if kx else {}
    seats = seat_of(pm)
    codes = sorted(set(quotes) | set(seats), key=contract_month)
    main = {}
    for product in ("au", "ag"):
        pool = [c for c in codes if c.startswith(product) and quotes.get(c, {}).get("oi")]
        if pool:
            main[product] = max(pool, key=lambda c: quotes[c]["oi"])
    rows = []
    for code in codes:
        year, month = contract_month(code)
        if (year, month) < (asof.year, asof.month):
            role = "交割月已过"
        elif (year, month) == (asof.year, asof.month):
            role = "当月"
        else:
            role = "后续"
        if code == main.get(code[:2]):
            role = "当月·主力" if role == "当月" else "主力"
        quote = quotes.get(code, {})
        seat = seats.get(code, {"ranked": False})
        oi = quote.get("oi")
        long_qty = seat.get("long")
        share = None
        if oi and long_qty is not None and oi > 0:
            share = long_qty / oi * 100.0
        rows.append(
            {
                "date": day,
                "code": code.upper(),
                "product": "沪金" if code.startswith("au") else "沪银",
                "month": month_label(code),
                "role": role,
                "close": quote.get("close"),
                "oi": oi,
                "oi_chg": quote.get("oi_chg"),
                "market_volume": quote.get("volume"),
                "long": long_qty,
                "long_chg": seat.get("long_chg"),
                "long_rank": seat.get("long_rank"),
                "short": seat.get("short"),
                "short_chg": seat.get("short_chg"),
                "short_rank": seat.get("short_rank"),
                "vol": seat.get("vol"),
                "vol_chg": seat.get("vol_chg"),
                "vol_rank": seat.get("vol_rank"),
                "long_cut": seat.get("long_cut"),
                "short_cut": seat.get("short_cut"),
                "long_board": seat.get("long_board") or 0,
                "short_board": seat.get("short_board") or 0,
                "vol_board": seat.get("vol_board") or 0,
                "ranked": bool(seat.get("ranked")),
                "share": share,
            }
        )
    return rows


def walk_days(start: date, end: date) -> list[list[dict]]:
    frames = []
    day = start
    while day <= end:
        if day.weekday() < 5:
            ymd = day.strftime("%Y%m%d")
            pm, kx = load_day(ymd)
            if pm:
                frames.append(build_day(ymd, pm, kx or {}))
                print(f"loaded {ymd} contracts={len(frames[-1])}")
            else:
                print(f"no ranking {ymd}")
        day += timedelta(days=1)
    if not frames:
        raise RuntimeError(f"no SHFE ranking files from {start} to {end}")
    return frames


def latest_session(today: date) -> date:
    day = today
    for _ in range(8):
        if day.weekday() < 5:
            pm, _kx = load_day(day.strftime("%Y%m%d"))
            if pm:
                return day
        day -= timedelta(days=1)
    raise RuntimeError("no recent SHFE ranking file")


def lots(value) -> str:
    if value is None:
        return "—"
    return f"{int(round(value)):,}"


def signed_lots(value) -> str:
    if value is None:
        return "—"
    number = int(round(value))
    return f"{number:+,}"


def price_text(value) -> str:
    if value is None:
        return "—"
    return f"{value:,.2f}"


def side_text(qty, rank, ranked: bool, board: int = 20) -> str:
    if not ranked:
        return "未公布排名"
    if qty is None:
        if board and board < 20:
            return f"不在榜（榜上{int(board)}名）"
        return "前20外"
    return f"{lots(qty)}（第{int(rank)}）"


def appeared(row: dict) -> bool:
    return row["long"] is not None or row["short"] is not None or row["vol"] is not None


def write_markdown(frames: list[list[dict]], path: Path) -> None:
    latest = frames[-1]
    day = latest[0]["date"]
    lines = [
        f"# 中财期货在沪金、沪银的持仓",
        "",
        f"交易日 {day}，日盘结束后的会员排名。当天 21:00 开始的夜盘记在下一个交易日。",
        "",
        "来源是上期所每日持仓排名 `pm` 和每日行情 `kx`。每个合约只公布多单、空单、成交量各自的前 20 名。表里的「前20外」表示这一侧没有进前 20，不是 0 手。持仓很薄的月份，交易所当天不公布会员排名。",
        "",
        "公布名称是「中财期货」，这份文件里没有再拆代客和自营。",
        "",
        "## 2026-10-13 这一天" if day == "2026-10-13" else f"## {day}",
        "",
    ]
    # fix header to use actual day always
    lines[-2] = f"## {day} 各合约"
    lines.append("")
    for product in ("沪金", "沪银"):
        rows = [row for row in latest if row["product"] == product]
        lines.append(f"### {product}")
        lines.append("")
        lines.append("| 合约 | 月份 | 角色 | 收盘 | 市场持仓 | 多单 | 多单增减 | 空单 | 空单增减 | 成交量排名 | 多单占持仓 |")
        lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in rows:
            share = "—" if row["share"] is None else f"{row['share']:.1f}%"
            lines.append(
                "| {code} | {month} | {role} | {close} | {oi} | {long} | {long_chg} | {short} | {short_chg} | {vol} | {share} |".format(
                    code=row["code"],
                    month=row["month"],
                    role=row["role"],
                    close=price_text(row["close"]),
                    oi=lots(row["oi"]),
                    long=side_text(row["long"], row["long_rank"], row["ranked"], row["long_board"]),
                    long_chg=signed_lots(row["long_chg"]) if row["long"] is not None else "—",
                    short=side_text(row["short"], row["short_rank"], row["ranked"], row["short_board"]),
                    short_chg=signed_lots(row["short_chg"]) if row["short"] is not None else "—",
                    vol=side_text(row["vol"], row["vol_rank"], row["ranked"], row["vol_board"]),
                    share=share,
                )
            )
        lines.append("")
        held = [row for row in rows if appeared(row)]
        if held:
            lines.append("榜上满 20 名时，中财没进去只说明这一侧低于第 20 名，具体手数没有公布。榜上不足 20 名时，已公布的席位里没有中财。")
            lines.append("")
            for row in rows:
                if not row["ranked"]:
                    continue
                bits = []
                if row["long"] is None and row["long_cut"] is not None:
                    if row["long_board"] < 20:
                        bits.append(f"多单榜 {int(row['long_board'])} 名，末位 {lots(row['long_cut'])} 手")
                    else:
                        bits.append(f"多单门槛 {lots(row['long_cut'])} 手")
                if row["short"] is None and row["short_cut"] is not None:
                    if row["short_board"] < 20:
                        bits.append(f"空单榜 {int(row['short_board'])} 名，末位 {lots(row['short_cut'])} 手")
                    else:
                        bits.append(f"空单门槛 {lots(row['short_cut'])} 手")
                if bits:
                    lines.append(f"- {row['code']}：" + "，".join(bits))
            lines.append("")
    lines.append("## 这段时间的变化")
    lines.append("")
    lines.append(
        f"区间 {frames[0][0]['date']} 至 {day}。下面只列中财至少有一天进过前 20 的合约。"
        "增减是上期所公布的当日增减。某一天变成「前20外」，表示那天没进前 20，中间的仓位没有公布。"
    )
    lines.append("")
    codes = []
    for frame in frames:
        for row in frame:
            if appeared(row) and row["code"] not in codes:
                codes.append(row["code"])
    codes.sort(key=lambda c: (c[:2], contract_month(c.lower())))
    for code in codes:
        series = []
        for frame in frames:
            found = next((row for row in frame if row["code"] == code), None)
            if found:
                series.append(found)
        if not series:
            continue
        first = next((row for row in series if row["long"] is not None), None)
        last = next((row for row in reversed(series) if row["long"] is not None), None)
        lines.append(f"### {code}")
        lines.append("")
        if first and last:
            delta = int(round(last["long"] - first["long"]))
            lines.append(
                f"多单从 {first['date']} 的 {lots(first['long'])} 手"
                f"到 {last['date']} 的 {lots(last['long'])} 手，"
                f"两端相差 {delta:+,} 手。"
            )
            lines.append("")
        lines.append("| 日期 | 多单 | 多单增减 | 空单 | 空单增减 | 成交量 |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for row in series:
            if not appeared(row):
                continue
            lines.append(
                f"| {row['date']} | {side_text(row['long'], row['long_rank'], True)} | "
                f"{signed_lots(row['long_chg']) if row['long'] is not None else '—'} | "
                f"{side_text(row['short'], row['short_rank'], True)} | "
                f"{signed_lots(row['short_chg']) if row['short'] is not None else '—'} | "
                f"{side_text(row['vol'], row['vol_rank'], True)} |"
            )
        lines.append("")
    lines.append("持仓排名只说明这个席位当天的多空和增减。不说明下一交易日价格方向，不构成投资建议。")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_html(frames: list[list[dict]], path: Path) -> None:
    latest = frames[-1]
    day = latest[0]["date"]

    def block(product: str) -> str:
        rows = []
        for row in latest:
            if row["product"] != product:
                continue
            share = "—" if row["share"] is None else f"{row['share']:.1f}%"
            long_chg = signed_lots(row["long_chg"]) if row["long"] is not None else "—"
            short_chg = signed_lots(row["short_chg"]) if row["short"] is not None else "—"
            long_color = "#ff6b6b" if (row["long_chg"] or 0) > 0 else "#2ec47c" if (row["long_chg"] or 0) < 0 else "#9db0c4"
            short_color = "#ff6b6b" if (row["short_chg"] or 0) > 0 else "#2ec47c" if (row["short_chg"] or 0) < 0 else "#9db0c4"
            rows.append(
                "<tr>"
                f"<td class='name'>{row['code']}<br><span>{row['month']}</span></td>"
                f"<td>{row['role']}</td>"
                f"<td>{price_text(row['close'])}</td>"
                f"<td>{lots(row['oi'])}</td>"
                f"<td>{side_text(row['long'], row['long_rank'], row['ranked'], row['long_board'])}</td>"
                f"<td style='color:{long_color}'>{long_chg}</td>"
                f"<td>{side_text(row['short'], row['short_rank'], row['ranked'], row['short_board'])}</td>"
                f"<td style='color:{short_color}'>{short_chg}</td>"
                f"<td>{share}</td>"
                "</tr>"
            )
        return (
            f"<h2>{product}</h2>"
            "<table><thead><tr><th>合约</th><th>角色</th><th>收盘</th><th>市场持仓</th>"
            "<th>中财多单</th><th>多单增减</th><th>中财空单</th><th>空单增减</th><th>多单占持仓</th>"
            "</tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table>"
        )

    codes = []
    for frame in frames:
        for row in frame:
            if appeared(row) and row["code"] not in codes:
                codes.append(row["code"])
    history_rows = []
    for frame in frames:
        cells = [f"<td>{frame[0]['date']}</td>"]
        by = {row["code"]: row for row in frame}
        for code in codes:
            row = by.get(code)
            if not row or row["long"] is None:
                cells.append("<td>—</td>")
            else:
                cells.append(f"<td>{lots(row['long'])}<br><span>{signed_lots(row['long_chg'])}</span></td>")
        history_rows.append("<tr>" + "".join(cells) + "</tr>")
    heads = "".join(f"<th>{code}</th>" for code in codes)
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>中财期货 沪金沪银 {day}</title>
  <style>
    body {{ margin: 0; background: #071222; color: #e8eef5; font-family: "PingFang SC", "Noto Sans SC", sans-serif; }}
    main {{ max-width: 1100px; margin: 0 auto; padding: 20px 16px 64px; }}
    h1 {{ font-size: 26px; margin: 8px 0; }}
    h2 {{ font-size: 18px; margin: 28px 0 8px; }}
    p {{ color: #9db0c4; font-size: 14px; line-height: 1.6; }}
    .scroll {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ text-align: right; padding: 8px 6px; border-bottom: 1px solid #163044; white-space: nowrap; }}
    th:first-child, td:first-child, td.name {{ text-align: left; }}
    td span {{ color: #7f93a8; }}
    img {{ width: 100%; border-radius: 12px; margin: 12px 0; display: block; }}
  </style>
</head>
<body>
  <main>
    <p>上期所会员持仓排名 · 每个交易日日盘后更新</p>
    <h1>中财期货 · 沪金与沪银 · {day}</h1>
    <p>当月合约和后面已上市的月份都列在下面。多单、空单只在进入该合约前 20 名时有手数。「前20外」不是 0 手。未公布排名的月份，是交易所当天没有发布这个合约的会员榜。公布名称是中财期货。不含当天 21:00 之后的夜盘。不构成投资建议。</p>
    <img src="latest.png" alt="当日持仓">
    {block("沪金")}
    {block("沪银")}
    <h2>多单逐日</h2>
    <p>单元格是多单手数，灰色数字是当日增减。区间 {frames[0][0]['date']} 至 {day}。只列出至少一天进过前 20 的合约。</p>
    <img src="history.png" alt="多单变化">
    <div class="scroll"><table><thead><tr><th>日期</th>{heads}</tr></thead><tbody>
    {''.join(history_rows)}
    </tbody></table></div>
  </main>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def font(size: int):
    return ImageFont.truetype(FONT_PATH, size)


def text_width(draw, value, face) -> int:
    return int(draw.textlength(value, font=face))


def render_latest(frames: list[list[dict]], path: Path) -> None:
    latest = [row for row in frames[-1] if row["role"] != "交割月已过"]
    width = 1400
    row_h = 46
    height = 210 + row_h * (len(latest) + 2)
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)
    title_font = font(40)
    small = font(20)
    row_font = font(22)
    day = frames[-1][0]["date"]
    title = f"中财期货 · 沪金沪银持仓 · {day}"
    draw.text(((width - text_width(draw, title, title_font)) / 2, 28), title, font=title_font, fill=WHITE)
    note = "前20外 = 这一侧未进前20，不是空仓。未公布 = 该合约当天没有会员排名。"
    draw.text(((width - text_width(draw, note, small)) / 2, 86), note, font=small, fill=TEAL)
    headers = ["合约", "角色", "收盘", "市场持仓", "多单", "增减", "空单", "增减", "多单/持仓"]
    xs = [36, 170, 300, 430, 580, 760, 900, 1080, 1220]
    y = 140
    for x, label in zip(xs, headers):
        draw.text((x, y), label, font=small, fill=MUTED)
    y += 36
    for row in latest:
        draw.rounded_rectangle((24, y, width - 24, y + row_h - 6), radius=8, fill=CARD)
        share = "—" if row["share"] is None else f"{row['share']:.1f}%"
        values = [
            row["code"],
            row["role"],
            price_text(row["close"]),
            lots(row["oi"]),
            side_text(row["long"], row["long_rank"], row["ranked"], row["long_board"]),
            signed_lots(row["long_chg"]) if row["long"] is not None else "—",
            side_text(row["short"], row["short_rank"], row["ranked"], row["short_board"]),
            signed_lots(row["short_chg"]) if row["short"] is not None else "—",
            share,
        ]
        for x, value in zip(xs, values):
            color = WHITE
            if value.startswith("+"):
                color = RED
            elif value.startswith("-") and value != "—":
                color = GREEN
            draw.text((x, y + 8), value, font=row_font, fill=color)
        y += row_h
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "PNG")


def render_history(frames: list[list[dict]], path: Path) -> None:
    codes = []
    for frame in frames:
        for row in frame:
            if row["long"] is not None and row["code"] not in codes:
                codes.append(row["code"])
    codes.sort(key=lambda c: (0 if c.startswith("AU") else 1, contract_month(c.lower())))
    if not codes:
        Image.new("RGB", (800, 200), BG).save(path, "PNG")
        return
    width = 1400
    height = 280 + 220 * ((len(codes) + 1) // 2)
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)
    title = font(36)
    small = font(18)
    draw.text((40, 24), "中财期货多单 · 进前20的那些合约", font=title, fill=WHITE)
    draw.text((40, 74), f"{frames[0][0]['date']} 至 {frames[-1][0]['date']}  ·  断档表示当天没进前20", font=small, fill=MUTED)
    panel_w = 640
    panel_h = 200
    for index, code in enumerate(codes):
        col = index % 2
        row_i = index // 2
        x0 = 40 + col * (panel_w + 40)
        y0 = 120 + row_i * (panel_h + 24)
        draw.rounded_rectangle((x0, y0, x0 + panel_w, y0 + panel_h), radius=12, fill=CARD)
        series = []
        for frame in frames:
            found = next((item for item in frame if item["code"] == code), None)
            series.append((frame[0]["date"], None if not found else found["long"]))
        known = [value for _day, value in series if value is not None]
        draw.text((x0 + 16, y0 + 12), code, font=font(22), fill=WHITE)
        if not known:
            continue
        if series[-1][1] is None:
            prev = next((item for item in reversed(series) if item[1] is not None))
            caption = f"{series[-1][0][5:]} 未上榜，上次 {prev[0][5:]} {int(round(prev[1])):,}"
        else:
            caption = f"最新 {int(round(series[-1][1])):,} 手"
        draw.text((x0 + 130, y0 + 14), caption, font=small, fill=AMBER)
        lo, hi = min(known), max(known)
        if hi == lo:
            hi = lo + 1
        left, right = x0 + 24, x0 + panel_w - 16
        top, bottom = y0 + 52, y0 + panel_h - 28
        segment = []
        for i, (_day, value) in enumerate(series):
            if value is None:
                if len(segment) >= 2:
                    draw.line(segment, fill=TEAL, width=3)
                segment = []
                continue
            x = left + (right - left) * (i / max(len(series) - 1, 1))
            y = bottom - (bottom - top) * ((value - lo) / (hi - lo))
            if segment:
                draw.line([segment[-1], (x, y)], fill=TEAL, width=3)
            segment.append((x, y))
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=WHITE)
        first = next(item for item in series if item[1] is not None)
        last_known = next(item for item in reversed(series) if item[1] is not None)
        draw.text((x0 + 16, y0 + panel_h - 24), f"{first[0][5:]} {int(round(first[1])):,}", font=small, fill=MUTED)
        end_label = f"{last_known[0][5:]} {int(round(last_known[1])):,}"
        draw.text((x0 + panel_w - 16 - text_width(draw, end_label, small), y0 + panel_h - 24), end_label, font=small, fill=MUTED)
    image.save(path, "PNG")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="")
    parser.add_argument("--end", default="")
    args = parser.parse_args()
    today = datetime.now(SHANGHAI).date()
    end = date.fromisoformat(args.end) if args.end else latest_session(today)
    start = date.fromisoformat(args.start) if args.start else end - timedelta(days=35)
    frames = walk_days(start, end)
    OUT.mkdir(parents=True, exist_ok=True)
    payload = [row for frame in frames for row in frame]
    (OUT / "series.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(frames, ROOT / "product" / "ZHONGCAI.md")
    write_html(frames, OUT / "index.html")
    render_latest(frames, OUT / "latest.png")
    render_history(frames, OUT / "history.png")
    latest = frames[-1]
    print(f"days {frames[0][0]['date']} -> {latest[0]['date']}")
    for row in latest:
        if row["product"] in {"沪金", "沪银"} and (appeared(row) or row["role"] in {"当月", "当月·主力", "主力"}):
            print(
                f"{row['code']} {row['role']} oi={row['oi']} "
                f"long={row['long']} {row['long_chg']} rank={row['long_rank']} "
                f"short={row['short']} {row['short_chg']} ranked={row['ranked']}"
            )


if __name__ == "__main__":
    main()
