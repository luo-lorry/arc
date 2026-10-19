#!/usr/bin/env python3
"""Build the daily 10-minute review: market scan, four quadrants, seat boards.

Update once per trading day, after the close. The scan and the quadrants use
the exchange daily tape (close versus the previous settlement, open-interest
change versus the previous open interest). Seat boards use the same member
rankings as render_boards.py and are attached when that feed answers.

Score, checked against the 2026-09-24 tape:

    score = |close / previous settlement - 1| + 0.4 * |open-interest change|

Percents are in percentage points. A volume add-on is not in the score; volume
is shown beside it. This page sorts and groups public numbers. It does not
say what to trade.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

import render_boards as boards
import run_study as study

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "product" / "review"
UA = {"User-Agent": "Mozilla/5.0"}

# Dalian and Guangzhou varieties are not in the SHFE/CZCE/CFFEX daily files
# used here. Sina's variety board lists the live contracts; the daily bar
# then supplies the previous settlement and the previous open interest.
SINA_NODES = {
    "bz_qh": "BZ",
    "byx_qh": "EB",
    "yec_qh": "EG",
    "tks_qh": "I",
    "jm_qh": "JM",
    "jt_qh": "J",
    "pvc_qh": "V",
    "lldpe_qh": "L",
    "jbx_qh": "PP",
    "dy_qh": "Y",
    "dd_qh": "A",
    "de_qh": "B",
    "dp_qh": "M",
    "hym_qh": "C",
    "ymdf_qh": "CS",
    "jd_qh": "JD",
    "lh_qh": "LH",
    "pg_qh": "PG",
    "zly_qh": "P",
    "gm_qh": "RR",
    "lg_qh": "LG",
    "jhb_qh": "BB",
    "xwb_qh": "FB",
    "lc_qh": "LC",
    "si_qh": "SI",
    "ps_qh": "PS",
    "pt_qh": "PT",
    "pd_qh": "PD",
}

NAMES = {
    **boards.VARIETY_NAME,
    "BZ": "纯苯",
    "LG": "原木",
    "PT": "铂",
    "PD": "钯",
    "IC": "中证500",
    "IM": "中证1000",
    "IF": "沪深300",
    "IH": "上证50",
}

# Drop a main contract this quiet. The reference card says "73 varieties";
# this cutoff lands near that count and removes contracts that barely traded.
MIN_VOLUME = 5000

BG = (7, 18, 34)
CARD = (12, 32, 56)
LINE = (28, 54, 84)
WHITE = (236, 242, 248)
MUTED = (154, 176, 196)
TEAL = (46, 196, 150)
AMBER = (232, 168, 74)
RED = (255, 107, 107)
GREEN = (46, 196, 124)


def http_bytes(url: str, timeout: int = 40) -> bytes:
    last = None
    for i in range(3):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(0.4 * (i + 1))
    raise RuntimeError(f"{url} failed: {last}")


def num(value) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if text in {"", "-", "--", "null"}:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    if out != out:
        return None
    return out


def product_of(code: str) -> str:
    match = re.match(r"([A-Za-z]+)", code)
    return match.group(1).upper() if match else code


def row(code, close, pre_settle, oi, oi_prev, volume, source) -> dict | None:
    if close is None or pre_settle is None or pre_settle == 0:
        return None
    if oi is None or oi_prev is None or oi_prev == 0 or volume is None:
        return None
    if volume <= 0 or oi <= 0:
        return None
    px = (close / pre_settle - 1.0) * 100.0
    oi_pct = (oi / oi_prev - 1.0) * 100.0
    score = abs(px) + 0.4 * abs(oi_pct)
    return {
        "code": code.upper(),
        "product": product_of(code),
        "close": close,
        "pre_settle": pre_settle,
        "px": px,
        "oi": oi,
        "oi_prev": oi_prev,
        "oi_pct": oi_pct,
        "volume": volume,
        "score": score,
        "source": source,
    }


def load_shfe(day: str) -> list[dict]:
    ymd = day.replace("-", "")
    url = f"https://www.shfe.com.cn/data/tradedata/future/dailydata/kx{ymd}.dat"
    payload = json.loads(http_bytes(url).decode("utf-8"))
    out = []
    for item in payload.get("o_curinstrument") or []:
        month = str(item.get("DELIVERYMONTH") or "")
        group = str(item.get("PRODUCTGROUPID") or "")
        if not re.fullmatch(r"\d{4}", month) or not re.fullmatch(r"[A-Za-z]+", group):
            continue
        close = num(item.get("CLOSEPRICE"))
        pre = num(item.get("PRESETTLEMENTPRICE"))
        oi = num(item.get("OPENINTEREST"))
        chg = num(item.get("OPENINTERESTCHG"))
        vol = num(item.get("VOLUME"))
        if oi is None or chg is None:
            continue
        made = row(group + month, close, pre, oi, oi - chg, vol, "shfe")
        if made:
            out.append(made)
    return out


def expand_czce(code: str, year: int) -> str | None:
    match = re.fullmatch(r"([A-Z]{1,2})(\d)(\d{2})", code.strip())
    if not match:
        return None
    digit = int(match.group(2))
    years = [y for y in range(year - 1, year + 9) if y % 10 == digit]
    if not years:
        return None
    return f"{match.group(1)}{years[0] % 100:02d}{match.group(3)}"


def load_czce(day: str) -> list[dict]:
    year, month, dd = day.split("-")
    url = (
        "http://www.czce.com.cn/cn/DFSStaticFiles/Future/"
        f"{year}/{year}{month}{dd}/FutureDataDaily.txt"
    )
    raw = http_bytes(url)
    text = raw.decode("gbk", errors="replace")
    out = []
    trade_year = int(year)
    for line in text.splitlines():
        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 12:
            continue
        code = expand_czce(parts[0], trade_year)
        if not code:
            continue
        pre = num(parts[1])
        close = num(parts[5])
        vol = num(parts[9])
        oi = num(parts[10])
        chg = num(parts[11])
        if oi is None or chg is None:
            continue
        made = row(code, close, pre, oi, oi - chg, vol, "czce")
        if made:
            out.append(made)
    return out


def load_cffex(day: str) -> list[dict]:
    year, month, dd = day.split("-")
    url = f"http://www.cffex.com.cn/sj/hqsj/rtj/{year}{month}/{dd}/index.xml"
    text = http_bytes(url).decode("utf-8", errors="replace")
    blocks = re.findall(r"<dailydata>(.*?)</dailydata>", text, flags=re.S)
    out = []
    for block in blocks:
        def field(name: str) -> str:
            found = re.search(rf"<{name}>(.*?)</{name}>", block)
            return found.group(1).strip() if found else ""

        code = field("instrumentid")
        if not re.fullmatch(r"[A-Z]{1,2}\d{4}", code):
            continue
        oi = num(field("openinterest"))
        prev = num(field("preopeninterest"))
        made = row(
            code,
            num(field("closeprice")),
            num(field("presettlementprice")),
            oi,
            prev,
            num(field("volume")),
            "cffex",
        )
        if made:
            out.append(made)
    return out


def sina_board(node: str) -> list[dict]:
    url = (
        "http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
        f"Market_Center.getHQFuturesData?page=1&num=80&sort=position&asc=0&node={node}"
    )
    payload = json.loads(http_bytes(url).decode("utf-8"))
    return payload if isinstance(payload, list) else []


def sina_bars(symbol: str) -> list[dict]:
    cache = DATA / "sina" / f"{symbol}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    url = study.SINA_KLINE.format(sym=symbol)
    text = http_bytes(url).decode("utf-8", errors="replace")
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end < start:
        return []
    bars = json.loads(text[start : end + 1])
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(bars, ensure_ascii=False))
    return bars


def sina_bar(symbol: str, day: str) -> dict | None:
    prev = None
    for bar in sina_bars(symbol):
        if bar.get("d") == day and prev is not None:
            return row(
                symbol,
                num(bar.get("c")),
                num(prev.get("s")),
                num(bar.get("p")),
                num(prev.get("p")),
                num(bar.get("v")),
                "sina",
            )
        prev = bar
    return None


def sina_live(item: dict, day: str) -> dict | None:
    """Use the live board when the daily bar is not in the history feed yet.

    Previous open interest comes from the last history bar before this session.
    """
    if str(item.get("tradedate") or "") != day:
        return None
    symbol = str(item.get("symbol") or "").upper()
    if not re.fullmatch(r"[A-Z]{1,2}\d{4}", symbol):
        return None
    prev_oi = None
    for bar in sina_bars(symbol):
        if str(bar.get("d") or "") < day:
            prev_oi = num(bar.get("p"))
    return row(
        symbol,
        num(item.get("close")),
        num(item.get("presettlement")),
        num(item.get("position")),
        prev_oi,
        num(item.get("volume")),
        "sina-live",
    )


def load_sina(day: str) -> list[dict]:
    listed: list[dict] = []
    for node in SINA_NODES:
        try:
            listed.extend(sina_board(node))
        except Exception as exc:  # noqa: BLE001
            print(f"node {node} skipped: {exc}")
    symbols = sorted(
        {
            str(item.get("symbol") or "").upper()
            for item in listed
            if re.fullmatch(r"[A-Z]{1,2}\d{4}", str(item.get("symbol") or "").upper())
        }
    )
    print(f"sina contracts {len(symbols)}")
    out = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(sina_bar, symbol, day): symbol for symbol in symbols}
        for fut in as_completed(futures):
            try:
                made = fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"kline {futures[fut]} skipped: {exc}")
                continue
            if made:
                out.append(made)
    have = {item["code"] for item in out}
    live = 0
    for item in listed:
        symbol = str(item.get("symbol") or "").upper()
        if symbol in have:
            continue
        try:
            made = sina_live(item, day)
        except Exception as exc:  # noqa: BLE001
            print(f"live {symbol} skipped: {exc}")
            continue
        if made:
            out.append(made)
            live += 1
    print(f"sina history={len(out) - live} live={live}")
    return out


def quote_table(day: str) -> pd.DataFrame:
    cache = DATA / "quotes" / f"{day}.json"
    if cache.exists():
        frame = pd.DataFrame(json.loads(cache.read_text()))
        print(f"load quotes {cache} rows={len(frame)}")
        return frame
    frames = []
    for name, loader in (
        ("shfe", load_shfe),
        ("czce", load_czce),
        ("cffex", load_cffex),
        ("sina", load_sina),
    ):
        try:
            rows = loader(day)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"{name} failed: {exc}") from exc
        print(f"{name} rows={len(rows)}")
        if not rows:
            raise RuntimeError(f"{name} returned no contracts for {day}")
        frames.extend(rows)
    if not frames:
        raise RuntimeError(f"no quotes for {day}")
    frame = pd.DataFrame(frames)
    # Official files win when Sina also has the contract.
    rank = {"shfe": 0, "czce": 0, "cffex": 0, "sina": 1, "sina-live": 1}
    frame["_src"] = frame["source"].map(rank).fillna(1)
    frame = frame.sort_values(["_src", "oi"], ascending=[True, False])
    frame = frame.drop_duplicates("code", keep="first").drop(columns="_src")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(frame.to_json(orient="records", force_ascii=False))
    print(f"saved {cache} rows={len(frame)}")
    return frame


def main_contracts(quotes: pd.DataFrame) -> pd.DataFrame:
    live = quotes[(quotes["volume"] >= MIN_VOLUME) & (quotes["oi"] > 0)].copy()
    live = live.sort_values(["product", "oi", "volume"], ascending=[True, False, False])
    mains = live.drop_duplicates("product", keep="first").copy()
    mains["score"] = mains["px"].abs() + 0.4 * mains["oi_pct"].abs()
    mains = mains.sort_values("score", ascending=False)
    mains["name"] = mains["product"].map(lambda p: NAMES.get(p, p))
    mains["tag"] = [
        tag_of(px, oi) for px, oi in zip(mains["px"], mains["oi_pct"])
    ]
    mains["rank"] = range(1, len(mains) + 1)
    return mains.reset_index(drop=True)


def tag_of(px: float, oi_pct: float) -> str:
    price = "上涨" if px > 0 else "下跌" if px < 0 else "平"
    book = "增仓" if oi_pct > 0 else "减仓" if oi_pct < 0 else "仓平"
    if price == "平" or book == "仓平":
        return "持平"
    return book + price


def price_text(value: float) -> str:
    if abs(value - round(value)) < 1e-6 and abs(value) >= 100:
        return f"{value:,.0f}"
    if abs(value) >= 1000:
        return f"{value:,.1f}"
    return f"{value:,.1f}"


def signed(value: float) -> str:
    return f"{value:+.2f}%"


def font(size: int):
    return boards.font(size)


def text_width(draw, value, face) -> int:
    return int(draw.textlength(value, font=face))


def render_scan(mains: pd.DataFrame, day: str, path: Path) -> None:
    width = 1080
    margin = 36
    rows = mains.head(10)
    height = 430 + 64 * len(rows)
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)
    title_font = font(48)
    sub_font = font(24)
    head_font = font(22)
    row_font = font(26)
    small_font = font(20)
    y = 40
    title = "全市场扫描 · 交易级排名"
    draw.text(((width - text_width(draw, title, title_font)) / 2, y), title, font=title_font, fill=WHITE)
    y += 70
    sub = f"交易日 {day}  ·  {len(mains)} 个品种各取持仓最大的合约"
    draw.text(((width - text_width(draw, sub, sub_font)) / 2, y), sub, font=sub_font, fill=MUTED)
    y += 40
    formula = "评分 = |收盘相对昨结算| + 0.4 × |持仓变化|    成交量只展示，不加进评分"
    draw.text(((width - text_width(draw, formula, small_font)) / 2, y), formula, font=small_font, fill=TEAL)
    y += 48
    headers = [(56, "#"), (110, "品种"), (430, "收盘"), (600, "涨跌"), (750, "持仓"), (900, "评分")]
    for x, label in headers:
        draw.text((x, y), label, font=head_font, fill=MUTED)
    y += 40
    for record in rows.itertuples(index=False):
        draw.rounded_rectangle((margin, y, width - margin, y + 56), radius=10, fill=CARD)
        draw.text((56, y + 14), str(record.rank), font=row_font, fill=MUTED)
        label = f"{record.name} {record.code}"
        draw.text((110, y + 14), label, font=row_font, fill=WHITE)
        draw.text((430, y + 14), price_text(record.close), font=row_font, fill=WHITE)
        px_color = RED if record.px > 0 else GREEN
        draw.text((590, y + 14), signed(record.px), font=row_font, fill=px_color)
        oi_color = RED if record.oi_pct > 0 else GREEN
        draw.text((730, y + 14), signed(record.oi_pct), font=row_font, fill=oi_color)
        draw.text((900, y + 14), f"{record.score:.2f}", font=row_font, fill=AMBER)
        y += 64
    foot = "数据源：上期所、能源中心、郑商所、中金所日行情，大商所与广期所用新浪日线。不构成投资建议。"
    draw.text((margin, height - 48), foot, font=small_font, fill=MUTED)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "PNG")


QUADRANTS = [
    ("增仓上涨", "价涨，持仓增加", TEAL),
    ("减仓上涨", "价涨，持仓减少", AMBER),
    ("增仓下跌", "价跌，持仓增加", RED),
    ("减仓下跌", "价跌，持仓减少", GREEN),
]


def render_quadrants(mains: pd.DataFrame, day: str, path: Path) -> None:
    width = 1080
    height = 1280
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)
    title_font = font(48)
    sub_font = font(24)
    name_font = font(32)
    row_font = font(24)
    small_font = font(20)
    y = 40
    title = "量价仓四象限"
    draw.text(((width - text_width(draw, title, title_font)) / 2, y), title, font=title_font, fill=WHITE)
    y += 68
    sub = f"交易日 {day}  ·  同一天的涨跌，持仓方向不同就分开"
    draw.text(((width - text_width(draw, sub, sub_font)) / 2, y), sub, font=sub_font, fill=MUTED)
    y += 56
    box_w = 480
    box_h = 500
    gap = 28
    left = 48
    for index, (name, caption, color) in enumerate(QUADRANTS):
        col = index % 2
        row_i = index // 2
        x0 = left + col * (box_w + gap)
        y0 = y + row_i * (box_h + gap)
        draw.rounded_rectangle((x0, y0, x0 + box_w, y0 + box_h), radius=18, fill=CARD)
        draw.rounded_rectangle((x0 + 24, y0 + 28, x0 + 40, y0 + 68), radius=4, fill=color)
        chunk = mains[mains["tag"] == name]
        draw.text((x0 + 56, y0 + 26), f"{name}  {len(chunk)}", font=name_font, fill=WHITE)
        draw.text((x0 + 56, y0 + 74), caption, font=small_font, fill=MUTED)
        yy = y0 + 130
        for record in chunk.head(6).itertuples(index=False):
            draw.text((x0 + 28, yy), f"{record.name} {record.code}", font=row_font, fill=WHITE)
            line = f"{signed(record.px)}   仓{signed(record.oi_pct)}"
            draw.text((x0 + 28, yy + 32), line, font=small_font, fill=MUTED)
            yy += 58
    foot = "四格只描述当天收盘价和持仓是否同向。不推断下一交易日。"
    draw.text(((width - text_width(draw, foot, small_font)) / 2, height - 48), foot, font=small_font, fill=MUTED)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "PNG")


def render_cover(day: str, n_products: int, path: Path) -> None:
    size = 1080
    image = Image.new("RGB", (size, size), BG)
    draw = ImageDraw.Draw(image)
    title_font = font(64)
    sub_font = font(30)
    body_font = font(32)
    draw.rounded_rectangle((64, 80, size - 64, size - 80), radius=28, fill=CARD)
    y = 160
    for line in ("10分钟盘后复盘", day):
        draw.text(((size - text_width(draw, line, title_font)) / 2, y), line, font=title_font, fill=WHITE)
        y += 86
    y += 20
    sub = f"{n_products} 个品种 · 每天收盘后更新 1 次"
    draw.text(((size - text_width(draw, sub, sub_font)) / 2, y), sub, font=sub_font, fill=TEAL)
    y += 100
    lines = [
        "1  全市场扫描    按评分看前 10",
        "2  量价仓四象限  涨跌和持仓是否同向",
        "3  三组席位      家人 / 机构 / 外资",
    ]
    for line in lines:
        draw.text((140, y), line, font=body_font, fill=WHITE)
        y += 72
    y += 40
    note = "公开数据整理。不喊单，不构成投资建议。"
    draw.text(((size - text_width(draw, note, sub_font)) / 2, y), note, font=sub_font, fill=MUTED)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "PNG")


def attach_seats(mains: pd.DataFrame, day: str, folder: Path) -> bool:
    frames = []
    for org in study.ORGS:
        try:
            rows = study.fetch_org_day(org, day)
        except Exception as exc:  # noqa: BLE001
            print(f"seat {org} failed: {exc}")
            return False
        if rows:
            frames.append(pd.DataFrame(rows))
        print(f"seat {org} rows={len(rows)}")
    if not frames:
        return False
    seats = study.clean_seats(pd.concat(frames, ignore_index=True))
    main = pd.DataFrame(
        {
            "date": day,
            "main_code": mains["code"],
            "variety": mains["product"],
            "oi": mains["oi"],
        }
    )
    day_pack = boards.build_day(seats, main, day)
    for board in day_pack["boards"]:
        boards.render_board(day_pack, board, folder / f"{board['key']}.png")
    (folder / "seats.json").write_text(
        json.dumps(day_pack, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"seats {day_pack['n_with_rank']}/{day_pack['n_main']} "
        f"boards={[b['title'] for b in day_pack['boards']]}"
    )
    return True


def html_page(mains: pd.DataFrame, day: str, has_seats: bool) -> str:
    def cells(record) -> str:
        px_color = "#ff6b6b" if record.px > 0 else "#2ec47c"
        oi_color = "#ff6b6b" if record.oi_pct > 0 else "#2ec47c"
        return (
            "<tr>"
            f"<td>{record.rank}</td>"
            f"<td>{record.name}<br><span>{record.code}</span></td>"
            f"<td>{price_text(record.close)}</td>"
            f"<td style='color:{px_color}'>{signed(record.px)}</td>"
            f"<td style='color:{oi_color}'>{signed(record.oi_pct)}</td>"
            f"<td>{record.volume:,.0f}</td>"
            f"<td>{record.score:.2f}</td>"
            f"<td>{record.tag}</td>"
            "</tr>"
        )

    body = "\n".join(cells(record) for record in mains.itertuples(index=False))
    quads = []
    for name, caption, _color in QUADRANTS:
        chunk = mains[mains["tag"] == name]
        items = "".join(
            f"<li>{record.name} {record.code}　{signed(record.px)}　仓{signed(record.oi_pct)}</li>"
            for record in chunk.head(5).itertuples(index=False)
        )
        quads.append(
            f"<section class='quad'><h3>{name} · {len(chunk)}</h3><p>{caption}</p><ol>{items}</ol></section>"
        )
    seat_block = ""
    if has_seats:
        seat_block = """
    <h2>3 三组席位</h2>
    <p>主力合约上，三大家人、四大机构、三大外资当天增仓和减仓各 10 条。单位是手。</p>
    <img src="retail.png" alt="三大家人">
    <img src="inst.png" alt="四大机构">
    <img src="foreign.png" alt="三大外资">
"""
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>10分钟盘后复盘 {day}</title>
  <style>
    body {{ margin: 0; background: #071222; color: #e8eef5; font-family: "PingFang SC", "Noto Sans SC", sans-serif; }}
    main {{ max-width: 880px; margin: 0 auto; padding: 20px 16px 64px; }}
    h1 {{ font-size: 26px; margin: 8px 0; }}
    h2 {{ font-size: 18px; margin: 28px 0 8px; }}
    p, li {{ color: #9db0c4; font-size: 14px; line-height: 1.6; }}
    img {{ width: 100%; border-radius: 12px; margin: 12px 0; display: block; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ text-align: right; padding: 8px 4px; border-bottom: 1px solid #163044; }}
    th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
    td span {{ color: #7f93a8; }}
    .quads {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
    .quad {{ background: #0c2038; border-radius: 12px; padding: 12px 14px; }}
    .quad h3 {{ margin: 0; font-size: 16px; }}
    .quad ol {{ margin: 8px 0 0; padding-left: 18px; }}
    @media (max-width: 640px) {{ .quads {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <main>
    <p>期货盘后复盘 · 每个交易日更新 1 次</p>
    <h1>{day}</h1>
    <p>先看评分前 10，再看价格和持仓是否同向，最后看三组席位动了谁。评分 = |收盘相对昨结算| + 0.4 × |持仓变化|。成交量写在表里，不加进评分。不下结论，不构成投资建议。</p>
    <img src="cover.png" alt="封面">
    <h2>1 全市场扫描</h2>
    <p>{len(mains)} 个品种，各取当天持仓最大的合约，成交量低于 {MIN_VOLUME:,} 手的不进榜。按评分从高到低。</p>
    <img src="scan.png" alt="扫描前10">
    <table>
      <thead><tr><th>#</th><th>品种</th><th>收盘</th><th>涨跌</th><th>持仓</th><th>成交量</th><th>评分</th><th>状态</th></tr></thead>
      <tbody>
      {body}
      </tbody>
    </table>
    <h2>2 量价仓四象限</h2>
    <p>同一涨跌方向，持仓增加和持仓减少分开看。格子里是这个状态里评分最高的几个合约。</p>
    <img src="quadrants.png" alt="四象限">
    <div class="quads">
      {''.join(quads)}
    </div>
    {seat_block}
    <p>扫描和四象限来自上期所（含能源中心）、郑商所、中金所每日行情，大商所和广期所来自新浪日线。席位来自交易所公布的会员持仓排名。持仓变化相对上一交易日。休市日页面留着上一个交易日。</p>
    <p>发给客户、也请客户收藏的固定链接是 <a href="https://luo-lorry.github.io/arc/">https://luo-lorry.github.io/arc/</a>。不要收藏带日期的这一页，带日期的地址只保留 {day}。</p>
  </main>
</body>
</html>
"""


def check_known_day(mains: pd.DataFrame, day: str) -> None:
    if day != "2026-09-24":
        return
    br = mains[mains["code"] == "BR2611"]
    if br.empty:
        print("CHECK BR2611 missing")
        return
    score = float(br.iloc[0]["score"])
    rank = int(br.iloc[0]["rank"])
    print(f"CHECK BR2611 score={score:.2f} rank={rank} close={br.iloc[0]['close']}")
    if abs(score - 12.72) > 0.05:
        print("CHECK BR2611 score is outside 12.72 ± 0.05")
    if rank != 1:
        print(f"CHECK BR2611 rank is {rank}, expected 1 on this tape")


def build(day: str, skip_seats: bool) -> bool:
    quotes = quote_table(day)
    mains = main_contracts(quotes)
    folder = OUT / day
    folder.mkdir(parents=True, exist_ok=True)
    render_cover(day, len(mains), folder / "cover.png")
    render_scan(mains, day, folder / "scan.png")
    render_quadrants(mains, day, folder / "quadrants.png")
    has_seats = False
    if not skip_seats:
        has_seats = attach_seats(mains, day, folder)
    payload = mains[
        ["rank", "name", "product", "code", "close", "px", "oi_pct", "volume", "oi", "score", "tag", "source"]
    ].copy()
    payload.to_json(folder / "scan.json", orient="records", force_ascii=False, indent=2)
    page = html_page(mains, day, has_seats)
    (folder / "index.html").write_text(page, encoding="utf-8")
    # The stable link is what a buyer bookmarks. Replace it only when the
    # seat boards are in, so a late ranking feed cannot wipe yesterday's page.
    # index.html is a short loader. It reads latest.json on every open, so a
    # phone that cached yesterday's shell still lands on the new trading day.
    if has_seats:
        (OUT / "latest.json").write_text(
            json.dumps({"day": day}, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (OUT / "index.html").write_text(stable_loader(day), encoding="utf-8")
        for name in ("cover.png", "scan.png", "quadrants.png", "retail.png", "inst.png", "foreign.png"):
            src = folder / name
            if src.exists():
                (OUT / name).write_bytes(src.read_bytes())
        print(f"stable link now opens {day}")
    else:
        print("seats missing, left the stable link on the previous complete day")
    check_known_day(mains, day)
    print(f"products {len(mains)} wrote {folder}")
    print(mains.head(10)[["rank", "code", "close", "px", "oi_pct", "score", "tag"]].to_string(index=False))
    return has_seats


def stable_loader(day: str) -> str:
    """Bookmark this file. Each open asks latest.json which day to show."""
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>期货盘后复盘</title>
  <style>
    body {{ margin: 0; background: #071222; color: #e8eef5; font-family: "PingFang SC", "Noto Sans SC", sans-serif; }}
    main {{ max-width: 560px; margin: 0 auto; padding: 48px 16px; }}
    a {{ color: #7fd1ff; }}
    p {{ color: #9db0c4; line-height: 1.6; }}
  </style>
</head>
<body>
  <main>
    <p>正在打开最近一个交易日的复盘。休市日打开上一个交易日。</p>
    <p><a id="go" href="{day}/index.html">打开 {day}</a></p>
  </main>
  <script>
    fetch("latest.json?t=" + Date.now(), {{cache: "no-store"}})
      .then(function (response) {{ return response.json(); }})
      .then(function (data) {{
        var url = data.day + "/index.html";
        var link = document.getElementById("go");
        link.href = url;
        link.textContent = "打开 " + data.day;
        location.replace(url);
      }});
  </script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="2026-09-24")
    parser.add_argument("--skip-seats", action="store_true")
    parser.add_argument(
        "--require-seats",
        action="store_true",
        help="exit 2 when the stable link was left on the previous day",
    )
    args = parser.parse_args()
    updated = build(args.date, args.skip_seats)
    if args.require_seats and not updated:
        sys.exit(2)


if __name__ == "__main__":
    main()
