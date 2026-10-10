#!/usr/bin/env python3
"""Render the three daily seat boards as phone images.

One trading day, one update. Exchange member rankings are published after the
day session, so the pictures are built once that table is out, before the
night session. This is the deliverable for the Xianyu listing: three images
a buyer can read without opening a website.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import run_study as study

OUT = study.ROOT / "product" / "samples"
FONT_PATH = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"

# Display names match the reference boards. Stored org names stay unchanged.
DISPLAY_ORG = {
    "平安期货": "平安期货",
    "东方财富期货": "东方财富期货",
    "徽商期货": "徽商期货",
    "东证期货": "东证期货",
    "中信期货": "中信期货",
    "国泰君安": "国泰君安期货",
    "银河期货": "银河期货",
    "摩根大通": "摩根大通期货",
    "高盛期货": "高盛期货",
}

VARIETY_NAME = {
    "A": "豆一", "AD": "铸造铝", "AG": "白银", "AL": "沪铝", "AO": "氧化铝",
    "AP": "苹果", "AU": "黄金", "B": "豆二", "BB": "胶合板", "BC": "国际铜",
    "BR": "丁二烯橡胶", "BU": "沥青", "C": "玉米", "CF": "棉花", "CJ": "红枣",
    "CS": "淀粉", "CU": "沪铜", "CY": "棉纱", "EB": "苯乙烯", "EC": "集运欧线",
    "EG": "乙二醇", "FB": "纤维板", "FG": "玻璃", "FU": "燃料油", "HC": "热轧卷板",
    "I": "铁矿石", "IC": "中证500", "IF": "沪深300", "IH": "上证50", "IM": "中证1000",
    "J": "焦炭", "JD": "鸡蛋", "JM": "焦煤", "L": "塑料", "LC": "碳酸锂",
    "LH": "生猪", "LU": "低硫燃油", "M": "豆粕", "MA": "甲醇", "NI": "沪镍",
    "NR": "20号胶", "OI": "菜油", "P": "棕榈油", "PB": "沪铅", "PF": "短纤",
    "PG": "液化气", "PK": "花生", "PL": "丙烯", "PP": "聚丙烯", "PR": "瓶片",
    "PS": "多晶硅", "PX": "对二甲苯", "RB": "螺纹钢", "RM": "菜粕", "RR": "粳米",
    "RS": "菜籽", "RU": "橡胶", "SA": "纯碱", "SC": "原油", "SF": "硅铁",
    "SH": "烧碱", "SI": "工业硅", "SM": "锰硅", "SN": "沪锡", "SP": "纸浆",
    "SR": "白糖", "SS": "不锈钢", "T": "10年国债", "TA": "PTA", "TF": "5年国债",
    "TL": "30年期国债期货", "TS": "2年国债", "UR": "尿素", "V": "PVC",
    "Y": "豆油", "ZN": "沪锌",
}

GROUPS = [
    {
        "key": "retail",
        "title": "三大家人",
        "orgs": study.RETAIL,
        "blurb": "散户喜欢扛单，尤其是逆势扛单",
    },
    {
        "key": "inst",
        "title": "四大机构",
        "orgs": study.INST,
        "blurb": "客户里有量化、套保和产业，方向要分开看",
    },
    {
        "key": "foreign",
        "title": "三大外资",
        "orgs": study.FOREIGN,
        "blurb": "主攻和外盘有关的商品",
    },
]

BG = (7, 18, 34)
CARD = (12, 32, 56)
LINE = (24, 52, 82)
WHITE = (236, 242, 248)
MUTED = (154, 176, 196)
BANNER = (14, 92, 78)
BANNER_TEXT = (111, 214, 176)
TEAL = (16, 163, 140)
RED = (214, 78, 78)
PURPLE = (124, 92, 196)
GREEN = (36, 150, 112)
UP = (255, 92, 96)
DOWN = (46, 196, 124)
SECTION = (46, 196, 150)


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_PATH, size)


def product_label(code: str, variety: str) -> str:
    name = VARIETY_NAME.get(variety, variety)
    return f"{name} {code}"


def comma(n: int) -> str:
    sign = "+" if n > 0 else ""
    return f"{sign}{n:,}"


def build_day(seats: pd.DataFrame, main: pd.DataFrame, date: str) -> dict:
    mains = main.loc[main["date"] == date, ["main_code", "variety", "oi"]]
    main_codes = set(mains["main_code"])
    events = study.events_from_rows(seats)
    day_events = events[events["date"] == date]
    seen = set(day_events.loc[day_events["code"].isin(main_codes), "code"])
    boards = []
    for group in GROUPS:
        on_main = day_events[
            day_events["org"].isin(group["orgs"]) & day_events["code"].isin(main_codes)
        ]
        add = study.top10(events[events["code"].isin(main_codes)], group["orgs"], date, "add")
        cut = study.top10(events[events["code"].isin(main_codes)], group["orgs"], date, "cut")
        boards.append(
            {
                "key": group["key"],
                "title": group["title"],
                "blurb": group["blurb"],
                "listed": int(len(on_main)),
                "add": rows_of(add),
                "cut": rows_of(cut),
            }
        )
    return {
        "date": date,
        "n_main": int(len(main_codes)),
        "n_with_rank": int(len(seen)),
        "boards": boards,
    }


def rows_of(frame: pd.DataFrame) -> list:
    rows = []
    for record in frame.itertuples(index=False):
        rows.append(
            {
                "side": record.side,
                "org": DISPLAY_ORG.get(record.org, record.org),
                "label": product_label(record.code, record.variety),
                "change": int(round(record.change)),
            }
        )
    return rows


def text_width(draw: ImageDraw.ImageDraw, value: str, face: ImageFont.FreeTypeFont) -> int:
    return int(draw.textlength(value, font=face))


def pill_color(side: str) -> tuple:
    return {"加多": RED, "加空": TEAL, "减多": GREEN, "减空": PURPLE}[side]


def render_board(day: dict, board: dict, path: Path) -> None:
    width = 1080
    margin = 36
    row_h = 58
    height = 250 + 86 + 46 + 10 * row_h + 70 + 10 * row_h + 150
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)
    title_font = font(46)
    sub_font = font(22)
    banner_font = font(24)
    name_font = font(34)
    blurb_font = font(20)
    section_font = font(26)
    row_font = font(24)
    small_font = font(18)

    y = 42
    title = f"龙虎榜持仓归类 · {board['title']}"
    draw.text(((width - text_width(draw, title, title_font)) / 2, y), title, font=title_font, fill=WHITE)
    y += 68
    sub = f"交易日 {day['date']}  |  主力合约  |  增仓/减仓各取 TOP10"
    draw.text(((width - text_width(draw, sub, sub_font)) / 2, y), sub, font=sub_font, fill=MUTED)
    y += 48
    banner = f"数据完整：{day['n_with_rank']}/{day['n_main']} 个主力合约见到跟踪席位"
    draw.rounded_rectangle((margin, y, width - margin, y + 54), radius=8, fill=BANNER)
    draw.text(
        ((width - text_width(draw, banner, banner_font)) / 2, y + 12),
        banner,
        font=banner_font,
        fill=BANNER_TEXT,
    )
    y += 74
    card_top = y
    card_bottom = height - 118
    draw.rounded_rectangle((margin, card_top, width - margin, card_bottom), radius=16, fill=CARD)
    y += 22
    draw.text((margin + 28, y), board["title"], font=name_font, fill=WHITE)
    blurb_x = margin + 28 + text_width(draw, board["title"], name_font) + 16
    draw.text((blurb_x, y + 10), board["blurb"], font=blurb_font, fill=MUTED)
    listed = f"当日上榜席位 {board['listed']} 条"
    draw.text((width - margin - 28 - text_width(draw, listed, small_font), y + 10), listed, font=small_font, fill=MUTED)
    y += 58
    y = draw_section(draw, y, width, margin, "增仓 TOP10（加多/加空）", board["add"], section_font, row_font, row_h)
    y += 16
    draw_section(draw, y, width, margin, "减仓 TOP10（减多/减空）", board["cut"], section_font, row_font, row_h)

    foot1 = "数据源：东方财富数据中心会员持仓排名。每个交易日收盘后更新一次。"
    foot2 = "统计口径为主力合约、代客席位。数据仅供参考，不构成投资建议。排名未齐时以图中数据完整一行说明。"
    draw.text(((width - text_width(draw, foot1, small_font)) / 2, height - 92), foot1, font=small_font, fill=MUTED)
    draw.text(((width - text_width(draw, foot2, small_font)) / 2, height - 62), foot2, font=small_font, fill=(120, 140, 158))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "PNG")


def draw_section(draw, y, width, margin, header, rows, section_font, row_font, row_h) -> int:
    draw.text((margin + 28, y), header, font=section_font, fill=SECTION)
    y += 42
    inner_left = margin + 16
    inner_right = width - margin - 16
    for index, row in enumerate(rows, start=1):
        if index > 1:
            draw.line((inner_left + 12, y, inner_right - 12, y), fill=LINE, width=1)
        rank = str(index)
        draw.text((inner_left + 18, y + 16), rank, font=row_font, fill=MUTED)
        color = pill_color(row["side"])
        pill = (inner_left + 52, y + 12, inner_left + 124, y + 46)
        draw.rounded_rectangle(pill, radius=8, fill=color)
        side_w = text_width(draw, row["side"], row_font)
        draw.text((pill[0] + (72 - side_w) / 2, y + 15), row["side"], font=row_font, fill=WHITE)
        draw.text((inner_left + 140, y + 16), row["org"], font=row_font, fill=WHITE)
        draw.text((inner_left + 360, y + 16), row["label"], font=row_font, fill=(210, 224, 236))
        change = comma(row["change"])
        change_color = UP if row["change"] > 0 else DOWN
        draw.text((inner_right - 18 - text_width(draw, change, row_font), y + 16), change, font=row_font, fill=change_color)
        y += row_h
    return y


def render_cover(path: Path) -> None:
    size = 1080
    image = Image.new("RGB", (size, size), BG)
    draw = ImageDraw.Draw(image)
    title_font = font(64)
    sub_font = font(32)
    body_font = font(28)
    small_font = font(22)
    draw.rounded_rectangle((64, 80, size - 64, size - 80), radius=28, fill=CARD)
    y = 150
    for line in ("期货龙虎榜", "三组席位日报"):
        draw.text(((size - text_width(draw, line, title_font)) / 2, y), line, font=title_font, fill=WHITE)
        y += 84
    y += 10
    sub = "每个交易日更新 1 次 · 夜盘前发出"
    draw.text(((size - text_width(draw, sub, sub_font)) / 2, y), sub, font=sub_font, fill=SECTION)
    y += 90
    cards = [
        (TEAL, "三大家人", "平安 · 东方财富 · 徽商"),
        (RED, "四大机构", "东证 · 中信 · 国泰君安 · 银河"),
        (PURPLE, "三大外资", "摩根大通 · 高盛"),
    ]
    for color, name, members in cards:
        draw.rounded_rectangle((140, y, 196, y + 56), radius=10, fill=color)
        draw.text((220, y + 8), name, font=body_font, fill=WHITE)
        draw.text((460, y + 12), members, font=small_font, fill=MUTED)
        y += 84
    y += 36
    note = "看当日谁在加减仓。不喊单，不保证收益。"
    draw.text(((size - text_width(draw, note, body_font)) / 2, y), note, font=body_font, fill=MUTED)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "PNG")


def load_frames():
    raw = study.download_seats()
    seats = study.clean_seats(raw)
    study.download_prices(seats["code"].unique().tolist())
    prices = study.load_prices(seats["code"].unique().tolist())
    return seats, study.main_contract_table(prices)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="2026-10-09")
    args = parser.parse_args()
    seats, main = load_frames()
    day = build_day(seats, main, args.date)
    OUT.mkdir(parents=True, exist_ok=True)
    written = []
    cover = OUT / "00-cover.png"
    render_cover(cover)
    written.append(cover)
    for board in day["boards"]:
        path = OUT / f"{args.date}-{board['key']}.png"
        render_board(day, board, path)
        written.append(path)
        print(
            f"{board['title']} listed={board['listed']} "
            f"add={len(board['add'])} cut={len(board['cut'])}"
        )
    (OUT / f"{args.date}.json").write_text(
        json.dumps(day, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"mains {day['n_with_rank']}/{day['n_main']}")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
