#!/usr/bin/env python3
"""Retail-seat position-flow study for Chinese futures long-hu-bang data.

The three reference boards (三大家人 / 四大机构 / 三大外资) are a filter and
a sort on public member position rankings. This script:

1. Downloads 代客 member long/short changes from Eastmoney datacenter.
2. Checks those changes against the lots printed on the reference boards.
3. Rebuilds the top-10 add/cut lists on the main contract of each variety.
4. Tests pre-specified, scale-free rules. Raw lot rankings are only used to
   match the pictures. The trading signal divides net lot flow by open interest
   so a large contract is not selected just because it has more lots.

Primary hypothesis, locked before the backtest is read:
  三大家人 (平安期货, 东方财富期货, 徽商期货) net flow is a contrarian signal.
  四大机构 flow is a weaker, less stable directional signal because the book
  mixes quant, hedge and brokerage flow.
  三大外资 is reported separately and is not pooled with either group.

Not investment advice. Top-20 disclosure censors seats that fall off the board.
"""

from __future__ import annotations

import json
import math
import re
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
PRICE_DIR = DATA / "prices"
RESULT = ROOT / "results"

EASTMONEY = "https://datacenter-web.eastmoney.com/api/data/v1/get"
SINA_KLINE = (
    "https://stock2.finance.sina.com.cn/futures/api/jsonp.php/"
    "var%20_{sym}=/InnerFuturesNewService.getDailyKLine?symbol={sym}"
)

RETAIL = ["平安期货", "东方财富期货", "徽商期货"]
INST = ["东证期货", "中信期货", "国泰君安", "银河期货"]
FOREIGN = ["摩根大通", "高盛期货"]
ORGS = RETAIL + INST + FOREIGN
FINANCIAL = {"IF", "IH", "IC", "IM", "TS", "TF", "T", "TL"}

# Lots copied from the three reference screenshots. Side is the board label.
# change is signed lots on that side (加 is positive, 减 is negative).
BOARDS = [
    {
        "name": "三大家人",
        "date": "2026-09-24",
        "group": RETAIL,
        "add": [
            ("加空", "平安期货", "EG2611", 2556),
            ("加空", "东方财富期货", "BU2611", 2487),
            ("加空", "徽商期货", "MA2701", 2455),
            ("加多", "徽商期货", "SA2701", 2438),
            ("加空", "平安期货", "PP2701", 2041),
            ("加空", "平安期货", "BU2611", 1555),
            ("加多", "徽商期货", "FG2701", 1257),
            ("加空", "平安期货", "FU2611", 1203),
            ("加空", "平安期货", "MA2701", 1194),
            ("加空", "平安期货", "EB2611", 1020),
        ],
        "cut": [
            ("减空", "平安期货", "FG2701", -2897),
            ("减多", "东方财富期货", "LH2611", -1238),
            ("减多", "徽商期货", "SH2611", -1229),
            ("减多", "东方财富期货", "SH2611", -834),
            ("减空", "平安期货", "TL2612", -822),
            ("减多", "徽商期货", "UR2701", -797),
            ("减多", "徽商期货", "RB2701", -747),
            ("减多", "东方财富期货", "AO2701", -676),
            ("减多", "平安期货", "RM2701", -623),
            ("减多", "平安期货", "C2611", -537),
        ],
    },
    {
        "name": "四大机构",
        "date": "2026-10-09",
        "group": INST,
        "add": [
            ("加多", "东证期货", "RB2701", 26170),
            ("加空", "中信期货", "MA2701", 18384),
            ("加多", "中信期货", "MA2701", 16936),
            ("加多", "中信期货", "M2701", 15389),
            ("加多", "东证期货", "RM2701", 13638),
            ("加多", "东证期货", "FG2701", 12820),
            ("加多", "东证期货", "SA2701", 12594),
            ("加空", "国泰君安", "MA2701", 10897),
            ("加多", "东证期货", "M2701", 10745),
            ("加空", "东证期货", "MA2701", 10365),
        ],
        "cut": [
            ("减多", "中信期货", "C2611", -20714),
            ("减空", "国泰君安", "C2611", -15893),
            ("减空", "东证期货", "FG2701", -14344),
            ("减多", "银河期货", "FG2701", -13916),
            ("减空", "东证期货", "SA2701", -13307),
            ("减空", "国泰君安", "HC2701", -11152),
            ("减多", "东证期货", "TA2701", -11005),
            ("减空", "银河期货", "SA2701", -8388),
            ("减空", "东证期货", "M2701", -8007),
            ("减空", "中信期货", "SR2701", -7830),
        ],
    },
    {
        "name": "三大外资",
        "date": "2026-10-09",
        "group": FOREIGN,
        "add": [
            ("加多", "摩根大通", "M2701", 10392),
            ("加多", "摩根大通", "Y2701", 7291),
            ("加空", "高盛期货", "SM2701", 3875),
            ("加空", "高盛期货", "HC2701", 3865),
            ("加空", "高盛期货", "SP2701", 3463),
            ("加多", "摩根大通", "RM2701", 3015),
            ("加空", "高盛期货", "RB2701", 2844),
            ("加空", "摩根大通", "P2701", 2803),
            ("加多", "高盛期货", "OI2701", 2625),
            ("加空", "高盛期货", "FG2701", 1695),
        ],
        "cut": [
            ("减多", "高盛期货", "RB2701", -34661),
            ("减空", "高盛期货", "SR2701", -17515),
            ("减空", "高盛期货", "V2701", -13696),
            ("减多", "高盛期货", "EB2611", -10130),
            ("减多", "摩根大通", "TA2701", -8294),
            ("减多", "高盛期货", "M2701", -8028),
            ("减空", "高盛期货", "RM2701", -7707),
            ("减多", "高盛期货", "Y2701", -6886),
            ("减多", "高盛期货", "MA2701", -4962),
            ("减多", "摩根大通", "P2701", -4243),
        ],
    },
]


def session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Referer": "https://data.eastmoney.com/",
        }
    )
    return s


def em_get(s: requests.Session, params: dict, tries: int = 4) -> dict:
    last = None
    for i in range(tries):
        try:
            r = s.get(EASTMONEY, params=params, timeout=40)
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(0.6 * (i + 1))
    raise RuntimeError(f"eastmoney failed: {last}")


def fetch_org_day(org: str, day: str) -> list:
    """One calendar day fits in a single page. Month-level paging drops rows."""
    s = session()
    rows = []
    page = 1
    pages = 1
    while page <= pages:
        params = {
            "reportName": "RPT_FUTU_DAILYPOSITION",
            "columns": (
                "SECUCODE,SECURITY_CODE,TRADE_DATE,ORG_NAME_ABBR_NEW,"
                "MEMBER_NAME_ABBR,TRADE_CODE,LPRANK,SPRANK,LP_CHANGE,SP_CHANGE,"
                "LONG_POSITION,SHORT_POSITION,SETTLE_PRICE,VOLUME,VOLUME_CHANGE"
            ),
            "filter": (
                f'(ORG_NAME_ABBR_NEW="{org}")(TRADE_DATE=\'{day}\')(TYPE="0")'
            ),
            "pageNumber": page,
            "pageSize": 500,
            "sortColumns": "SECURITY_CODE",
            "sortTypes": "1",
            "source": "WEB",
            "client": "WEB",
        }
        payload = em_get(s, params)
        result = payload.get("result") or {}
        data = result.get("data") or []
        pages = int(result.get("pages") or 1)
        rows.extend(data)
        if not data:
            break
        page += 1
        if page > 5:
            break
    return rows


def weekdays(start="2025-11-20", end="2026-10-09"):
    cur = pd.Timestamp(start)
    stop = pd.Timestamp(end)
    while cur <= stop:
        if cur.weekday() < 5:
            yield cur.strftime("%Y-%m-%d")
        cur += pd.Timedelta(days=1)


def download_seats() -> pd.DataFrame:
    cache = DATA / "seat_rows.csv.gz"
    if cache.exists():
        print(f"load seat cache {cache}")
        return pd.read_csv(cache)
    DATA.mkdir(parents=True, exist_ok=True)
    jobs = [(org, day) for org in ORGS for day in weekdays()]
    print(f"download {len(jobs)} org-days")
    frames = []
    done = 0
    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = {pool.submit(fetch_org_day, org, day): (org, day) for org, day in jobs}
        for fut in as_completed(futs):
            org, day = futs[fut]
            done += 1
            try:
                rows = fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"FAIL {org} {day}: {exc}")
                continue
            if rows:
                frames.append(pd.DataFrame(rows))
            if done % 200 == 0 or done == len(jobs):
                print(f"  seats {done}/{len(jobs)}")
    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(cache, index=False, compression="gzip")
    print(f"saved {cache} rows={len(raw)}")
    return raw


def clean_seats(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["date"] = pd.to_datetime(df["TRADE_DATE"]).dt.strftime("%Y-%m-%d")
    df["code"] = df["SECURITY_CODE"].astype(str).str.upper()
    df["org"] = df["ORG_NAME_ABBR_NEW"].astype(str)
    df["member"] = df["MEMBER_NAME_ABBR"].fillna("").astype(str)
    df["variety"] = df["TRADE_CODE"].astype(str).str.upper()
    for col in [
        "LPRANK",
        "SPRANK",
        "LP_CHANGE",
        "SP_CHANGE",
        "LONG_POSITION",
        "SHORT_POSITION",
        "SETTLE_PRICE",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # 9999 means the seat is not in that side's top 20, so the change is not disclosed.
    df.loc[df["LPRANK"] >= 9999, ["LP_CHANGE", "LONG_POSITION"]] = np.nan
    df.loc[df["SPRANK"] >= 9999, ["SP_CHANGE", "SHORT_POSITION"]] = np.nan
    df = df.dropna(subset=["LP_CHANGE", "SP_CHANGE"], how="all")
    df["prefer"] = df["member"].str.contains("代客").astype(int)
    df["not_prop"] = (~df["member"].str.contains("自营")).astype(int)
    df["gross"] = df[["LONG_POSITION", "SHORT_POSITION"]].abs().sum(axis=1)
    df = df.sort_values(
        ["prefer", "not_prop", "gross"], ascending=[False, False, False]
    )
    before = len(df)
    df = df.drop_duplicates(["date", "code", "org"], keep="first")
    df["is_prop"] = df["member"].str.contains("自营")
    print(
        f"clean seats {before} -> {len(df)} unique org-contract-days, "
        f"prop rows kept={int(df['is_prop'].sum())}"
    )
    keep = [
        "date",
        "code",
        "org",
        "member",
        "variety",
        "LP_CHANGE",
        "SP_CHANGE",
        "LONG_POSITION",
        "SHORT_POSITION",
        "SETTLE_PRICE",
        "is_prop",
    ]
    return df[keep].reset_index(drop=True)


def sina_candidates(code: str) -> list:
    raw = str(code).strip()
    out = []
    for item in (raw, raw.upper(), raw.lower()):
        if item and item not in out:
            out.append(item)
    return out


def fetch_one_price(code: str) -> tuple:
    dest = PRICE_DIR / f"{code.upper()}.csv"
    if dest.exists() and dest.stat().st_size > 0:
        return code, "cache", 0
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://finance.sina.com.cn/futures/",
        }
    )
    last_err = "empty"
    for sym in sina_candidates(code):
        url = SINA_KLINE.format(sym=urllib.parse.quote(sym))
        try:
            r = s.get(url, timeout=30)
            r.raise_for_status()
            text = r.text
            match = re.search(r"(\[.*\])", text, re.S)
            if not match:
                last_err = "no-json"
                continue
            payload = json.loads(match.group(1))
            if not payload:
                last_err = "empty"
                continue
            frame = pd.DataFrame(payload)
            frame = frame.rename(
                columns={
                    "d": "date",
                    "o": "open",
                    "h": "high",
                    "l": "low",
                    "c": "close",
                    "v": "volume",
                    "p": "oi",
                    "s": "settle",
                }
            )
            for col in ["open", "high", "low", "close", "volume", "oi", "settle"]:
                if col not in frame.columns:
                    frame[col] = np.nan
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
            frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
            frame = frame.dropna(subset=["date", "open", "close"])
            frame = frame[frame["open"] > 0]
            if len(frame) < 3:
                last_err = "short"
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(dest, index=False)
            return code, "ok", len(frame)
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)[:80]
            time.sleep(0.2)
    return code, last_err, 0


def download_prices(codes: list) -> None:
    PRICE_DIR.mkdir(parents=True, exist_ok=True)
    pending = [c for c in sorted(set(codes)) if not (PRICE_DIR / f"{c.upper()}.csv").exists()]
    print(f"prices cached={len(set(codes))-len(pending)} to_fetch={len(pending)}")
    ok = 0
    fail = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = [pool.submit(fetch_one_price, code) for code in pending]
        for i, fut in enumerate(as_completed(futs), 1):
            code, status, n = fut.result()
            if status in ("ok", "cache"):
                ok += 1
            else:
                fail += 1
                if fail <= 15:
                    print(f"  price miss {code}: {status}")
            if i % 50 == 0 or i == len(futs):
                print(f"  prices {i}/{len(futs)} ok={ok} fail={fail}")


def load_prices(codes: list) -> pd.DataFrame:
    frames = []
    for code in sorted(set(codes)):
        path = PRICE_DIR / f"{code.upper()}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame["code"] = code.upper()
        frames.append(frame)
    px = pd.concat(frames, ignore_index=True)
    px["variety"] = px["code"].str.extract(r"([A-Za-z]+)", expand=False).str.upper()
    px = px.dropna(subset=["variety", "date"])
    px = px.sort_values(["code", "date"])
    grouped = px.groupby("code", sort=False)
    px["next_open"] = grouped["open"].shift(-1)
    px["next_close"] = grouped["close"].shift(-1)
    px["close_5"] = grouped["close"].shift(-5)
    px["ret_1"] = px["next_close"] / px["next_open"] - 1.0
    px["ret_5"] = px["close_5"] / px["next_open"] - 1.0
    px.loc[~np.isfinite(px["ret_1"]), "ret_1"] = np.nan
    px.loc[~np.isfinite(px["ret_5"]), "ret_5"] = np.nan
    return px


def main_contract_table(px: pd.DataFrame) -> pd.DataFrame:
    live = px.dropna(subset=["oi"]).copy()
    live = live[live["oi"] > 0]
    live = live.sort_values(["date", "variety", "oi", "volume"], ascending=[True, True, False, False])
    main = live.drop_duplicates(["date", "variety"], keep="first")
    return main[
        ["date", "variety", "code", "oi", "volume", "open", "close", "settle", "ret_1", "ret_5"]
    ].rename(columns={"code": "main_code", "settle": "main_settle"})


def events_from_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per disclosed side change. This is the picture's ranking unit."""
    parts = []
    long_rows = rows.dropna(subset=["LP_CHANGE"]).copy()
    long_rows = long_rows[long_rows["LP_CHANGE"] != 0]
    long_rows["side"] = np.where(long_rows["LP_CHANGE"] > 0, "加多", "减多")
    long_rows["change"] = long_rows["LP_CHANGE"]
    short_rows = rows.dropna(subset=["SP_CHANGE"]).copy()
    short_rows = short_rows[short_rows["SP_CHANGE"] != 0]
    short_rows["side"] = np.where(short_rows["SP_CHANGE"] > 0, "加空", "减空")
    short_rows["change"] = short_rows["SP_CHANGE"]
    parts = [long_rows, short_rows]
    ev = pd.concat(parts, ignore_index=True)
    return ev[["date", "org", "code", "variety", "side", "change", "member"]]


def top10(ev: pd.DataFrame, group: list, date: str, kind: str) -> pd.DataFrame:
    sub = ev[(ev["date"] == date) & (ev["org"].isin(group))].copy()
    if kind == "add":
        sub = sub[sub["side"].isin(["加多", "加空"])]
        sub = sub.sort_values("change", ascending=False)
    else:
        sub = sub[sub["side"].isin(["减多", "减空"])]
        sub = sub.sort_values("change", ascending=True)
    return sub.head(10)


def value_hit(ev: pd.DataFrame, spec: tuple) -> bool:
    side, org, code, change = spec
    hit = ev[
        (ev["org"] == org)
        & (ev["code"] == code.upper())
        & (ev["side"] == side)
        & (np.isclose(ev["change"], change))
    ]
    return len(hit) > 0


def reconstruct(seats: pd.DataFrame, main: pd.DataFrame) -> dict:
    ev_all = events_from_rows(seats)
    report = []
    for board in BOARDS:
        date = board["date"]
        mains = set(main.loc[main["date"] == date, "main_code"])
        ev_main = ev_all[ev_all["code"].isin(mains)] if mains else ev_all.iloc[0:0]
        for kind, expected in (("add", board["add"]), ("cut", board["cut"])):
            got = top10(ev_main, board["group"], date, kind)
            got_keys = list(zip(got["side"], got["org"], got["code"], got["change"].round().astype(int)))
            exp_keys = [(a, b, c.upper(), int(d)) for a, b, c, d in expected]
            in_board = 0
            in_data = 0
            for spec in expected:
                if value_hit(ev_all[ev_all["date"] == date], spec):
                    in_data += 1
                key = (spec[0], spec[1], spec[2].upper(), int(spec[3]))
                if key in got_keys:
                    in_board += 1
            listed = ev_all[
                (ev_all["date"] == date) & (ev_all["org"].isin(board["group"]))
            ]
            listed_main = listed[listed["code"].isin(mains)] if mains else listed.iloc[0:0]
            report.append(
                {
                    "board": board["name"],
                    "date": date,
                    "kind": kind,
                    "exact_value_anywhere": in_data,
                    "exact_in_rebuilt_top10": in_board,
                    "expected": 10,
                    "main_contracts_that_day": int(len(mains)),
                    "side_rows_all_contracts": int(len(listed)),
                    "side_rows_on_oi_main": int(len(listed_main)),
                    "rebuilt_top10": [
                        {
                            "side": r.side,
                            "org": r.org,
                            "code": r.code,
                            "change": int(round(r.change)),
                        }
                        for r in got.itertuples(index=False)
                    ],
                }
            )
            print(
                f"board {board['name']} {date} {kind}: "
                f"values {in_data}/10, rebuilt top10 {in_board}/10, "
                f"mains={len(mains)}"
            )
    return {"boards": report}


def group_flow(seats: pd.DataFrame, orgs: list) -> pd.DataFrame:
    sub = seats[seats["org"].isin(orgs)].copy()
    sub["lp"] = sub["LP_CHANGE"]
    sub["sp"] = sub["SP_CHANGE"]
    # A seat's own net uses 0 for a side that was not disclosed, but only if the
    # other side was disclosed. If neither side is disclosed the row is absent.
    sub["seat_net"] = sub["lp"].fillna(0) - sub["sp"].fillna(0)
    sub["disclosed"] = sub["lp"].notna() | sub["sp"].notna()
    sub["pos"] = (sub["seat_net"] > 0).astype(int)
    sub["neg"] = (sub["seat_net"] < 0).astype(int)
    grouped = sub.groupby(["date", "code", "variety"], as_index=False).agg(
        net_lots=("seat_net", "sum"),
        n_seats=("disclosed", "sum"),
        n_pos=("pos", "sum"),
        n_neg=("neg", "sum"),
        long_chg=("lp", "sum"),
        short_chg=("sp", "sum"),
        long_pos=("LONG_POSITION", "sum"),
        short_pos=("SHORT_POSITION", "sum"),
    )
    grouped["net_pos"] = grouped["long_pos"] - grouped["short_pos"]
    return grouped


def build_panel(seats: pd.DataFrame, main: pd.DataFrame) -> pd.DataFrame:
    flows = {
        "retail": group_flow(seats, RETAIL),
        "inst": group_flow(seats, INST),
        "foreign": group_flow(seats, FOREIGN),
    }
    for org, key in [
        ("平安期货", "pingan"),
        ("东方财富期货", "dongcai"),
        ("徽商期货", "huishang"),
    ]:
        flows[key] = group_flow(seats, [org])
    panel = main.copy()
    for name, flow in flows.items():
        keep = flow.rename(
            columns={
                "code": "main_code",
                "net_lots": f"{name}_net",
                "n_seats": f"{name}_n",
                "n_pos": f"{name}_pos",
                "n_neg": f"{name}_neg",
                "net_pos": f"{name}_level_lots",
            }
        )
        panel = panel.merge(
            keep[
                [
                    "date",
                    "main_code",
                    f"{name}_net",
                    f"{name}_n",
                    f"{name}_pos",
                    f"{name}_neg",
                    f"{name}_level_lots",
                ]
            ],
            on=["date", "main_code"],
            how="left",
        )
    for name in flows:
        panel[f"{name}_oi"] = panel[f"{name}_net"] / panel["oi"]
        # Position level, not the daily change. This is the quantity the
        # seat-P&L articles discuss. It is a secondary check, not the rule
        # that reproduces the top-10 board.
        panel[f"{name}_level"] = panel[f"{name}_level_lots"] / panel["oi"]
    panel["is_financial"] = panel["variety"].isin(FINANCIAL)
    panel = panel[panel["oi"] >= 5000].copy()
    # Z-score against the previous 60 days on which this variety had a disclosed
    # retail flow. Missing days are not treated as zero flow.
    panel = panel.sort_values(["variety", "date"])
    panel["retail_z"] = np.nan
    for _, idx in panel.groupby("variety").groups.items():
        observed = panel.loc[idx, "retail_oi"].dropna()
        if len(observed) < 41:
            continue
        mean = observed.shift(1).rolling(60, min_periods=40).mean()
        std = observed.shift(1).rolling(60, min_periods=40).std()
        z = (observed - mean) / std
        panel.loc[z.index, "retail_z"] = z.to_numpy()
    panel.loc[~np.isfinite(panel["retail_z"]), "retail_z"] = np.nan
    return panel


def tstat(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    sd = x.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return float("nan")
    return float(x.mean() / sd * math.sqrt(n))


def summarize_returns(rets: list, period: str) -> dict:
    arr = np.asarray(rets, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(len(arr))
    if n == 0:
        return {"n": 0}
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if n > 1 else float("nan")
    ann = 252 if period == "1d" else 252 / 5
    sharpe = float(mean / sd * math.sqrt(ann)) if sd and np.isfinite(sd) and sd > 0 else float("nan")
    equity = float(np.prod(1.0 + arr) - 1.0)
    return {
        "n": n,
        "mean": mean,
        "t": tstat(arr),
        "hit": float((arr > 0).mean()),
        "sharpe": sharpe,
        "cum": equity,
    }


def daily_books(panel: pd.DataFrame, score: str, ret_col: str, mode: str, mask: pd.Series):
    """Equal-weight quintile book. Fade shorts the high score and longs the low score."""
    rows = []
    use = panel.loc[mask].dropna(subset=[score, ret_col])
    for date, day in use.groupby("date"):
        if len(day) < 15:
            continue
        day = day.copy()
        day["pct"] = day[score].rank(pct=True, method="average")
        low = day[day["pct"] <= 0.2]
        high = day[day["pct"] >= 0.8]
        if len(low) < 3 or len(high) < 3:
            continue
        if mode == "fade":
            port = float(low[ret_col].mean() - high[ret_col].mean())
            long_names = set(low["variety"])
            short_names = set(high["variety"])
        else:
            port = float(high[ret_col].mean() - low[ret_col].mean())
            long_names = set(high["variety"])
            short_names = set(low["variety"])
        rows.append(
            {
                "date": date,
                "ret": port,
                "n_long": len(long_names),
                "n_short": len(short_names),
                "long": long_names,
                "short": short_names,
            }
        )
    return rows


def apply_turnover_cost(books: list, bps_per_side: float = 1.0) -> list:
    """Charge bps_per_side when a name enters or leaves. 1 bp per side is 2 bp round trip."""
    prev_long, prev_short = set(), set()
    out = []
    unit = bps_per_side / 10000.0
    for row in books:
        n_pos = row["n_long"] + row["n_short"]
        entries = len(row["long"] - prev_long) + len(row["short"] - prev_short)
        exits = len(prev_long - row["long"]) + len(prev_short - row["short"])
        # The first day has no exit. Charge entries only.
        if not prev_long and not prev_short:
            cost = entries * unit / max(n_pos, 1)
        else:
            cost = (entries + exits) * unit / max(n_pos, 1)
        out.append({**row, "ret_net": row["ret"] - cost, "cost": cost})
        prev_long, prev_short = row["long"], row["short"]
    return out


def ic_series(panel: pd.DataFrame, score: str, ret_col: str, mask: pd.Series) -> list:
    use = panel.loc[mask].dropna(subset=[score, ret_col])
    out = []
    for date, day in use.groupby("date"):
        if len(day) < 10:
            continue
        if day[score].nunique() < 3 or day[ret_col].nunique() < 3:
            continue
        # Pearson correlation of ranks. Avoids a scipy dependency.
        ic = day[score].rank().corr(day[ret_col].rank())
        if pd.notna(ic):
            out.append((date, float(ic)))
    return out


def eval_strategy(panel, score, mode, mask, ret_col, period, calendar_dates) -> dict:
    if period == "1d":
        books = daily_books(panel, score, ret_col, mode, mask)
        books = apply_turnover_cost(books)
        gross = summarize_returns([r["ret"] for r in books], "1d")
        net = summarize_returns([r["ret_net"] for r in books], "1d")
        costs = [r["cost"] for r in books]
        return {
            "gross": gross,
            "net": net,
            "avg_cost": float(np.mean(costs)) if costs else None,
            "books": [{"date": r["date"], "ret": r["ret"], "ret_net": r["ret_net"]} for r in books],
        }
    # Five non-overlapping phases. Report the median so a lucky start date is not selected.
    phases = []
    for offset in range(5):
        chosen = set(calendar_dates[offset::5])
        phase_mask = mask & panel["date"].isin(chosen)
        books = daily_books(panel, score, ret_col, mode, phase_mask)
        # Holding 5 sessions, so the next rebalance replaces the book. Charge the
        # round trip once per rebalance using the same entry/exit accounting.
        books = apply_turnover_cost(books)
        phases.append(
            {
                "offset": offset,
                "gross": summarize_returns([r["ret"] for r in books], "5d"),
                "net": summarize_returns([r["ret_net"] for r in books], "5d"),
            }
        )
    def median_field(path):
        vals = []
        for phase in phases:
            cur = phase
            for key in path:
                cur = cur.get(key) if isinstance(cur, dict) else None
            if cur is not None and np.isfinite(cur):
                vals.append(cur)
        return float(np.median(vals)) if vals else None

    return {
        "phases": [{k: v for k, v in p.items() if k != "books"} for p in phases],
        "median_gross_mean": median_field(("gross", "mean")),
        "median_gross_t": median_field(("gross", "t")),
        "median_gross_sharpe": median_field(("gross", "sharpe")),
        "median_net_sharpe": median_field(("net", "sharpe")),
        "median_hit": median_field(("gross", "hit")),
        "median_n": median_field(("gross", "n")),
    }


def deciles(panel, score, ret_col, mask) -> list:
    use = panel.loc[mask].dropna(subset=[score, ret_col]).copy()
    rows = []
    for date, day in use.groupby("date"):
        if len(day) < 20:
            continue
        day = day.copy()
        day["dec"] = pd.qcut(day[score].rank(method="first"), 10, labels=False, duplicates="drop")
        for dec, part in day.groupby("dec"):
            rows.append({"date": date, "dec": int(dec), "ret": float(part[ret_col].mean())})
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    out = (
        frame.groupby("dec")["ret"]
        .agg(["mean", "count"])
        .reset_index()
        .to_dict(orient="records")
    )
    return out


def run_backtest(panel: pd.DataFrame) -> dict:
    calendar_dates = sorted(panel["date"].unique())
    commodity = ~panel["is_financial"]
    # The vendor feed is sparse before mid-2026: a major broker has only a
    # handful of stored contracts. Require the institution group to be disclosed
    # on at least 25 commodity main contracts that day, otherwise the
    # cross-section is the database, not the market.
    inst_breadth = (
        panel.loc[commodity & panel["inst_oi"].notna()]
        .groupby("date")["variety"]
        .nunique()
    )
    eligible_dates = set(inst_breadth[inst_breadth >= 25].index)
    commodity = commodity & panel["date"].isin(eligible_dates)
    specs = [
        ("retail_fade_oi", "retail_oi", "fade", commodity),
        ("retail_follow_oi", "retail_oi", "follow", commodity),
        ("inst_fade_oi", "inst_oi", "fade", commodity),
        ("inst_follow_oi", "inst_oi", "follow", commodity),
        ("foreign_fade_oi", "foreign_oi", "fade", commodity),
        ("foreign_follow_oi", "foreign_oi", "follow", commodity),
        ("retail_fade_raw_lots", "retail_net", "fade", commodity),
        ("inst_follow_raw_lots", "inst_net", "follow", commodity),
        (
            "retail_fade_consensus",
            "retail_oi",
            "fade",
            commodity & ((panel["retail_pos"] >= 2) | (panel["retail_neg"] >= 2)),
        ),
        (
            "retail_fade_when_opposite_inst",
            "retail_oi",
            "fade",
            commodity
            & panel["inst_oi"].notna()
            & (np.sign(panel["retail_oi"]) * np.sign(panel["inst_oi"]) < 0),
        ),
        (
            "retail_fade_oi_all_assets",
            "retail_oi",
            "fade",
            panel["date"].isin(eligible_dates) & panel["retail_oi"].notna(),
        ),
        ("retail_fade_level", "retail_level", "fade", commodity),
        ("inst_follow_level", "inst_level", "follow", commodity),
        ("inst_fade_level", "inst_level", "fade", commodity),
    ]
    strategies = {}
    primary_books = None
    for name, score, mode, mask in specs:
        print(f"strategy {name}")
        one = eval_strategy(panel, score, mode, mask, "ret_1", "1d", calendar_dates)
        five = eval_strategy(panel, score, mode, mask, "ret_5", "5d", calendar_dates)
        strategies[name] = {
            "one_day": {k: v for k, v in one.items() if k != "books"},
            "five_day": five,
        }
        if name == "retail_fade_oi":
            primary_books = one["books"]
            # Year split on the primary gross daily book.
            by_year = {}
            frame = pd.DataFrame(one["books"])
            if len(frame):
                frame["year"] = frame["date"].str.slice(0, 4)
                for year, part in frame.groupby("year"):
                    by_year[year] = summarize_returns(part["ret"].tolist(), "1d")
            strategies[name]["by_year"] = by_year

    ics = {}
    for name, score in [
        ("retail", "retail_oi"),
        ("inst", "inst_oi"),
        ("foreign", "foreign_oi"),
        ("pingan", "pingan_oi"),
        ("dongcai", "dongcai_oi"),
        ("huishang", "huishang_oi"),
        ("retail_raw_lots", "retail_net"),
        ("retail_level", "retail_level"),
        ("inst_level", "inst_level"),
    ]:
        series = ic_series(panel, score, "ret_1", commodity)
        vals = [v for _, v in series]
        ics[name] = {
            "n_days": len(vals),
            "mean_ic": float(np.mean(vals)) if vals else None,
            "t": tstat(np.asarray(vals)) if vals else None,
            "hit_negative": float(np.mean(np.asarray(vals) < 0)) if vals else None,
        }
        print(f"ic {name}: {ics[name]}")

    both = panel.loc[commodity].dropna(subset=["retail_oi", "inst_oi"])
    corr = float(both["retail_oi"].corr(both["inst_oi"])) if len(both) > 10 else None
    decile = {
        "retail_fade_shape_low_score_is_decile_0": deciles(panel, "retail_oi", "ret_1", commodity),
        "inst_shape": deciles(panel, "inst_oi", "ret_1", commodity),
    }
    # Time-series rule: fade retail when |z| > 1. Each day is the equal-weight
    # average of varieties that fire, so days are the sample, not contract-days.
    ts_rows = []
    ts = panel.loc[commodity].dropna(subset=["retail_z", "ret_1"])
    ts = ts[ts["retail_z"].abs() > 1]
    for date, day in ts.groupby("date"):
        # Fade: position = -sign(z)
        contrib = -np.sign(day["retail_z"]) * day["ret_1"]
        ts_rows.append(float(contrib.mean()))
    ts_summary = summarize_returns(ts_rows, "1d")
    coverage = {
        "variety_days": int(len(panel.loc[commodity])),
        "retail_observed_share": float(panel.loc[commodity, "retail_oi"].notna().mean()),
        "inst_observed_share": float(panel.loc[commodity, "inst_oi"].notna().mean()),
        "foreign_observed_share": float(panel.loc[commodity, "foreign_oi"].notna().mean()),
        "retail_inst_corr": corr,
        "date_min": str(panel["date"].min()),
        "date_max": str(panel["date"].max()),
        "n_dates": int(panel["date"].nunique()),
        "eligible_dates": int(len(eligible_dates)),
        "eligible_min": min(eligible_dates) if eligible_dates else None,
        "eligible_max": max(eligible_dates) if eligible_dates else None,
        "n_varieties": int(panel.loc[commodity, "variety"].nunique()) if commodity.any() else 0,
    }
    return {
        "coverage": coverage,
        "ic": ics,
        "strategies": strategies,
        "deciles": decile,
        "retail_timeseries_z": ts_summary,
        "primary_books": primary_books or [],
    }


def fmt_pct(x):
    if x is None or not isinstance(x, (int, float)) or not np.isfinite(x):
        return ""
    return f"{100 * x:.3f}%"


def fmt_num(x, digits=2):
    if x is None or not isinstance(x, (int, float)) or not np.isfinite(x):
        return ""
    return f"{x:.{digits}f}"


def write_tables(board_report: dict, study: dict) -> None:
    RESULT.mkdir(parents=True, exist_ok=True)
    lines = ["# Board reconstruction", ""]
    for item in board_report["boards"]:
        lines.append(
            f"- {item['board']} {item['date']} {item['kind']}: "
            f"screenshot lots found in the feed {item['exact_value_anywhere']}/10, "
            f"inside rebuilt OI-main top10 {item['exact_in_rebuilt_top10']}/10, "
            f"OI mains that day {item['main_contracts_that_day']}, "
            f"disclosed side-rows on those mains {item['side_rows_on_oi_main']}"
        )
        for row in item["rebuilt_top10"]:
            lines.append(
                f"    {row['side']} {row['org']} {row['code']} {row['change']:+d}"
            )
    lines += ["", "# Strategies (commodity varieties, gross then net of 1bp per side)", ""]
    lines.append("| strategy | 1d mean | 1d t | 1d hit | 1d sharpe | 1d net sharpe | 5d median sharpe | 5d median t |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, payload in study["strategies"].items():
        one = payload["one_day"]["gross"]
        net = payload["one_day"]["net"]
        five = payload["five_day"]
        lines.append(
            "| {name} | {mean} | {t} | {hit} | {sh} | {nsh} | {fsh} | {ft} |".format(
                name=name,
                mean=fmt_pct(one.get("mean")),
                t=fmt_num(one.get("t")),
                hit=fmt_num(one.get("hit")),
                sh=fmt_num(one.get("sharpe")),
                nsh=fmt_num(net.get("sharpe")),
                fsh=fmt_num(five.get("median_gross_sharpe")),
                ft=fmt_num(five.get("median_gross_t")),
            )
        )
    lines += ["", "# Cross-sectional Spearman IC vs next open-to-close", ""]
    for name, payload in study["ic"].items():
        lines.append(
            f"- {name}: mean IC {fmt_num(payload['mean_ic'], 4)} "
            f"t {fmt_num(payload['t'])} days {payload['n_days']} "
            f"share of days IC<0 {fmt_num(payload['hit_negative'])}"
        )
    lines += ["", "# Retail net/OI deciles (0 = most net short add)", ""]
    for row in study["deciles"]["retail_fade_shape_low_score_is_decile_0"]:
        lines.append(
            f"- decile {row['dec']}: mean next open-to-close {fmt_pct(row['mean'])} (days {int(row['count'])})"
        )
    (RESULT / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    RESULT.mkdir(parents=True, exist_ok=True)
    raw = download_seats()
    seats = clean_seats(raw)
    print("download prices")
    download_prices(seats["code"].unique().tolist())
    px = load_prices(seats["code"].unique().tolist())
    print(f"price rows {len(px)} codes {px['code'].nunique()}")
    main_px = main_contract_table(px)
    print(f"main contract rows {len(main_px)}")
    board_report = reconstruct(seats, main_px)
    panel = build_panel(seats, main_px)
    panel_path = DATA / "signals_daily.csv.gz"
    panel.to_csv(panel_path, index=False, compression="gzip")
    print(f"panel {panel_path} {len(panel)}")
    study = run_backtest(panel)
    books = study.pop("primary_books")
    pd.DataFrame(books).to_csv(RESULT / "retail_fade_daily.csv", index=False)
    payload = {"boards": board_report, "study": study}
    (RESULT / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_tables(board_report, study)
    print("wrote", RESULT / "summary.json")


if __name__ == "__main__":
    main()
