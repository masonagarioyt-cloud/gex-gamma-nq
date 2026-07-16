"""
Multi-Asset Gamma Exposure (GEX) Level Generator — "May Gamma & Gex Levels"
-----------------------------------------------------------------------------
Free-data estimate of gamma exposure levels for Nasdaq (NQ) and S&P (ES)
futures, derived from QQQ and SPY options open interest respectively (the
closest free, liquid proxies for NDX/NQ and SPX/ES). Outputs a complete,
ready-to-paste TradingView Pine Script with the levels hardcoded and a
full settings panel (per-level show/hide, colors, text size/position,
merge distance, per-source visibility).

IMPORTANT / HONEST LIMITATIONS:
- Data source is Yahoo Finance's free, unofficial feed (via yfinance).
  It is NOT a licensed real-time feed. Expect occasional delays,
  missing data, or breakage if Yahoo changes something.
- GEX is computed using the standard public convention (dealers assumed
  long calls / short puts). This is the same simplifying assumption used
  by nearly every free GEX calculator. It is NOT SpotGamma's or
  MenthorQ's proprietary model and will not exactly match their numbers.
- QQQ/SPY options are used as proxies for NQ/ES, scaled via the live
  futures/ETF price ratio at run time.
- "Strength %" on ranked GEX levels is this strike's gamma exposure
  magnitude relative to the single largest strike found in that chain.
  It is NOT a probability or a confidence score.
- This script intentionally does NOT attempt to replicate "HVL" (that's
  volume-profile data, a different data source) or "Blind Spots" (an
  undocumented MenthorQ-proprietary concept with no public formula).
"""

import sys
import math
import datetime as dt

import numpy as np
import yfinance as yf
from scipy.stats import norm

RISK_FREE_RATE = 0.05
CONTRACT_MULTIPLIER = 100
TOP_N_LEVELS = 8
TOP_N_LEVELS_0DTE = 5
GEX_LEVEL_COLOR = "#ace5dc"

SOURCES = [
    {"underlying": "QQQ", "options_ticker": "QQQ"},
    {"underlying": "SPY", "options_ticker": "SPY"},
]

DISPLAY_TARGETS = [
    {"key": "QQQ", "underlying": "QQQ", "price_ticker": None, "native": True},
    {"key": "NQ", "underlying": "QQQ", "price_ticker": "NQ=F", "native": False},
    {"key": "SPY", "underlying": "SPY", "price_ticker": None, "native": True},
    {"key": "ES", "underlying": "SPY", "price_ticker": "ES=F", "native": False},
]


def bs_gamma(spot, strike, t_years, iv, r=RISK_FREE_RATE):
    if t_years <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
        return 0.0
    d1 = (math.log(spot / strike) + (r + 0.5 * iv ** 2) * t_years) / (iv * math.sqrt(t_years))
    return norm.pdf(d1) / (spot * iv * math.sqrt(t_years))


def pick_expiration(expirations, today):
    """Nearest expiration STRICTLY AFTER today (skips 0DTE noise for
    the 'main' calculation; 0DTE gets its own separate pass)."""
    dated = sorted(expirations)
    for e in dated:
        exp_date = dt.datetime.strptime(e, "%Y-%m-%d").date()
        if exp_date > today:
            return e
    return dated[-1] if dated else None


def clamp_iv(iv, low=0.05, high=3.0):
    if iv is None or iv <= 0 or math.isnan(iv):
        return 0.20
    return max(low, min(high, iv))


def compute_expiry_data(ticker_obj, expiry, spot, today):
    exp_date = dt.datetime.strptime(expiry, "%Y-%m-%d").date()
    t_years = max((exp_date - today).days, 0.5) / 365.0

    chain = ticker_obj.option_chain(expiry)
    calls, puts = chain.calls, chain.puts
    strikes = sorted(set(calls["strike"]).union(set(puts["strike"])))

    gex_by_strike, call_oi_by_strike, put_oi_by_strike, total_gamma_by_strike = {}, {}, {}, {}
    atm_iv, atm_diff = None, None

    for k in strikes:
        c_row = calls[calls["strike"] == k]
        p_row = puts[puts["strike"] == k]

        c_oi = float(c_row["openInterest"].iloc[0]) if not c_row.empty and not np.isnan(c_row["openInterest"].iloc[0]) else 0.0
        p_oi = float(p_row["openInterest"].iloc[0]) if not p_row.empty and not np.isnan(p_row["openInterest"].iloc[0]) else 0.0
        c_iv = float(c_row["impliedVolatility"].iloc[0]) if not c_row.empty else 0.0
        p_iv = float(p_row["impliedVolatility"].iloc[0]) if not p_row.empty else 0.0

        c_gamma = bs_gamma(spot, k, t_years, c_iv)
        p_gamma = bs_gamma(spot, k, t_years, p_iv)

        gex_by_strike[k] = (c_oi * c_gamma - p_oi * p_gamma) * CONTRACT_MULTIPLIER * spot ** 2 * 0.01
        total_gamma_by_strike[k] = (c_oi * c_gamma + p_oi * p_gamma) * CONTRACT_MULTIPLIER * spot ** 2 * 0.01
        call_oi_by_strike[k] = c_oi
        put_oi_by_strike[k] = p_oi

        diff = abs(k - spot)
        if atm_diff is None or diff < atm_diff:
            atm_diff = diff
            atm_iv = c_iv if c_iv > 0 else p_iv

    return {
        "gex": gex_by_strike, "call_oi": call_oi_by_strike, "put_oi": put_oi_by_strike,
        "total_gamma": total_gamma_by_strike, "t_years": t_years, "atm_iv": atm_iv,
    }


def wall_levels(gex_by_strike):
    call_wall = max(gex_by_strike, key=lambda k: gex_by_strike[k])
    put_wall = min(gex_by_strike, key=lambda k: gex_by_strike[k])
    return call_wall, put_wall


def gamma_flip_level(gex_by_strike, spot):
    strikes_sorted = sorted(gex_by_strike.keys())
    gex_values = [gex_by_strike[k] for k in strikes_sorted]
    cumulative = np.cumsum(gex_values)
    for i in range(1, len(cumulative)):
        if cumulative[i - 1] < 0 <= cumulative[i]:
            k0, k1 = strikes_sorted[i - 1], strikes_sorted[i]
            c0, c1 = cumulative[i - 1], cumulative[i]
            frac = -c0 / (c1 - c0) if (c1 - c0) != 0 else 0
            return k0 + frac * (k1 - k0)
    return spot


def top_n_ranked(gex_by_strike, exclude_strikes, n):
    max_abs = max((abs(v) for v in gex_by_strike.values()), default=1.0)
    ranked = sorted(
        ((k, v) for k, v in gex_by_strike.items() if k not in exclude_strikes),
        key=lambda kv: abs(kv[1]), reverse=True,
    )[:n]
    out = []
    for k, v in ranked:
        pct = round(100 * abs(v) / max_abs) if max_abs else 0
        out.append((k, pct))
    return out


def oi_and_gamma_wall(data):
    call_res_oi = max(data["call_oi"], key=lambda k: data["call_oi"][k]) if data["call_oi"] else None
    put_sup_oi = max(data["put_oi"], key=lambda k: data["put_oi"][k]) if data["put_oi"] else None
    gamma_wall = max(data["total_gamma"], key=lambda k: data["total_gamma"][k]) if data["total_gamma"] else None
    return call_res_oi, put_sup_oi, gamma_wall


def to_futures(price_etf, etf_spot, futures_spot):
    return price_etf * (futures_spot / etf_spot)


def fetch_last_price(ticker_symbol):
    hist = yf.Ticker(ticker_symbol).history(period="1d")
    if hist.empty:
        raise RuntimeError(f"Could not fetch price for {ticker_symbol}.")
    return float(hist["Close"].iloc[-1])


def compute_underlying(options_ticker, today):
    etf_spot = fetch_last_price(options_ticker)
    ticker_obj = yf.Ticker(options_ticker)
    expirations = ticker_obj.options
    if not expirations:
        raise RuntimeError(f"No option expirations returned for {options_ticker}.")

    main_expiry = pick_expiration(expirations, today)
    main = compute_expiry_data(ticker_obj, main_expiry, etf_spot, today)
    gex_main = main["gex"]

    call_wall, put_wall = wall_levels(gex_main)
    gamma_flip = gamma_flip_level(gex_main, etf_spot)
    ranked = top_n_ranked(gex_main, exclude_strikes={call_wall, put_wall}, n=TOP_N_LEVELS)
    call_res_oi, put_sup_oi, gamma_wall = oi_and_gamma_wall(main)

    today_str = today.strftime("%Y-%m-%d")
    zero_dte_expiry = today_str if today_str in expirations else None
    call_res_0dte = put_sup_0dte = None
    ranked_0dte = []
    call_res_oi_0dte = put_sup_oi_0dte = gamma_wall_0dte = None
    if zero_dte_expiry:
        z = compute_expiry_data(ticker_obj, zero_dte_expiry, etf_spot, today)
        gex_0dte = z["gex"]
        if gex_0dte:
            call_res_0dte, put_sup_0dte = wall_levels(gex_0dte)
            ranked_0dte = top_n_ranked(gex_0dte, exclude_strikes={call_res_0dte, put_sup_0dte}, n=TOP_N_LEVELS_0DTE)
            call_res_oi_0dte, put_sup_oi_0dte, gamma_wall_0dte = oi_and_gamma_wall(z)

    atm_iv = clamp_iv(main["atm_iv"])
    em = etf_spot * atm_iv * math.sqrt(1.0 / 365.0)
    em_low, em_high = etf_spot - em, etf_spot + em

    return {
        "etf_spot": etf_spot, "main_expiry": main_expiry, "zero_dte_expiry": zero_dte_expiry,
        "call_wall": call_wall, "put_wall": put_wall, "gamma_flip": gamma_flip,
        "gamma_wall": gamma_wall, "call_res_oi": call_res_oi, "put_sup_oi": put_sup_oi,
        "call_res_0dte": call_res_0dte, "put_sup_0dte": put_sup_0dte, "gamma_wall_0dte": gamma_wall_0dte,
        "call_res_oi_0dte": call_res_oi_0dte, "put_sup_oi_0dte": put_sup_oi_0dte,
        "ranked": ranked, "ranked_0dte": ranked_0dte, "em_low": em_low, "em_high": em_high,
    }


def build_levels_for_target(target, underlying_data):
    key = target["key"]
    u = underlying_data
    etf_spot = u["etf_spot"]

    if target["native"]:
        F = lambda p: p
        display_spot = etf_spot
    else:
        futures_spot = fetch_last_price(target["price_ticker"])
        F = lambda p: to_futures(p, etf_spot, futures_spot)
        display_spot = futures_spot

    lv = []

    def add(name, price, pct, color, group):
        if price is None:
            return
        lv.append({"name": f"{name} ({key})", "price": F(price), "pct": pct, "color": color, "group": group, "source": key})

    add("Call Wall", u["call_wall"], 100, "color.green", "Level Filters")
    add("Put Wall", u["put_wall"], 100, "color.red", "Level Filters")
    add("Gamma Flip", u["gamma_flip"], 100, "color.orange", "Level Filters")
    add("Gamma Wall", u["gamma_wall"], 100, "color.white", "Level Filters")
    add("Call Resistance (OI)", u["call_res_oi"], 90, "color.lime", "Level Filters")
    add("Put Support (OI)", u["put_sup_oi"], 90, "color.maroon", "Level Filters")
    add("Call Resistance 0DTE", u["call_res_0dte"], 95, "color.new(color.red, 20)", "Level Filters")
    add("Put Support 0DTE", u["put_sup_0dte"], 95, "color.new(color.teal, 20)", "Level Filters")
    add("Gamma Wall 0DTE", u["gamma_wall_0dte"], 90, "color.silver", "Level Filters")
    add("Call Resistance 0DTE (OI)", u["call_res_oi_0dte"], 85, "color.new(color.lime, 20)", "Level Filters")
    add("Put Support 0DTE (OI)", u["put_sup_oi_0dte"], 85, "color.new(color.maroon, 20)", "Level Filters")

    for i, (k, pct) in enumerate(u["ranked"], start=1):
        add(f"GEX {i}", k, pct, GEX_LEVEL_COLOR, "GEX Filters")
    for i, (k, pct) in enumerate(u["ranked_0dte"], start=1):
        add(f"GEX {i} 0DTE", k, pct, GEX_LEVEL_COLOR, "GEX Filters")

    add("1D Expected Move High", u["em_high"], 68, "color.new(color.yellow, 30)", "Level Filters")
    add("1D Expected Move Low", u["em_low"], 68, "color.new(color.yellow, 30)", "Level Filters")

    return lv, display_spot


def _pine_ident(i):
    return f"lvl{i}"


def tooltip_for(base_name):
    n = base_name
    if n == "Call Wall":
        return "Strike with the largest positive gamma exposure. Often acts like resistance."
    if n == "Put Wall":
        return "Strike with the largest negative gamma exposure. Often acts like support."
    if n == "Gamma Flip":
        return "The price where dealer hedging flips character. Below it, hedging tends to amplify moves. Above it, hedging tends to dampen moves. A regime marker, not a wall."
    if n.startswith("Gamma Wall"):
        base = "Strike with the single largest TOTAL gamma exposure (calls + puts combined) - the biggest overall hedging pressure point, regardless of direction."
        return base + " Today's expiry only." if "0DTE" in n else base
    if n.startswith("Call Resistance") and "OI" in n:
        base = "Strike with the most open call contracts outstanding. A raw positioning measure, separate from gamma exposure."
        return base + " Today's expiry only." if "0DTE" in n else base
    if n.startswith("Put Support") and "OI" in n:
        base = "Strike with the most open put contracts outstanding. A raw positioning measure, separate from gamma exposure."
        return base + " Today's expiry only." if "0DTE" in n else base
    if n.startswith("Call Resistance"):
        return "Same concept as Call Wall, using only options expiring today (0DTE)."
    if n.startswith("Put Support"):
        return "Same concept as Put Wall, using only options expiring today (0DTE)."
    if n.startswith("GEX"):
        base = "A strong gamma-exposure strike, ranked by magnitude."
        return base + " Today's expiry only." if "0DTE" in n else base
    if "Expected Move" in n:
        return "Estimated 1-day price range based on implied volatility. A statistical estimate, not a hard boundary."
    return ""


def generate_pine_script(levels, source_meta, generated_at):
    lines = []
    lines.append('//@version=6')
    lines.append('indicator("May Gamma & Gex Levels", overlay=true, max_lines_count=500, max_labels_count=500)')
    lines.append('')
    lines.append('// ============================================================')
    lines.append(f'// AUTO-GENERATED — {generated_at} UTC')
    for s in source_meta:
        conv = f'{s["underlying"]} spot {s["etf_spot"]:.2f} -> {s["key"]} spot {s["display_spot"]:.2f}' if not s["native"] else f'{s["key"]} spot {s["display_spot"]:.2f} (native, no conversion)'
        lines.append(f'// {s["key"]}: from {s["underlying"]} options, main expiry {s["main_expiry"]}'
                      + (f', 0DTE {s["zero_dte_expiry"]}' if s["zero_dte_expiry"] else ' (no 0DTE today)')
                      + f' | {conv}')
    lines.append('// FREE-DATA ESTIMATE using the standard public GEX convention')
    lines.append('// (dealers long calls / short puts). NOT SpotGamma\'s or')
    lines.append('// MenthorQ\'s proprietary model.')
    lines.append('// Paste this whole script over the old version each morning.')
    lines.append('// ============================================================')
    lines.append('')

    lines.append('labelSizeInput = input.string("Normal", "Labels Text Size", options=["Small", "Normal", "Large"], group="Display")')
    lines.append('labelPosInput = input.string("Above", "Label Position", options=["Above", "Below", "Left", "Right"], group="Display")')
    lines.append('offsetBars = input.int(3, "Left/Right Offset (bars)", minval=0, maxval=200, group="Display")')
    lines.append('showPctInput = input.bool(true, "Show % on labels", group="Display")')
    lines.append('mergeWithinInput = input.float(15.0, "Merge levels within (pts)", minval=0.0, tooltip="When two levels land within this many points of each other, only the higher-priority one is shown. Set to 0 to show every level.", group="Display")')
    lines.append('lineWidthInput = input.int(3, "Line Width", minval=1, maxval=6, group="Display")')
    lines.append('')
    lines.append('f_labelSize(s) =>')
    lines.append('    s == "Small" ? size.small : s == "Large" ? size.large : size.normal')
    lines.append('f_labelStyle(p) =>')
    lines.append('    p == "Below" ? label.style_label_up : p == "Left" ? label.style_label_right : p == "Right" ? label.style_label_left : label.style_label_down')
    lines.append('resolvedSize = f_labelSize(labelSizeInput)')
    lines.append('resolvedStyle = f_labelStyle(labelPosInput)')
    lines.append('')

    lines.append('autoDetectInput = input.bool(true, "Auto-detect chart symbol", tooltip="When on, only the level set matching the symbol you have loaded is shown automatically (e.g. an NQ chart shows NQ levels). Turn off to control visibility fully with the toggles below.", group="Source Visibility")')
    for s in source_meta:
        lines.append(f'showSource_{s["key"]} = input.bool(true, "Show {s["key"]} Levels", group="Source Visibility")')
        lines.append(f'matchSym_{s["key"]} = str.contains(syminfo.ticker, "{s["key"]}") or str.contains(syminfo.root, "{s["key"]}")')
        lines.append(f'effectiveShow_{s["key"]} = showSource_{s["key"]} and (not autoDetectInput or matchSym_{s["key"]})')
    lines.append('')

    for i, lv in enumerate(levels):
        ident = _pine_ident(i)
        base_name_for_tip = lv["name"].rsplit(" (", 1)[0]
        tip = tooltip_for(base_name_for_tip)
        tooltip_arg = f', tooltip="{tip}"' if tip else ''
        lines.append(f'show_{ident} = input.bool(true, "{lv["name"]}", group="{lv["group"]}"{tooltip_arg})')
    lines.append('')

    for i, lv in enumerate(levels):
        ident = _pine_ident(i)
        lines.append(f'col_{ident} = input.color({lv["color"]}, "{lv["name"]} Color", group="Colors")')
    lines.append('')

    for i in range(len(levels)):
        ident = _pine_ident(i)
        lines.append(f'var line ln_{ident} = na')
        lines.append(f'var label lbl_{ident} = na')
    lines.append('')

    lines.append('var float[] activePrices = array.new_float(0)')

    CHUNK_SIZE = 12
    for chunk_start in range(0, len(levels), CHUNK_SIZE):
        chunk = list(enumerate(levels))[chunk_start:chunk_start + CHUNK_SIZE]
        lines.append('if barstate.islast')
        if chunk_start == 0:
            lines.append('    array.clear(activePrices)')
        for i, lv in chunk:
            ident = _pine_ident(i)
            price = f'{lv["price"]:.2f}'
            lines.append(f'    label.delete(lbl_{ident})')
            lines.append(f'    line.delete(ln_{ident})')
            lines.append(f'    tooClose_{ident} = false')
            lines.append('    if mergeWithinInput > 0 and array.size(activePrices) > 0')
            lines.append('        for j = 0 to array.size(activePrices) - 1')
            lines.append(f'            if math.abs({price} - array.get(activePrices, j)) <= mergeWithinInput')
            lines.append(f'                tooClose_{ident} := true')
            lines.append(f'    if show_{ident} and effectiveShow_{lv["source"]} and not tooClose_{ident}')
            lines.append(f'        array.push(activePrices, {price})')
            lines.append(f'        ln_{ident} := line.new(bar_index - 10, {price}, bar_index, {price}, color=col_{ident}, width=lineWidthInput, extend=extend.both)')
            lines.append(f'        txt_{ident} = showPctInput ? "{lv["name"]}  {lv["pct"]}%" : "{lv["name"]}"')
            lines.append(
                f'        lbl_{ident} := label.new(bar_index + offsetBars, {price}, txt_{ident}, '
                f'xloc=xloc.bar_index, style=resolvedStyle, color=color.new(color.black, 100), '
                f'textcolor=col_{ident}, size=resolvedSize)'
            )
        lines.append('')

    return "\n".join(lines)


def main():
    generated_at = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    today = dt.date.today()

    underlying_cache = {}
    for source in SOURCES:
        underlying_cache[source["underlying"]] = compute_underlying(source["options_ticker"], today)

    all_levels = []
    source_meta = []
    for target in DISPLAY_TARGETS:
        u = underlying_cache[target["underlying"]]
        lv, display_spot = build_levels_for_target(target, u)
        all_levels.extend(lv)
        source_meta.append({
            "key": target["key"], "underlying": target["underlying"], "native": target["native"],
            "main_expiry": u["main_expiry"], "zero_dte_expiry": u["zero_dte_expiry"],
            "etf_spot": u["etf_spot"], "display_spot": display_spot,
        })

    pine = generate_pine_script(all_levels, source_meta, generated_at)

    with open("output.pine", "w") as f:
        f.write(pine)

    print(f"Generated output.pine with {len(all_levels)} total levels across {len(DISPLAY_TARGETS)} display targets")
    for lv in all_levels:
        print(f"  {lv['name']:<32} {lv['price']:.2f}  ({lv['pct']}%)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
