"""
NQ Gamma Exposure (GEX) Level Generator
-----------------------------------------
Free-data estimate of gamma exposure levels for Nasdaq futures (NQ),
derived from QQQ options open interest (the closest free, liquid proxy
for NDX/NQ). Outputs a complete, ready-to-paste TradingView Pine Script
with the levels hardcoded.

IMPORTANT / HONEST LIMITATIONS:
- Data source is Yahoo Finance's free, unofficial feed (via yfinance).
  It is NOT a licensed real-time feed. Expect occasional delays,
  missing data, or breakage if Yahoo changes something.
- GEX is computed using the standard public convention (dealers assumed
  long calls / short puts). This is the same simplifying assumption used
  by nearly every free GEX calculator. It is NOT SpotGamma's or
  MenthorQ's proprietary model and will not exactly match their numbers.
- QQQ options are used as a proxy for NDX/NQ, scaled via the live
  NQ/QQQ price ratio at run time.
- "Strength %" on ranked GEX levels is this strike's gamma exposure
  magnitude relative to the single largest strike found in the chain
  (100% = the biggest one). It is NOT a probability or a confidence score.
- "Call Resistance (OI)" / "Put Support (OI)" are based on raw open
  interest (contract count), a different and complementary metric from
  the gamma-exposure-based Call Wall / Put Wall.
- "Gamma Wall" is the strike with the largest TOTAL gamma exposure
  (calls + puts combined, unsigned) - i.e. where the biggest overall
  dealer hedging flow is concentrated, regardless of direction.
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


def bs_gamma(spot, strike, t_years, iv, r=RISK_FREE_RATE):
    if t_years <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
        return 0.0
    d1 = (math.log(spot / strike) + (r + 0.5 * iv ** 2) * t_years) / (iv * math.sqrt(t_years))
    return norm.pdf(d1) / (spot * iv * math.sqrt(t_years))


def pick_expiration(expirations, today):
    """Pick the nearest expiration STRICTLY AFTER today (skips 0DTE
    noise for the 'main' calculation; 0DTE gets its own separate pass)."""
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


def compute_expiry_data(qqq, expiry, spot, today):
    """Returns per-strike GEX, open interest (calls/puts separately),
    total unsigned gamma exposure, plus time-to-expiry and ATM IV."""
    exp_date = dt.datetime.strptime(expiry, "%Y-%m-%d").date()
    t_years = max((exp_date - today).days, 0.5) / 365.0

    chain = qqq.option_chain(expiry)
    calls, puts = chain.calls, chain.puts
    strikes = sorted(set(calls["strike"]).union(set(puts["strike"])))

    gex_by_strike = {}
    call_oi_by_strike = {}
    put_oi_by_strike = {}
    total_gamma_by_strike = {}
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
        "gex": gex_by_strike,
        "call_oi": call_oi_by_strike,
        "put_oi": put_oi_by_strike,
        "total_gamma": total_gamma_by_strike,
        "t_years": t_years,
        "atm_iv": atm_iv,
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
        key=lambda kv: abs(kv[1]),
        reverse=True,
    )[:n]
    out = []
    for k, v in ranked:
        pct = round(100 * abs(v) / max_abs) if max_abs else 0
        out.append((k, pct, v >= 0))
    return out


def fetch_nq_price():
    nq = yf.Ticker("NQ=F")
    hist = nq.history(period="1d")
    if hist.empty:
        raise RuntimeError("Could not fetch NQ=F price.")
    return float(hist["Close"].iloc[-1])


def to_nq(price_qqq, qqq_spot, nq_spot):
    return price_qqq * (nq_spot / qqq_spot)


def oi_and_gamma_wall(data):
    call_res_oi = max(data["call_oi"], key=lambda k: data["call_oi"][k]) if data["call_oi"] else None
    put_sup_oi = max(data["put_oi"], key=lambda k: data["put_oi"][k]) if data["put_oi"] else None
    gamma_wall = max(data["total_gamma"], key=lambda k: data["total_gamma"][k]) if data["total_gamma"] else None
    return call_res_oi, put_sup_oi, gamma_wall


def build_levels(qqq_spot, nq_spot, today):
    qqq = yf.Ticker("QQQ")
    expirations = qqq.options
    if not expirations:
        raise RuntimeError("No QQQ option expirations returned.")

    main_expiry = pick_expiration(expirations, today)
    main = compute_expiry_data(qqq, main_expiry, qqq_spot, today)
    gex_main = main["gex"]

    call_wall, put_wall = wall_levels(gex_main)
    gamma_flip = gamma_flip_level(gex_main, qqq_spot)
    ranked = top_n_ranked(gex_main, exclude_strikes={call_wall, put_wall}, n=TOP_N_LEVELS)
    call_res_oi, put_sup_oi, gamma_wall = oi_and_gamma_wall(main)

    today_str = today.strftime("%Y-%m-%d")
    zero_dte_expiry = today_str if today_str in expirations else None
    call_res_0dte = put_sup_0dte = None
    ranked_0dte = []
    call_res_oi_0dte = put_sup_oi_0dte = gamma_wall_0dte = None
    if zero_dte_expiry:
        z = compute_expiry_data(qqq, zero_dte_expiry, qqq_spot, today)
        gex_0dte = z["gex"]
        if gex_0dte:
            call_res_0dte, put_sup_0dte = wall_levels(gex_0dte)
            ranked_0dte = top_n_ranked(gex_0dte, exclude_strikes={call_res_0dte, put_sup_0dte}, n=TOP_N_LEVELS_0DTE)
            call_res_oi_0dte, put_sup_oi_0dte, gamma_wall_0dte = oi_and_gamma_wall(z)

    atm_iv = clamp_iv(main["atm_iv"])
    em = qqq_spot * atm_iv * math.sqrt(1.0 / 365.0)
    em_low, em_high = qqq_spot - em, qqq_spot + em

    levels = []
    levels.append({"name": "Call Wall", "price": to_nq(call_wall, qqq_spot, nq_spot), "pct": 100, "color": "color.green", "group": "Level Filters"})
    levels.append({"name": "Put Wall", "price": to_nq(put_wall, qqq_spot, nq_spot), "pct": 100, "color": "color.red", "group": "Level Filters"})
    levels.append({"name": "Gamma Flip", "price": to_nq(gamma_flip, qqq_spot, nq_spot), "pct": 100, "color": "color.orange", "group": "Level Filters"})
    levels.append({"name": "Gamma Wall", "price": to_nq(gamma_wall, qqq_spot, nq_spot), "pct": 100, "color": "color.white", "group": "Level Filters"})
    levels.append({"name": "Call Resistance (OI)", "price": to_nq(call_res_oi, qqq_spot, nq_spot), "pct": 90, "color": "color.lime", "group": "Level Filters"})
    levels.append({"name": "Put Support (OI)", "price": to_nq(put_sup_oi, qqq_spot, nq_spot), "pct": 90, "color": "color.maroon", "group": "Level Filters"})

    if call_res_0dte is not None:
        levels.append({"name": "Call Resistance 0DTE", "price": to_nq(call_res_0dte, qqq_spot, nq_spot), "pct": 95, "color": "color.new(color.red, 20)", "group": "Level Filters"})
    if put_sup_0dte is not None:
        levels.append({"name": "Put Support 0DTE", "price": to_nq(put_sup_0dte, qqq_spot, nq_spot), "pct": 95, "color": "color.new(color.teal, 20)", "group": "Level Filters"})
    if gamma_wall_0dte is not None:
        levels.append({"name": "Gamma Wall 0DTE", "price": to_nq(gamma_wall_0dte, qqq_spot, nq_spot), "pct": 90, "color": "color.silver", "group": "Level Filters"})
    if call_res_oi_0dte is not None:
        levels.append({"name": "Call Resistance 0DTE (OI)", "price": to_nq(call_res_oi_0dte, qqq_spot, nq_spot), "pct": 85, "color": "color.new(color.lime, 20)", "group": "Level Filters"})
    if put_sup_oi_0dte is not None:
        levels.append({"name": "Put Support 0DTE (OI)", "price": to_nq(put_sup_oi_0dte, qqq_spot, nq_spot), "pct": 85, "color": "color.new(color.maroon, 20)", "group": "Level Filters"})

    for i, (k, pct, is_call_side) in enumerate(ranked, start=1):
        levels.append({
            "name": f"GEX {i}",
            "price": to_nq(k, qqq_spot, nq_spot),
            "pct": pct,
            "color": "color.new(color.aqua, 10)" if is_call_side else "color.new(color.fuchsia, 10)",
            "group": "GEX Filters",
        })

    for i, (k, pct, is_call_side) in enumerate(ranked_0dte, start=1):
        levels.append({
            "name": f"GEX {i} 0DTE",
            "price": to_nq(k, qqq_spot, nq_spot),
            "pct": pct,
            "color": "color.new(color.blue, 15)" if is_call_side else "color.new(color.purple, 15)",
            "group": "GEX Filters",
        })

    levels.append({"name": "1D Expected Move High", "price": to_nq(em_high, qqq_spot, nq_spot), "pct": 68, "color": "color.new(color.yellow, 30)", "group": "Level Filters"})
    levels.append({"name": "1D Expected Move Low", "price": to_nq(em_low, qqq_spot, nq_spot), "pct": 68, "color": "color.new(color.yellow, 30)", "group": "Level Filters"})

    return levels, main_expiry, zero_dte_expiry


def _pine_ident(i):
    return f"lvl{i}"


def tooltip_for(name):
    """Plain-language explanation shown as a hover tooltip in the settings
    panel next to each level's show/hide toggle."""
    n = name
    if n == "Call Wall":
        return "Strike with the largest positive gamma exposure. Often acts like resistance - price stalling or reversing down here is common."
    if n == "Put Wall":
        return "Strike with the largest negative gamma exposure. Often acts like support - price stalling or reversing up here is common."
    if n == "Gamma Flip":
        return "The price where dealer hedging flips character. Below it, hedging tends to amplify moves (more volatile). Above it, hedging tends to dampen moves (more range-bound). A regime marker, not a wall."
    if n.startswith("Gamma Wall"):
        base = "Strike with the single largest TOTAL gamma exposure (calls + puts combined) - the biggest overall hedging pressure point, regardless of direction."
        return base + " Calculated using only today's expiring options." if "0DTE" in n else base
    if n.startswith("Call Resistance") and "OI" in n:
        base = "Strike with the most open call contracts outstanding. A raw positioning measure, separate from gamma exposure."
        return base + " Today's expiry only." if "0DTE" in n else base
    if n.startswith("Put Support") and "OI" in n:
        base = "Strike with the most open put contracts outstanding. A raw positioning measure, separate from gamma exposure."
        return base + " Today's expiry only." if "0DTE" in n else base
    if n.startswith("Call Resistance"):
        return "Same concept as Call Wall, but calculated using only options expiring today (0DTE)."
    if n.startswith("Put Support"):
        return "Same concept as Put Wall, but calculated using only options expiring today (0DTE)."
    if n.startswith("GEX"):
        base = "A strong gamma-exposure strike, ranked by magnitude (not necessarily the single strongest - see the number)."
        return base + " Today's expiry only." if "0DTE" in n else base
    if "Expected Move" in n:
        return "Estimated 1-day price range based on implied volatility. A statistical estimate, not a hard boundary."
    return ""


def generate_pine_script(levels, main_expiry, zero_dte_expiry, qqq_spot, nq_spot, generated_at):
    lines = []
    lines.append('//@version=6')
    lines.append('indicator("NQ GEX Levels (Auto-Generated, Free Data Estimate)", overlay=true, max_lines_count=100, max_labels_count=100)')
    lines.append('')
    lines.append('// ============================================================')
    lines.append(f'// AUTO-GENERATED — {generated_at} UTC')
    lines.append(f'// Source: QQQ options chain (free/unofficial). Main expiry: {main_expiry}' + (f', 0DTE: {zero_dte_expiry}' if zero_dte_expiry else ' (no 0DTE chain available today)'))
    lines.append(f'// QQQ spot at calc time: {qqq_spot:.2f} | NQ spot at calc time: {nq_spot:.2f}')
    lines.append('// FREE-DATA ESTIMATE using the standard public GEX convention')
    lines.append('// (dealers long calls / short puts). NOT SpotGamma\'s or')
    lines.append('// MenthorQ\'s proprietary model. "Strength %" = this level\'s')
    lines.append('// magnitude relative to the single strongest strike found.')
    lines.append('// Paste this whole script over the old version each morning.')
    lines.append('// ============================================================')
    lines.append('')

    # ---- Display settings group ----
    lines.append('labelSizeInput = input.string("Normal", "Labels Text Size", options=["Small", "Normal", "Large"], group="Display")')
    lines.append('labelPosInput = input.string("Above", "Label Position", options=["Above", "Below", "Left", "Right"], group="Display")')
    lines.append('offsetBars = input.int(3, "Left/Right Offset (bars)", minval=0, maxval=200, group="Display")')
    lines.append('showPctInput = input.bool(true, "Show % on labels", group="Display")')
    lines.append('mergeWithinInput = input.float(15.0, "Merge levels within (pts)", minval=0.0, tooltip="When two levels land within this many points of each other, only the higher-priority one is shown, to reduce clutter. Set to 0 to show every level.", group="Display")')
    lines.append('lineWidthInput = input.int(3, "Line Width", minval=1, maxval=6, group="Display")')
    lines.append('')
    lines.append('f_labelSize(s) =>')
    lines.append('    s == "Small" ? size.small : s == "Large" ? size.large : size.normal')
    lines.append('f_labelStyle(p) =>')
    lines.append('    p == "Below" ? label.style_label_up : p == "Left" ? label.style_label_right : p == "Right" ? label.style_label_left : label.style_label_down')
    lines.append('resolvedSize = f_labelSize(labelSizeInput)')
    lines.append('resolvedStyle = f_labelStyle(labelPosInput)')
    lines.append('')

    # ---- Per-level show/hide toggles, grouped like a Level Filters / GEX Filters panel ----
    for i, lv in enumerate(levels):
        ident = _pine_ident(i)
        tip = tooltip_for(lv["name"])
        tooltip_arg = f', tooltip="{tip}"' if tip else ''
        lines.append(f'show_{ident} = input.bool(true, "{lv["name"]}", group="{lv["group"]}"{tooltip_arg})')
    lines.append('')

    # ---- Per-level color pickers, own settings group, defaulting to the computed color ----
    for i, lv in enumerate(levels):
        ident = _pine_ident(i)
        lines.append(f'col_{ident} = input.color({lv["color"]}, "{lv["name"]} Color", group="Colors")')
    lines.append('')

    # ---- var line/label declarations ----
    for i in range(len(levels)):
        ident = _pine_ident(i)
        lines.append(f'var line ln_{ident} = na')
        lines.append(f'var label lbl_{ident} = na')
    lines.append('')

    # ---- Rebuild everything on the last bar: delete old, redraw active, respecting toggles + merge distance ----
    lines.append('var float[] activePrices = array.new_float(0)')
    lines.append('if barstate.islast')
    lines.append('    array.clear(activePrices)')
    for i, lv in enumerate(levels):
        ident = _pine_ident(i)
        price = f'{lv["price"]:.2f}'
        lines.append(f'    label.delete(lbl_{ident})')
        lines.append(f'    line.delete(ln_{ident})')
        lines.append(f'    tooClose_{ident} = false')
        lines.append('    if mergeWithinInput > 0 and array.size(activePrices) > 0')
        lines.append('        for j = 0 to array.size(activePrices) - 1')
        lines.append(f'            if math.abs({price} - array.get(activePrices, j)) <= mergeWithinInput')
        lines.append(f'                tooClose_{ident} := true')
        lines.append(f'    if show_{ident} and not tooClose_{ident}')
        lines.append(f'        array.push(activePrices, {price})')
        lines.append(f'        ln_{ident} := line.new(bar_index - 10, {price}, bar_index, {price}, color=col_{ident}, width=lineWidthInput, extend=extend.both)')
        lines.append(
            f'        txt_{ident} = showPctInput ? "{lv["name"]}  {lv["pct"]}%" : "{lv["name"]}"'
        )
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

    qqq_hist = yf.Ticker("QQQ").history(period="1d")
    if qqq_hist.empty:
        raise RuntimeError("Could not fetch QQQ spot price.")
    qqq_spot = float(qqq_hist["Close"].iloc[-1])

    nq_spot = fetch_nq_price()

    levels, main_expiry, zero_dte_expiry = build_levels(qqq_spot, nq_spot, today)
    pine = generate_pine_script(levels, main_expiry, zero_dte_expiry, qqq_spot, nq_spot, generated_at)

    with open("output.pine", "w") as f:
        f.write(pine)

    print(f"Generated output.pine with {len(levels)} levels | QQQ {qqq_spot:.2f} -> NQ {nq_spot:.2f}")
    for lv in levels:
        print(f"  {lv['name']:<28} {lv['price']:.2f}  ({lv['pct']}%)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
