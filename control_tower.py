"""
Parcel Network Control Tower — One-Day Delivery Planning
=========================================================
A planning tool for supply chain / network analysts at a national parcel or
e-commerce carrier. For a sample planning day it decides, for every order:

    1. WHICH facility ships it   (inventory + capacity aware sourcing)
    2. WHICH mode carries it     (ground linehaul vs. next-day air via the air hub)
    3. WHETHER it meets promise  (order cutoff -> pick/pack -> linehaul/flight ->
                                  hub sort -> delivery station -> last mile)

and minimises total landed cost (zone-based rates + fuel + residential +
handling + packaging + late-delivery penalties) subject to facility capacity
and nightly air-lift capacity, using a linear program (HiGHS).

Run:   pip install streamlit plotly scikit-learn scipy pandas numpy requests
       streamlit run control_tower.py

Optional AI analyst: set GEMINI_API_KEY as an environment variable, in
.streamlit/secrets.toml, or paste it on the "AI Network Analyst" page.
Only aggregated KPIs are sent to the model — never order-level rows.

All volumes are a scaled sample day. Rates, speeds and times are planning
assumptions, editable in the sidebar and on the "Data & Assumptions" page.
"""

import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# ─────────────────────────────────────────────────────────────────────────────
# Constants & reference data
# ─────────────────────────────────────────────────────────────────────────────
APP_TITLE = "Parcel Network Control Tower"
EARTH_RADIUS_MILES = 3958.7613
UNFULFILLED = "Unfulfilled"

NAVY, TEAL, AMBER, RED, GREY, LIGHT = "#0B1F3A", "#0E7C86", "#E09F1F", "#C0392B", "#64748B", "#EEF4F7"
MODE_COLORS = {"Ground": TEAL, "Air": AMBER, UNFULFILLED: RED}

SERVICE_PROMISE_DAYS = {"Next Day": 1, "2 Day": 2, "Ground": 5}
SERVICE_MIX = {"Next Day": 0.45, "2 Day": 0.35, "Ground": 0.20}
CATEGORY_MIX = {"A": 0.60, "B": 0.30, "C": 0.10}  # A = fast movers, C = long tail

# Zone bands on origin→destination great-circle miles (zones 2–8, carrier style)
ZONE_BREAKS = [150, 300, 600, 1000, 1400, 1800]
ZONES = [2, 3, 4, 5, 6, 7, 8]
# Contract (discounted) sample rate card — base $ and $ per billable lb, by zone
GROUND_BASE = np.array([5.10, 5.45, 5.90, 6.40, 6.95, 7.40, 7.95])
GROUND_PER_LB = np.array([0.12, 0.15, 0.20, 0.28, 0.36, 0.42, 0.48])
AIR_BASE = np.array([14.00, 16.50, 19.00, 21.50, 23.50, 25.00, 26.50])
AIR_PER_LB = np.array([0.90, 1.10, 1.40, 1.70, 1.90, 2.10, 2.30])
DIM_DIVISOR = 139

# (market, region, lat, lon, tz offset vs Eastern, demand weight)
MARKETS = pd.DataFrame(
    [
        ("New York, NY", "Northeast", 40.7128, -74.0060, 0, 0.11),
        ("Jersey City / Newark, NJ", "Northeast", 40.7300, -74.1000, 0, 0.05),
        ("Boston, MA", "Northeast", 42.3601, -71.0589, 0, 0.05),
        ("Philadelphia, PA", "Northeast", 39.9526, -75.1652, 0, 0.05),
        ("Washington, DC", "Mid-Atlantic", 38.9072, -77.0369, 0, 0.05),
        ("Charlotte, NC", "Southeast", 35.2271, -80.8431, 0, 0.03),
        ("Atlanta, GA", "Southeast", 33.7490, -84.3880, 0, 0.05),
        ("Orlando, FL", "Southeast", 28.5383, -81.3792, 0, 0.03),
        ("Miami, FL", "Southeast", 25.7617, -80.1918, 0, 0.05),
        ("Nashville, TN", "Central", 36.1627, -86.7816, -1, 0.03),
        ("Detroit, MI", "Midwest", 42.3314, -83.0458, 0, 0.03),
        ("Chicago, IL", "Midwest", 41.8781, -87.6298, -1, 0.07),
        ("Minneapolis, MN", "Midwest", 44.9778, -93.2650, -1, 0.03),
        ("Dallas, TX", "South Central", 32.7767, -96.7970, -1, 0.05),
        ("Houston, TX", "South Central", 29.7604, -95.3698, -1, 0.05),
        ("Denver, CO", "Mountain", 39.7392, -104.9903, -2, 0.03),
        ("Phoenix, AZ", "Mountain", 33.4484, -112.0740, -3, 0.03),
        ("Los Angeles, CA", "West", 34.0522, -118.2437, -3, 0.08),
        ("San Francisco, CA", "West", 37.7749, -122.4194, -3, 0.04),
        ("Seattle, WA", "West", 47.6062, -122.3321, -3, 0.03),
    ],
    columns=["Market", "Region", "Lat", "Lon", "TZ_Offset", "Weight"],
)

DEFAULT_FACILITIES = pd.DataFrame(
    [
        # id, name, city, type, lat, lon, tz, handling $, daily cap, stocked, cutoff, departure
        ("FC-EWR1", "Edison FC", "Edison, NJ", "Fulfillment Center", 40.5187, -74.4121, 0, 2.40, 520, "A,B,C", 20, 22),
        ("FC-ABE1", "Allentown FC", "Allentown, PA", "Fulfillment Center", 40.6084, -75.4902, 0, 2.30, 380, "A,B", 20, 22),
        ("FSL-JFK1", "Queens Forward Stocking", "Queens, NY", "Forward Stocking Location", 40.7282, -73.7949, 0, 3.40, 220, "A", 18, 21),
        ("HUB-SDF", "Louisville Air Hub & FC", "Louisville, KY", "Air Hub + FC", 38.2527, -85.7585, 0, 2.60, 360, "A,B,C", 21, 23),
        ("FC-ATL1", "Atlanta FC", "Atlanta, GA", "Fulfillment Center", 33.7490, -84.3880, 0, 2.35, 300, "A,B", 20, 22),
        ("FSL-MCO1", "Orlando Forward Stocking", "Orlando, FL", "Forward Stocking Location", 28.5383, -81.3792, 0, 3.10, 160, "A", 18, 21),
        ("FC-ORD1", "Joliet FC", "Joliet, IL", "Fulfillment Center", 41.5250, -88.0817, -1, 2.35, 380, "A,B,C", 20, 22),
        ("FC-DFW1", "Dallas–Fort Worth FC", "Fort Worth, TX", "Fulfillment Center", 32.7555, -97.3308, -1, 2.30, 330, "A,B,C", 20, 22),
        ("FC-DEN1", "Denver FC", "Aurora, CO", "Fulfillment Center", 39.7294, -104.8319, -2, 2.50, 160, "A,B", 19, 21),
        ("FC-PHX1", "Phoenix FC", "Goodyear, AZ", "Fulfillment Center", 33.4353, -112.3577, -3, 2.40, 170, "A,B", 19, 21),
        ("FC-ONT1", "Inland Empire FC", "Ontario, CA", "Fulfillment Center", 34.0633, -117.6509, -3, 2.45, 420, "A,B,C", 20, 22),
        ("FC-SEA1", "Seattle FC", "Kent, WA", "Fulfillment Center", 47.3809, -122.2348, -3, 2.55, 200, "A,B", 19, 21),
    ],
    columns=[
        "Facility_ID", "Name", "City", "Type", "Lat", "Lon", "TZ_Offset", "Handling_Cost",
        "Daily_Capacity", "Stocked_Categories", "Order_Cutoff_Hour", "Linehaul_Departure_Hour",
    ],
)
AIR_HUB_FALLBACK = (38.2527, -85.7585)  # used if no facility is typed as an air hub

ORDER_COLUMNS = [
    "Order_ID", "Dest_Market", "Dest_Region", "Dest_Lat", "Dest_Lon", "Dest_TZ",
    "Order_Hour_ET", "Service_Level", "SKU_Category", "Weight_Lb",
    "Length_In", "Width_In", "Height_In", "Order_Value",
]

# ─────────────────────────────────────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────────────────────────────────────
CSS = f"""
<style>
.block-container {{padding-top: 1.6rem; max-width: 1400px;}}
h1, h2, h3 {{color: {NAVY}; letter-spacing: -0.01em;}}
.app-header {{background: linear-gradient(90deg, {NAVY} 0%, #13345E 100%); color: white;
  padding: 18px 24px; border-radius: 10px; margin-bottom: 14px;}}
.app-header .eyebrow {{font-size: 11px; letter-spacing: .14em; text-transform: uppercase; color: #9FC9D0;}}
.app-header .title {{font-size: 26px; font-weight: 700; margin: 2px 0;}}
.app-header .sub {{font-size: 14px; color: #D3DEE8;}}
.kpi-grid {{display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 10px; margin: 6px 0 14px;}}
.kpi {{background: white; border: 1px solid #DCE5EA; border-left: 4px solid {TEAL}; border-radius: 8px; padding: 10px 14px;}}
.kpi.warn {{border-left-color: {AMBER};}} .kpi.bad {{border-left-color: {RED};}}
.kpi .label {{font-size: 11.5px; color: {GREY}; text-transform: uppercase; letter-spacing: .05em;}}
.kpi .value {{font-size: 24px; font-weight: 700; color: {NAVY}; line-height: 1.25;}}
.kpi .delta {{font-size: 12px; color: {GREY};}}
.lens {{background: {LIGHT}; border: 1px solid #C8DCE2; border-radius: 8px; padding: 12px 16px; margin: 4px 0 16px;}}
.lens .tag {{font-size: 11px; font-weight: 700; letter-spacing: .12em; color: {TEAL}; text-transform: uppercase;}}
.lens ul {{margin: 6px 0 0 0; padding-left: 18px;}} .lens li {{margin-bottom: 3px; font-size: 14px;}}
.step {{border-left: 3px solid {TEAL}; padding: 4px 0 4px 14px; margin: 8px 0;}}
.step .n {{font-size: 12px; color: {TEAL}; font-weight: 700;}}
.foot {{font-size: 12px; color: {GREY};}}
section[data-testid="stSidebar"] {{background: #F6F9FB;}}
</style>
"""


def header(title, subtitle):
    st.markdown(
        f"<div class='app-header'><div class='eyebrow'>{APP_TITLE}</div>"
        f"<div class='title'>{title}</div><div class='sub'>{subtitle}</div></div>",
        unsafe_allow_html=True,
    )


def one_day_lens(points):
    items = "".join(f"<li>{p}</li>" for p in points)
    st.markdown(
        f"<div class='lens'><div class='tag'>How this page drives one-day delivery</div><ul>{items}</ul></div>",
        unsafe_allow_html=True,
    )


def kpis(items):
    """items: list of (label, value, note, status) with status in {'ok','warn','bad'}."""
    cards = "".join(
        f"<div class='kpi {s}'><div class='label'>{l}</div><div class='value'>{v}</div><div class='delta'>{n}</div></div>"
        for l, v, n, s in items
    )
    st.markdown(f"<div class='kpi-grid'>{cards}</div>", unsafe_allow_html=True)


def style_fig(fig, height=380):
    fig.update_layout(
        template="simple_white", height=height, margin=dict(l=10, r=10, t=50, b=10),
        font=dict(family="Inter, Segoe UI, sans-serif", size=12, color=NAVY),
        title_font=dict(size=15), legend_title_text="",
    )
    return fig


def pct(x):
    return f"{100 * x:.1f}%"


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def simulate_orders(n, seed=42):
    rng = np.random.default_rng(seed)
    w = MARKETS["Weight"] / MARKETS["Weight"].sum()
    m = MARKETS.iloc[rng.choice(len(MARKETS), size=n, p=w)].reset_index(drop=True)
    hour_w = np.array([1, .5, .3, .2, .2, .3, .6, 1.2, 2, 2.8, 3.4, 3.6, 3.8, 3.6, 3.3, 3.1,
                       3.0, 3.1, 3.4, 3.9, 4.2, 3.8, 2.8, 1.8])
    hours = rng.choice(24, size=n, p=hour_w / hour_w.sum()) + rng.uniform(0, 1, n)
    weight = np.clip(rng.lognormal(1.0, 0.75, n), 0.3, 50)
    side = np.cbrt(weight * 160) * rng.uniform(0.85, 1.15, n)
    return pd.DataFrame({
        "Order_ID": [f"ORD-{i:06d}" for i in range(1, n + 1)],
        "Dest_Market": m["Market"],
        "Dest_Region": m["Region"],
        "Dest_Lat": m["Lat"].to_numpy() + rng.normal(0, 0.08, n),
        "Dest_Lon": m["Lon"].to_numpy() + rng.normal(0, 0.10, n),
        "Dest_TZ": m["TZ_Offset"].to_numpy(),
        "Order_Hour_ET": np.round(hours, 2),
        "Service_Level": rng.choice(list(SERVICE_MIX), size=n, p=list(SERVICE_MIX.values())),
        "SKU_Category": rng.choice(list(CATEGORY_MIX), size=n, p=list(CATEGORY_MIX.values())),
        "Weight_Lb": np.round(weight, 2),
        "Length_In": np.round(side * 1.4, 1),
        "Width_In": np.round(side, 1),
        "Height_In": np.round(side * 0.6, 1),
        "Order_Value": np.round(np.clip(rng.lognormal(4.0, 0.6, n), 12, 900), 2),
    })


def validate_orders(df):
    missing = [c for c in ORDER_COLUMNS if c not in df.columns]
    if missing:
        return None, f"Missing columns: {', '.join(missing)}"
    df = df[ORDER_COLUMNS].copy()
    bad_service = ~df["Service_Level"].isin(SERVICE_PROMISE_DAYS)
    if bad_service.any():
        return None, f"Service_Level must be one of {list(SERVICE_PROMISE_DAYS)}"
    df = df.dropna(subset=["Dest_Lat", "Dest_Lon", "Order_Hour_ET"]).reset_index(drop=True)
    return df, None


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1 = np.radians(np.asarray(lat1, float))[:, None], np.radians(np.asarray(lon1, float))[:, None]
    lat2, lon2 = np.radians(np.asarray(lat2, float))[None, :], np.radians(np.asarray(lon2, float))[None, :]
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_MILES * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def billable_weight(orders):
    dim = orders["Length_In"] * orders["Width_In"] * orders["Height_In"] / DIM_DIVISOR
    return np.ceil(np.maximum(orders["Weight_Lb"], dim)).to_numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Network physics: time-in-transit, promise, zone rates
# ─────────────────────────────────────────────────────────────────────────────
def compute_options(orders, fac, p):
    """Return (N x F) arrays for both modes: delivery day, planned delivery hour,
    days late vs. promise, transport cost, and the inventory mask.
    Day 0 = today. Hours are counted from midnight of day 0, in the relevant local time."""
    fac_tz = fac["TZ_Offset"].to_numpy(float)[None, :]
    dest_tz = orders["Dest_TZ"].to_numpy(float)[:, None]
    order_et = orders["Order_Hour_ET"].to_numpy(float)[:, None]

    dist = haversine(orders["Dest_Lat"], orders["Dest_Lon"], fac["Lat"], fac["Lon"])
    road = dist * p["circuity"]

    # 1) Order cutoff -> ship day (facility local time)
    cutoff = fac["Order_Cutoff_Hour"].to_numpy(float)[None, :] + p.get("cutoff_shift", 0.0)
    ship_day = ((order_et + fac_tz) >= cutoff).astype(int)
    depart = ship_day * 24 + fac["Linehaul_Departure_Hour"].to_numpy(float)[None, :]
    tz_diff = dest_tz - fac_tz

    # 2) Ground: linehaul + hub sorts -> arrive delivery station -> out for delivery
    sorts = 1 + np.floor(road / 600)
    g_arrive = depart + road / p["ground_speed"] + sorts * p["sort_hours"] + tz_diff
    g_day = np.maximum(np.ceil((g_arrive - p["ofd_hour"]) / 24), 1)
    g_time = g_day * 24 + 14.0  # average residential stop ~2 pm

    # 3) Air: gateway tender -> flight to air hub -> hub sort -> outbound flight -> next-day delivery
    hubs = fac[fac["Type"].str.contains("Air Hub", case=False, na=False)]
    hub_lat, hub_lon = (hubs.iloc[0]["Lat"], hubs.iloc[0]["Lon"]) if len(hubs) else AIR_HUB_FALLBACK
    d_hub = haversine(fac["Lat"], fac["Lon"], [hub_lat], [hub_lon])[:, 0][None, :]
    flight = d_hub / 500 + 0.5
    is_hub = fac["Type"].str.contains("Air Hub", case=False, na=False).to_numpy()[None, :]
    drive = np.where(is_hub, 0.5, 1.5)
    tender = np.minimum(p["air_tender_hour"], p["hub_inbound_et"] + fac_tz - flight)
    ready = ship_day * 24 + cutoff + 1.0 + drive  # air wave released 1h after cutoff
    a_day = np.ceil((ready - tender) / 24) + 1
    a_time = a_day * 24 + 10.5

    # 4) Customer promise (checkout cutoff in customer local time)
    cust_local = order_et + dest_tz
    promise_day = (orders["Service_Level"].map(SERVICE_PROMISE_DAYS).to_numpy()[:, None]
                   + (cust_local >= p["promise_cutoff"]).astype(int))
    g_late = np.maximum(g_day - promise_day, 0)
    a_late = np.maximum(a_day - promise_day, 0)

    # 5) Zone-based transport cost
    zi = np.digitize(dist, ZONE_BREAKS)
    bw = billable_weight(orders)[:, None]
    mult = (1 + p["fuel_pct"]) * p["rate_mult"]
    g_cost = (GROUND_BASE[zi] + GROUND_PER_LB[zi] * bw) * mult + p["resi_fee"]
    a_cost = (AIR_BASE[zi] + AIR_PER_LB[zi] * bw) * mult + p["resi_fee"]

    stocked = [set(str(s).replace(" ", "").split(",")) for s in fac["Stocked_Categories"]]
    cat = orders["SKU_Category"].to_numpy()
    in_stock = np.column_stack([np.isin(cat, list(s)) for s in stocked])

    return dict(dist=dist, zone=np.array(ZONES)[zi], ship_day=ship_day, promise_day=promise_day,
                g_day=g_day, g_time=g_time, g_late=g_late, g_cost=g_cost,
                a_day=a_day, a_time=a_time, a_late=a_late, a_cost=a_cost,
                in_stock=in_stock, bw=bw[:, 0])


def penalty_per_day(orders, p):
    base = orders["Service_Level"].map({"Next Day": 35.0, "2 Day": 15.0, "Ground": 5.0}).to_numpy()
    return base * p["penalty_mult"]


@st.cache_data(show_spinner="Optimizing sourcing and mode selection…")
def solve_plan(orders, fac, p):
    o = compute_options(orders, fac, p)
    N, F = o["dist"].shape
    handling = fac["Handling_Cost"].to_numpy(float)[None, :] + p["packaging"]
    pen = penalty_per_day(orders, p)[:, None]

    transport = np.stack([o["g_cost"], o["a_cost"]], axis=2)
    late = np.stack([o["g_late"], o["a_late"]], axis=2)
    cost = transport + handling[:, :, None] + pen[:, :, None] * late
    allowed = np.repeat(o["in_stock"][:, :, None], 2, axis=2)
    if not p["air_enabled"]:
        allowed[:, :, 1] = False

    oi, fi, mi = np.nonzero(allowed)
    K = len(oi)
    c = np.concatenate([cost[oi, fi, mi], np.full(N, p["unfulfilled_cost"])])
    rows = np.concatenate([oi, np.arange(N)])
    A_eq = coo_matrix((np.ones(K + N), (rows, np.arange(K + N))), shape=(N, K + N)).tocsr()
    cap_rows = np.concatenate([fi, np.full((mi == 1).sum(), F)])
    cap_cols = np.concatenate([np.arange(K), np.flatnonzero(mi == 1)])
    A_ub = coo_matrix((np.ones(len(cap_rows)), (cap_rows, cap_cols)), shape=(F + 1, K + N)).tocsr()
    b_ub = np.append(np.floor(fac["Daily_Capacity"].to_numpy(float) * p["cap_mult"]), p["air_lift"])

    sol = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.ones(N), bounds=(0, 1), method="highs")
    if not sol.success:
        raise RuntimeError(f"Optimization failed: {sol.message}")

    pick = pd.DataFrame({"order": rows, "var": np.arange(K + N), "x": sol.x}).sort_values(
        ["order", "x"], ascending=[True, False]).drop_duplicates("order").sort_values("order")["var"].to_numpy()
    plan = build_plan(orders, fac, o, p, pick, oi, fi, mi, K)
    return plan, o


def build_plan(orders, fac, o, p, pick, oi, fi, mi, K):
    N = len(orders)
    unf = pick >= K
    f = np.where(unf, -1, fi[np.minimum(pick, K - 1)])
    m = np.where(unf, -1, mi[np.minimum(pick, K - 1)])
    r = np.arange(N)
    fs = np.maximum(f, 0)
    air = m == 1

    plan = orders.copy()
    plan["Assigned_Node"] = np.where(unf, UNFULFILLED, fac["Facility_ID"].to_numpy()[fs])
    plan["Mode"] = np.where(unf, UNFULFILLED, np.where(air, "Air", "Ground"))
    plan["Distance_Mi"] = np.where(unf, np.nan, o["dist"][r, fs]).round(0)
    plan["Zone"] = np.where(unf, np.nan, o["zone"][r, fs])
    plan["Billable_Lb"] = o["bw"]
    plan["Ship_Day"] = np.where(unf, np.nan, o["ship_day"][r, fs])
    plan["Promise_Day"] = o["promise_day"][r, 0]
    plan["Delivery_Day"] = np.where(unf, np.nan, np.where(air, o["a_day"][r, fs], o["g_day"][r, fs]))
    plan["Planned_Delivery_Hr"] = np.where(unf, np.nan, np.where(air, o["a_time"][r, fs], o["g_time"][r, fs]))
    plan["Days_Late"] = np.where(unf, np.nan, np.maximum(plan["Delivery_Day"] - plan["Promise_Day"], 0))
    plan["On_Time_Planned"] = plan["Days_Late"].eq(0)
    plan["Slack_Hrs"] = (plan["Promise_Day"] * 24 + 21) - plan["Planned_Delivery_Hr"]
    plan["Transport_Cost"] = np.where(unf, np.nan, np.where(air, o["a_cost"][r, fs], o["g_cost"][r, fs])).round(2)
    plan["Handling_Cost"] = np.where(unf, np.nan, fac["Handling_Cost"].to_numpy()[fs] + p["packaging"])
    plan["Late_Penalty"] = (plan["Days_Late"] * penalty_per_day(orders, p)).round(2)
    plan["Cost_to_Serve"] = (plan["Transport_Cost"] + plan["Handling_Cost"]).round(2)
    plan["Gross_Margin"] = (plan["Order_Value"] * p["margin_pct"]).round(2)
    plan["Contribution"] = (plan["Gross_Margin"] - plan["Cost_to_Serve"] - plan["Late_Penalty"]).round(2)

    # Root cause for every planned miss — where one-day breaks
    g_ok = (o["g_late"] == 0) & o["in_stock"]
    a_ok = (o["a_late"] == 0) & o["in_stock"] & p["air_enabled"]
    possible = (g_ok | a_ok).any(axis=1)
    any_node_on_time = ((o["g_late"] == 0) | ((o["a_late"] == 0) & p["air_enabled"])).any(axis=1)
    reason = np.full(N, "On time", dtype=object)
    late_mask = ~plan["On_Time_Planned"].to_numpy() | unf
    reason[late_mask & possible] = "Capacity / air-lift limit or cost trade-off"
    reason[late_mask & ~possible & any_node_on_time] = "Inventory not positioned within reach"
    reason[late_mask & ~possible & ~any_node_on_time] = "Ordered after network cutoff"
    reason[unf] = "Unfulfilled — no capacity"
    plan["Miss_Reason"] = reason
    return plan


@st.cache_data
def baseline_plan(orders, fac, p):
    """Legacy rule: ship from the nearest node that stocks the item, by ground, ignore capacity."""
    o = compute_options(orders, fac, p)
    d = np.where(o["in_stock"], o["dist"], np.inf)
    f = d.argmin(axis=1)
    r = np.arange(len(orders))
    late = o["g_late"][r, f]
    ctsv = o["g_cost"][r, f] + fac["Handling_Cost"].to_numpy()[f] + p["packaging"]
    load = pd.Series(fac["Facility_ID"].to_numpy()[f]).value_counts()
    over = (load.reindex(fac["Facility_ID"]).fillna(0).to_numpy()
            - np.floor(fac["Daily_Capacity"].to_numpy() * p["cap_mult"])).clip(min=0).sum()
    return dict(on_time=float((late == 0).mean()),
                nd_on_time=float((late[orders["Service_Level"].eq("Next Day").to_numpy()] == 0).mean()),
                cost=float(ctsv.mean()), penalty=float((late * penalty_per_day(orders, p)).mean()),
                overload=int(over))


@st.cache_data
def simulate_actuals(plan, util, disrupted_regions, seed=7):
    """Execution simulation: add realistic dwell/weather/congestion delays to the plan."""
    rng = np.random.default_rng(seed)
    df = plan[plan["Assigned_Node"] != UNFULFILLED].copy()
    air = df["Mode"].eq("Air").to_numpy()
    delay = np.where(air, rng.gamma(1.0, 0.8, len(df)), rng.gamma(1.3, 1.6, len(df)))
    weather = df["Dest_Region"].isin(disrupted_regions).to_numpy() & (rng.random(len(df)) < 0.55)
    delay += weather * rng.uniform(4, 14, len(df))
    u = df["Assigned_Node"].map(util).fillna(0).to_numpy()
    delay += (u > 0.92) * rng.gamma(1.5, 2.5, len(df))
    df["Actual_Delivery_Hr"] = df["Planned_Delivery_Hr"] + delay
    df["Late_Actual"] = df["Actual_Delivery_Hr"] > df["Promise_Day"] * 24 + 21
    df["Node_Utilization"] = u
    return df


@st.cache_data(show_spinner="Training SLA-risk model…")
def score_risk(actuals):
    X = pd.DataFrame({
        "Distance_Mi": actuals["Distance_Mi"], "Zone": actuals["Zone"],
        "Is_Air": actuals["Mode"].eq("Air").astype(int),
        "Order_Hour_ET": actuals["Order_Hour_ET"], "Slack_Hrs": actuals["Slack_Hrs"],
        "Node_Utilization": actuals["Node_Utilization"], "Billable_Lb": actuals["Billable_Lb"],
        "Region_Code": actuals["Dest_Region"].astype("category").cat.codes,
        "Next_Day": actuals["Service_Level"].eq("Next Day").astype(int),
    }).fillna(0)
    y = actuals["Late_Actual"].astype(int)
    if y.nunique() < 2 or y.value_counts().min() < 3:
        return np.full(len(X), y.mean()), pd.Series(dtype=float)
    clf = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42, n_jobs=-1,
                                 class_weight="balanced_subsample")
    folds = StratifiedKFold(n_splits=min(5, int(y.value_counts().min())), shuffle=True, random_state=42)
    prob = cross_val_predict(clf, X, y, cv=folds, method="predict_proba")[:, 1]
    clf.fit(X, y)
    return prob, pd.Series(clf.feature_importances_, index=X.columns).sort_values()


def ground_reach_miles(fac, p):
    hours = 24 + p["ofd_hour"] - fac["Linehaul_Departure_Hour"] - p["sort_hours"]
    return (hours.clip(lower=0) * p["ground_speed"] / p["circuity"]).round(0)


# ─────────────────────────────────────────────────────────────────────────────
# Pages
# ─────────────────────────────────────────────────────────────────────────────
def ctx():
    return st.session_state["ctx"]


def page_overview():
    c = ctx(); plan, base, p = c["plan"], c["base"], c["p"]
    header("Executive Overview", "Network-wide service, cost and capacity for today's planning cycle")
    one_day_lens([
        "One-day delivery is a <b>clock</b>: every order must clear <b>cutoff → pick/pack → linehaul or flight → hub sort → delivery station → last mile</b> before its promise time.",
        "The optimizer picks the node and mode for every order so the clock is met at the <b>lowest landed cost</b>, within facility and air-lift capacity.",
        "The headline number is <b>Next-Day promise attainment</b>; cost per package and air mix show what it costs to hold it.",
    ])
    f = plan[plan["Assigned_Node"] != UNFULFILLED]
    nd = plan[plan["Service_Level"] == "Next Day"]
    nd_ot = nd["On_Time_Planned"].mean()
    ot = plan["On_Time_Planned"].mean()
    cpp = f["Cost_to_Serve"].mean()
    air_mix = f["Mode"].eq("Air").mean()
    util = c["util"]
    net_util = (util["Assigned"].sum() / util["Capacity"].sum()) if util["Capacity"].sum() else 0
    kpis([
        ("Next-Day promise attainment", pct(nd_ot), f"Legacy rule: {pct(base['nd_on_time'])}", "ok" if nd_ot >= .95 else "warn" if nd_ot >= .9 else "bad"),
        ("All-service on-time (plan)", pct(ot), f"Legacy rule: {pct(base['on_time'])}", "ok" if ot >= .97 else "warn"),
        ("Landed cost / package", f"${cpp + plan['Late_Penalty'].mean():.2f}", f"Legacy rule: ${base['cost'] + base['penalty']:.2f} (incl. penalties)", "ok"),
        ("Air mix", pct(air_mix), f"{int(f['Mode'].eq('Air').sum()):,} of {p['air_lift']:,} lift", "warn" if air_mix > .2 else "ok"),
        ("Network utilization", pct(net_util), f"Legacy overload: {base['overload']:,} pkgs", "warn" if net_util > .9 else "ok"),
        ("Unfulfilled orders", f"{int((plan['Assigned_Node'] == UNFULFILLED).sum()):,}", f"of {len(plan):,} orders", "bad" if (plan['Assigned_Node'] == UNFULFILLED).any() else "ok"),
    ])
    l, r = st.columns(2)
    svc = plan.groupby("Service_Level").agg(Orders=("Order_ID", "count"), On_Time=("On_Time_Planned", "mean")).reindex(list(SERVICE_PROMISE_DAYS)).reset_index()
    fig = px.bar(svc, x="Service_Level", y="On_Time", text=svc["On_Time"].map(pct), title="Planned promise attainment by service level", color_discrete_sequence=[TEAL])
    fig.add_hline(y=0.95, line_dash="dot", line_color=AMBER, annotation_text="95% target")
    fig.update_yaxes(tickformat=".0%", range=[0, 1.05], title=""); fig.update_xaxes(title="")
    l.plotly_chart(style_fig(fig), width="stretch")

    misses = plan[plan["Miss_Reason"] != "On time"]["Miss_Reason"].value_counts().reset_index()
    misses.columns = ["Reason", "Orders"]
    if misses.empty:
        r.success("Every order is planned on time.")
    else:
        fig = px.bar(misses, y="Reason", x="Orders", orientation="h", title="Where one-day breaks — planned misses by root cause", color_discrete_sequence=[RED])
        fig.update_yaxes(title="", categoryorder="total ascending")
        r.plotly_chart(style_fig(fig), width="stretch")

    comp = pd.DataFrame({
        "Component": ["Transport", "Handling + packaging", "Late penalties"],
        "Optimized": [f["Transport_Cost"].mean(), f["Handling_Cost"].mean(), plan["Late_Penalty"].mean()],
    })
    fig = px.bar(comp, x="Component", y="Optimized", text=comp["Optimized"].map(lambda v: f"${v:.2f}"), title="Landed cost per package — components", color_discrete_sequence=[NAVY])
    fig.update_yaxes(title="USD / package"); fig.update_xaxes(title="")
    st.plotly_chart(style_fig(fig, 330), width="stretch")
    st.markdown("<div class='foot'>Legacy rule = ship from nearest stocking node by ground, ignoring capacity. Volumes are a scaled sample day.</div>", unsafe_allow_html=True)


def page_playbook():
    c = ctx(); plan, fac, p = c["plan"], c["fac"], c["p"]
    header("One-Day Delivery Playbook", "The end-to-end clock, the lever at each step, and today's live reading")
    one_day_lens([
        "One-day is won or lost in <b>hours</b>. Each step below has a time budget; overrun any one and the package misses its van.",
        "Ground one-day works inside a <b>reach radius</b> of each node. Outside it, the network either <b>flies</b> the package or <b>positions inventory closer</b>.",
        "Use this page to brief operations: what we do at each step and which metric proves it is working.",
    ])
    f = plan[plan["Assigned_Node"] != UNFULFILLED]
    nd = plan[plan["Service_Level"] == "Next Day"]
    before_cut = (f["Ship_Day"] == 0).mean()
    window = (fac["Linehaul_Departure_Hour"] - fac["Order_Cutoff_Hour"]).mean()
    reach = ground_reach_miles(fac, p)
    steps = [
        ("1", "Promise at checkout", f"Customer local, before {p['promise_cutoff']:.0f}:00",
         "Show 'arrives tomorrow' only when a node with stock can make the clock. Promise cutoff sets demand for tomorrow.",
         "Promise cutoff (sidebar)", f"{pct(nd['On_Time_Planned'].mean())} of Next-Day orders planned on time"),
        ("2", "Order release & node selection", "Continuous, every few minutes",
         "Optimizer sources each order from the node that meets the promise at lowest landed cost, inside capacity.",
         "Fulfillment Optimizer page", f"{f['Assigned_Node'].nunique()} nodes active; {pct(f['Mode'].eq('Air').mean())} on air"),
        ("3", "Order cutoff → pick / pack", f"Avg cutoff {fac['Order_Cutoff_Hour'].mean():.1f}h local",
         "Waves are timed so the last order before cutoff is packed before the trailer or air tender closes.",
         "Cutoffs & Time-in-Transit page", f"{pct(before_cut)} of orders make today's cutoff; avg pack window {window:.1f} h"),
        ("4a", "Ground linehaul", f"Departs ~{fac['Linehaul_Departure_Hour'].mean():.0f}:00",
         f"Trailers run overnight at ~{p['ground_speed']:.0f} mph effective. One-day ground reach ≈ {reach.median():.0f} mi from each node.",
         "Ground speed, sort hours (sidebar)", f"{pct((f['Mode'] == 'Ground').mean())} of packages on ground"),
        ("4b", "Next-day air", f"Gateway tender by {p['air_tender_hour']:.1f}h local",
         "Packages outside ground reach are tendered to the gateway, flown to the air hub, sorted overnight and flown to the destination gateway.",
         "Air lift capacity (sidebar)", f"{int(f['Mode'].eq('Air').sum()):,} air pkgs vs {p['air_lift']:,} lift"),
        ("5", "Hub / delivery station sort", f"{p['sort_hours']:.1f} h per sort",
         "Inbound trailers and air cans are sorted to delivery routes. Station must be loaded before out-for-delivery.",
         "Sort hours (sidebar)", f"Out-for-delivery at {p['ofd_hour']:.0f}:00 local"),
        ("6", "Last-mile delivery", "Next day by 21:00",
         "Routes are densified so the extra one-day volume rides existing stops rather than new vans.",
         "Risk & Exceptions page", f"{int((plan['Miss_Reason'] != 'On time').sum()):,} planned misses to act on"),
    ]
    for n, title, clock, what, lever, live in steps:
        st.markdown(
            f"<div class='step'><div class='n'>STEP {n} · {clock}</div><b>{title}</b><br>{what}"
            f"<br><span class='foot'>Lever: {lever} &nbsp;|&nbsp; Today: <b>{live}</b></span></div>",
            unsafe_allow_html=True,
        )

    st.subheader("Clock for a typical next-day package")
    day0 = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
    dep = fac["Linehaul_Departure_Hour"].median()
    g = [("Order placed", 15, 16), ("Pick / pack wave", 16, fac["Order_Cutoff_Hour"].median() + 1),
         ("Ground linehaul", dep, dep + 4), ("Hub sort", dep + 4, dep + 4 + p["sort_hours"]),
         ("Station load", 24 + p["ofd_hour"] - 2, 24 + p["ofd_hour"]), ("Last-mile delivery", 24 + p["ofd_hour"], 24 + 19)]
    a = [("Order placed", 15, 16), ("Pick / pack wave", 16, fac["Order_Cutoff_Hour"].median() + 1),
         ("Gateway tender", fac["Order_Cutoff_Hour"].median() + 1, p["air_tender_hour"]),
         ("Flight to air hub", p["air_tender_hour"] + .5, 25.5), ("Air hub sort", 25.5, 28.5),
         ("Outbound flight", 28.5, 30.5), ("Last-mile delivery", 32, 34.5)]
    rows = [dict(Lane="Ground (within reach)", Step=s, Start=day0 + pd.Timedelta(hours=b), End=day0 + pd.Timedelta(hours=e)) for s, b, e in g]
    rows += [dict(Lane="Next-day air (beyond reach)", Step=s, Start=day0 + pd.Timedelta(hours=b), End=day0 + pd.Timedelta(hours=e)) for s, b, e in a]
    fig = px.timeline(pd.DataFrame(rows), x_start="Start", x_end="End", y="Lane", color="Step", title="Hours from order to doorstep (illustrative, node local time)")
    fig.add_vline(x=day0 + pd.Timedelta(hours=24), line_dash="dot", line_color=GREY)
    st.plotly_chart(style_fig(fig, 320), width="stretch")


def page_network():
    c = ctx(); plan, fac, p = c["plan"], c["fac"], c["p"]
    header("Network Map", "Nodes, demand, one-day ground reach and today's sourcing lanes")
    one_day_lens([
        "Shaded circles are each node's <b>one-day ground reach</b>. Demand inside a circle can go next day by truck, the cheapest way.",
        "Demand outside every circle needs <b>air</b> or a <b>new forward stocking location</b>. This map is where network design starts.",
    ])
    reach = ground_reach_miles(fac, p)
    dm = plan.groupby(["Dest_Market", "Dest_Region"]).agg(Lat=("Dest_Lat", "mean"), Lon=("Dest_Lon", "mean"), Orders=("Order_ID", "count")).reset_index()
    dist = haversine(dm["Lat"], dm["Lon"], fac["Lat"], fac["Lon"])
    dm["Within_Reach"] = (dist <= reach.to_numpy()[None, :]).any(axis=1)
    covered = dm.loc[dm["Within_Reach"], "Orders"].sum() / dm["Orders"].sum()
    kpis([("Demand inside ground reach", pct(covered), "share of today's orders", "ok" if covered > .8 else "warn"),
          ("Median ground reach", f"{reach.median():.0f} mi", "from linehaul departure", "ok"),
          ("Active nodes", f"{len(fac)}", f"{(fac['Type'].str.contains('Forward')).sum()} forward stocking", "ok")])

    show_routes = st.slider("Sampled lanes to draw", 0, 400, 150, 25)
    fig = go.Figure()
    for (_, row), rmi in zip(fac.iterrows(), reach):
        t = np.linspace(0, 2 * np.pi, 60)
        fig.add_trace(go.Scattermap(lat=row["Lat"] + (rmi / 69) * np.sin(t), lon=row["Lon"] + (rmi / (69 * np.cos(np.radians(row["Lat"])))) * np.cos(t),
                                    mode="lines", fill="toself", fillcolor="rgba(14,124,134,0.08)", line=dict(width=1, color="rgba(14,124,134,0.4)"),
                                    hoverinfo="skip", showlegend=False))
    f = plan[plan["Assigned_Node"] != UNFULFILLED]
    if show_routes:
        s = f.sample(min(show_routes, len(f)), random_state=1).merge(fac[["Facility_ID", "Lat", "Lon"]], left_on="Assigned_Node", right_on="Facility_ID")
        for mode, color in [("Ground", TEAL), ("Air", AMBER)]:
            sm = s[s["Mode"] == mode]
            lat, lon = [], []
            for rr in sm.itertuples():
                lat += [rr.Lat, rr.Dest_Lat, None]; lon += [rr.Lon, rr.Dest_Lon, None]
            fig.add_trace(go.Scattermap(lat=lat, lon=lon, mode="lines", line=dict(width=1, color=color), opacity=.5, name=f"{mode} lanes", hoverinfo="skip"))
    fig.add_trace(go.Scattermap(lat=dm["Lat"], lon=dm["Lon"], mode="markers", name="Demand",
                                marker=dict(size=np.sqrt(dm["Orders"]) * 1.6, color=np.where(dm["Within_Reach"], NAVY, RED), opacity=.75),
                                text=dm["Dest_Market"] + "<br>" + dm["Orders"].astype(str) + " orders", hoverinfo="text"))
    fig.add_trace(go.Scattermap(lat=fac["Lat"], lon=fac["Lon"], mode="markers+text", name="Nodes", text=fac["Facility_ID"], textposition="top right",
                                marker=dict(size=13, color="black"), hovertext=fac["Name"] + " · " + fac["Type"], hoverinfo="text"))
    fig.update_layout(map=dict(style="carto-positron", center=dict(lat=38.5, lon=-96), zoom=3.1), height=620, margin=dict(l=0, r=0, t=0, b=0),
                      legend=dict(orientation="h", y=0.02, x=0.01, bgcolor="rgba(255,255,255,.8)"))
    st.plotly_chart(fig, width="stretch")
    st.markdown("<div class='foot'>Red demand bubbles sit outside every ground reach circle. Lanes are straight-line node→customer, not road routes.</div>", unsafe_allow_html=True)


def page_sourcing():
    c = ctx(); plan, fac = c["plan"], c["fac"]
    header("Fulfillment Optimizer", "Order sourcing and mode selection — the plan the network executes")
    one_day_lens([
        "For each order the LP compares every node that holds stock × {ground, air} and picks the cheapest option that <b>meets the promise</b>.",
        "Late options carry a <b>penalty</b>, so the model only accepts a miss when no on-time option fits in capacity or the upgrade costs more than the penalty.",
        "Download the plan to feed WMS release or carrier manifesting.",
    ])
    f = plan[plan["Assigned_Node"] != UNFULFILLED]
    flow = f.groupby(["Assigned_Node", "Dest_Region", "Mode"]).size().reset_index(name="Orders")
    nodes = list(fac["Facility_ID"]) + sorted(plan["Dest_Region"].unique())
    idx = {n: i for i, n in enumerate(nodes)}
    fig = go.Figure(go.Sankey(
        node=dict(label=nodes, pad=12, thickness=14, color=[NAVY] * len(fac) + [TEAL] * (len(nodes) - len(fac))),
        link=dict(source=flow["Assigned_Node"].map(idx), target=flow["Dest_Region"].map(idx), value=flow["Orders"],
                  color=np.where(flow["Mode"] == "Air", "rgba(224,159,31,.45)", "rgba(14,124,134,.30)"))))
    fig.update_layout(title="Node → destination region (amber = air, teal = ground)")
    st.plotly_chart(style_fig(fig, 520), width="stretch")

    mix = f.groupby(["Dest_Market", "Mode"]).size().reset_index(name="Orders")
    fig = px.bar(mix, x="Dest_Market", y="Orders", color="Mode", color_discrete_map=MODE_COLORS, title="Mode mix by destination market")
    fig.update_xaxes(title="", tickangle=-35)
    st.plotly_chart(style_fig(fig, 380), width="stretch")

    st.subheader("Order-level plan")
    a, b, cc = st.columns(3)
    nodes_sel = a.multiselect("Node", sorted(plan["Assigned_Node"].unique()))
    svc_sel = b.multiselect("Service level", list(SERVICE_PROMISE_DAYS))
    late_only = cc.toggle("Planned misses only")
    view = plan
    if nodes_sel: view = view[view["Assigned_Node"].isin(nodes_sel)]
    if svc_sel: view = view[view["Service_Level"].isin(svc_sel)]
    if late_only: view = view[view["Miss_Reason"] != "On time"]
    cols = ["Order_ID", "Dest_Market", "Service_Level", "SKU_Category", "Assigned_Node", "Mode", "Zone", "Distance_Mi",
            "Promise_Day", "Delivery_Day", "Cost_to_Serve", "Late_Penalty", "Contribution", "Miss_Reason"]
    st.dataframe(view[cols], width="stretch", hide_index=True, height=360)
    st.download_button("Download fulfillment plan (CSV)", plan.to_csv(index=False).encode(), "fulfillment_plan.csv", "text/csv")


def page_cutoffs():
    c = ctx(); plan, fac, p, orders = c["plan"], c["fac"], c["p"], c["orders"]
    header("Cutoffs & Time-in-Transit", "How late we can take an order and still deliver tomorrow")
    one_day_lens([
        "The <b>order cutoff</b> is the single biggest lever for one-day volume. Every hour later captures more demand, but shrinks the pick/pack window.",
        "The sensitivity below re-runs the transit clock with all node cutoffs shifted, holding trailer and flight times fixed.",
    ])
    hourly = orders.assign(Hour=orders["Order_Hour_ET"].astype(int)).groupby(["Hour", "Service_Level"]).size().reset_index(name="Orders")
    fig = px.bar(hourly, x="Hour", y="Orders", color="Service_Level", title="Order arrival curve (Eastern time)",
                 color_discrete_sequence=[NAVY, TEAL, "#9BB4C4"])
    for h in sorted(fac["Order_Cutoff_Hour"].unique()):
        fig.add_vline(x=h, line_dash="dot", line_color=AMBER)
    fig.update_xaxes(dtick=2, title="Hour (ET)")
    st.plotly_chart(style_fig(fig, 340), width="stretch")

    tbl = fac[["Facility_ID", "City", "Order_Cutoff_Hour", "Linehaul_Departure_Hour"]].copy()
    tbl["Pick_Pack_Window_Hrs"] = tbl["Linehaul_Departure_Hour"] - tbl["Order_Cutoff_Hour"]
    tbl["Ground_Reach_Mi"] = ground_reach_miles(fac, p).to_numpy()
    tbl["Orders_Assigned"] = tbl["Facility_ID"].map(plan["Assigned_Node"].value_counts()).fillna(0).astype(int)
    st.dataframe(tbl, width="stretch", hide_index=True)

    st.subheader("Cutoff sensitivity — Next-Day orders with an on-time option")
    nd = orders[orders["Service_Level"] == "Next Day"].reset_index(drop=True)
    rows = []
    for shift in [-2, -1, 0, 1, 2]:
        o = compute_options(nd, fac, {**p, "cutoff_shift": shift})
        g_ok = ((o["g_late"] == 0) & o["in_stock"]).any(axis=1).mean()
        a_ok = (((o["g_late"] == 0) | ((o["a_late"] == 0) & p["air_enabled"])) & o["in_stock"]).any(axis=1).mean()
        rows += [dict(Shift=shift, Option="Ground only", Share=g_ok), dict(Shift=shift, Option="Ground + air", Share=a_ok)]
    sens = pd.DataFrame(rows)
    fig = px.line(sens, x="Shift", y="Share", color="Option", markers=True, color_discrete_map={"Ground only": TEAL, "Ground + air": AMBER},
                  title="Shift all node cutoffs by N hours")
    fig.update_yaxes(tickformat=".0%", title="Next-Day orders feasible"); fig.update_xaxes(dtick=1, title="Cutoff shift (hours)")
    st.plotly_chart(style_fig(fig, 340), width="stretch")
    if (tbl["Pick_Pack_Window_Hrs"] + 1 < 1.5).any():
        st.warning("A +1h shift leaves some nodes under 1.5 h to pick and pack. Add labor or move the trailer before committing.")

    z = plan[plan["Assigned_Node"] != UNFULFILLED].groupby(["Zone", "Mode"]).size().reset_index(name="Orders")
    fig = px.bar(z, x="Zone", y="Orders", color="Mode", color_discrete_map=MODE_COLORS, title="Zone distribution by mode — lower zones = closer = cheaper")
    st.plotly_chart(style_fig(fig, 320), width="stretch")


def page_capacity():
    c = ctx(); plan, fac, p, util, orders = c["plan"], c["fac"], c["p"], c["util"], c["orders"]
    header("Capacity & Inventory Positioning", "Where volume lands, where it is tight, and where to stock next")
    one_day_lens([
        "A node above ~90% utilization starts missing trailer departures. That spill is what pushes one-day orders onto air or late.",
        "Inventory placement decides reach: a long-tail item stocked in only 3 nodes cannot go next-day by ground to most of the country.",
        "Use the what-if to test forward-positioning a category before moving stock.",
    ])
    u = util.reset_index()
    fig = px.bar(u, x="Facility_ID", y="Utilization", text=u["Utilization"].map(pct), title="Outbound capacity utilization",
                 color=np.where(u["Utilization"] > .95, "Critical", np.where(u["Utilization"] > .85, "Watch", "Healthy")),
                 color_discrete_map={"Healthy": TEAL, "Watch": AMBER, "Critical": RED})
    fig.add_hline(y=.85, line_dash="dot", line_color=AMBER); fig.add_hline(y=.95, line_dash="dot", line_color=RED)
    fig.update_yaxes(tickformat=".0%", range=[0, 1.1], title=""); fig.update_xaxes(title="")
    st.plotly_chart(style_fig(fig, 360), width="stretch")
    air_used = int(plan["Mode"].eq("Air").sum())
    kpis([("Air lift used", f"{air_used:,} / {p['air_lift']:,}", pct(air_used / max(p['air_lift'], 1)), "warn" if air_used > .9 * p["air_lift"] else "ok"),
          ("Nodes ≥ 95%", f"{int((util['Utilization'] >= .95).sum())}", "at risk of missing departure", "bad" if (util['Utilization'] >= .95).any() else "ok")])

    st.subheader("Next-day ground coverage by SKU category and region")
    o = c["opts"]
    g_ok = ((o["g_late"] == 0) & o["in_stock"]).any(axis=1)
    cov = orders.assign(Ground_ND=g_ok).groupby(["SKU_Category", "Dest_Region"])["Ground_ND"].mean().unstack()
    fig = px.imshow(cov, text_auto=".0%", color_continuous_scale=["#F3D9D5", "#FFFFFF", TEAL], zmin=0, zmax=1, aspect="auto",
                    title="Share of orders a stocking node can deliver on time by ground")
    st.plotly_chart(style_fig(fig, 300), width="stretch")

    st.subheader("What-if: forward-position a category")
    a, b = st.columns(2)
    cat = a.selectbox("Category", list(CATEGORY_MIX), index=2)
    cand = fac.loc[~fac["Stocked_Categories"].str.contains(cat), "Facility_ID"].tolist()
    if not cand:
        st.info(f"Category {cat} is already stocked at every node.")
        return
    node = b.selectbox("Add stock at", cand)
    sub = orders[orders["SKU_Category"] == cat].reset_index(drop=True)
    before = ((compute_options(sub, fac, p)["g_late"] == 0) & compute_options(sub, fac, p)["in_stock"]).any(axis=1).mean()
    fac2 = fac.copy()
    fac2.loc[fac2["Facility_ID"] == node, "Stocked_Categories"] += f",{cat}"
    o2 = compute_options(sub, fac2, p)
    after = ((o2["g_late"] == 0) & o2["in_stock"]).any(axis=1).mean()
    kpis([(f"Category {cat} ground one-day — today", pct(before), f"{len(sub):,} orders", "warn"),
          (f"After stocking at {node}", pct(after), f"+{100 * (after - before):.1f} pts", "ok" if after > before else "warn")])


def page_cost():
    c = ctx(); plan, base = c["plan"], c["base"]
    header("Cost-to-Serve & Profitability", "Landed cost per package and contribution by lane")
    one_day_lens([
        "One-day is only sustainable if it is <b>profitable per order</b>. Air protects the promise but costs ~3× ground.",
        "Most cost is set by <b>zone</b>: shipping from a closer node is the cheapest way to deliver faster.",
    ])
    f = plan[plan["Assigned_Node"] != UNFULFILLED]
    loss = f["Contribution"] < 0
    kpis([("Cost to serve / package", f"${f['Cost_to_Serve'].mean():.2f}", f"Legacy: ${base['cost']:.2f}", "ok"),
          ("Late penalties / package", f"${plan['Late_Penalty'].mean():.2f}", f"Legacy: ${base['penalty']:.2f}", "ok"),
          ("Loss-making orders", f"{int(loss.sum()):,}", pct(loss.mean()), "warn" if loss.mean() > .05 else "ok"),
          ("Avg contribution / order", f"${f['Contribution'].mean():.2f}", "margin − cost − penalty", "ok")])
    l, r = st.columns(2)
    z = f.groupby(["Zone", "Mode"])["Cost_to_Serve"].mean().reset_index()
    fig = px.bar(z, x="Zone", y="Cost_to_Serve", color="Mode", barmode="group", color_discrete_map=MODE_COLORS, title="Cost to serve by zone and mode")
    fig.update_yaxes(title="USD / package")
    l.plotly_chart(style_fig(fig), width="stretch")
    m = f.groupby("Dest_Market").agg(Cost=("Cost_to_Serve", "mean"), Margin=("Gross_Margin", "mean"), Orders=("Order_ID", "count")).reset_index()
    fig = px.scatter(m, x="Cost", y="Margin", size="Orders", text="Dest_Market", title="Market economics — margin vs. cost per order", color_discrete_sequence=[NAVY])
    fig.update_traces(textposition="top center", textfont_size=9)
    lim = max(m["Cost"].max(), m["Margin"].max()) * 1.1
    fig.add_shape(type="line", x0=0, y0=0, x1=lim, y1=lim, line=dict(dash="dot", color=RED))
    fig.update_xaxes(title="Cost to serve ($)"); fig.update_yaxes(title="Gross margin ($)")
    r.plotly_chart(style_fig(fig), width="stretch")
    st.subheader("Loss-making orders")
    st.dataframe(f[loss].sort_values("Contribution")[["Order_ID", "Dest_Market", "Service_Level", "Assigned_Node", "Mode", "Zone",
                                                     "Order_Value", "Gross_Margin", "Cost_to_Serve", "Late_Penalty", "Contribution"]],
                 width="stretch", hide_index=True, height=300)


def page_risk():
    c = ctx(); act, imp = c["actuals"], c["importance"]
    header("SLA Risk & Exceptions", "Predict which planned on-time packages will still miss, and act before the van leaves")
    one_day_lens([
        "A plan that looks on time can still miss from <b>weather, hub congestion or thin slack</b>. The model scores every package's miss risk.",
        "Act at the right step: <b>upgrade to air</b> before tender cutoff, <b>re-source</b> away from a congested node, or <b>notify the customer</b> early.",
    ])
    thr = st.slider("Risk threshold", 0.0, 1.0, 0.5, 0.05)
    hr = act[act["Risk"] >= thr].copy()

    def action(r):
        if r["Mode"] == "Ground" and r["Slack_Hrs"] < 8:
            return "Upgrade to air before tender cutoff"
        if r["Node_Utilization"] > 0.92:
            return "Re-source to alternate node"
        if r["Dest_Region"] in c["disrupted"]:
            return "Proactive delay notification"
        return "Monitor at station"
    hr["Recommended_Action"] = hr.apply(action, axis=1) if not hr.empty else []
    kpis([("Simulated actual on-time", pct(1 - act["Late_Actual"].mean()), "after execution noise", "ok" if act["Late_Actual"].mean() < .05 else "warn"),
          ("Packages above threshold", f"{len(hr):,}", pct(len(hr) / max(len(act), 1)), "warn" if len(hr) else "ok"),
          ("Next-Day at risk", f"{int((hr['Service_Level'] == 'Next Day').sum()):,}", "highest penalty", "bad" if (hr['Service_Level'] == 'Next Day').any() else "ok")])
    l, r = st.columns(2)
    reg = act.groupby("Dest_Region")["Risk"].mean().sort_values().reset_index()
    fig = px.bar(reg, y="Dest_Region", x="Risk", orientation="h", title="Average predicted miss risk by region", color_discrete_sequence=[AMBER])
    fig.update_xaxes(tickformat=".0%"); fig.update_yaxes(title="")
    l.plotly_chart(style_fig(fig), width="stretch")
    if not imp.empty:
        fig = px.bar(imp.reset_index(), y="index", x=0, orientation="h", title="What drives misses (feature importance)", color_discrete_sequence=[NAVY])
        fig.update_yaxes(title=""); fig.update_xaxes(title="")
        r.plotly_chart(style_fig(fig), width="stretch")
    if not hr.empty:
        st.subheader("Exception queue")
        st.dataframe(hr.sort_values("Risk", ascending=False)[["Order_ID", "Dest_Market", "Service_Level", "Assigned_Node", "Mode", "Slack_Hrs",
                                                             "Node_Utilization", "Risk", "Recommended_Action"]].round(2),
                     width="stretch", hide_index=True, height=320)
        st.download_button("Download exception queue (CSV)", hr.to_csv(index=False).encode(), "exception_queue.csv", "text/csv")
    st.markdown("<div class='foot'>Risk = out-of-fold Random Forest probability trained on the simulated execution day. Retrain on historical scan data before operational use.</div>", unsafe_allow_html=True)


def kpi_summary():
    c = ctx(); plan, util, base, p = c["plan"], c["util"], c["base"], c["p"]
    f = plan[plan["Assigned_Node"] != UNFULFILLED]
    return {
        "orders": len(plan),
        "next_day_attainment": round(plan.loc[plan["Service_Level"] == "Next Day", "On_Time_Planned"].mean(), 4),
        "legacy_next_day_attainment": round(base["nd_on_time"], 4),
        "cost_per_package": round(f["Cost_to_Serve"].mean(), 2),
        "legacy_cost_per_package": round(base["cost"], 2),
        "air_mix": round(f["Mode"].eq("Air").mean(), 4),
        "air_lift_capacity": p["air_lift"],
        "unfulfilled": int((plan["Assigned_Node"] == UNFULFILLED).sum()),
        "miss_reasons": plan["Miss_Reason"].value_counts().to_dict(),
        "node_utilization": util["Utilization"].round(3).to_dict(),
        "disrupted_regions": c["disrupted"],
        "simulated_actual_late_rate": round(float(c["actuals"]["Late_Actual"].mean()), 4),
        "assumptions": {k: p[k] for k in ["ground_speed", "sort_hours", "ofd_hour", "promise_cutoff", "air_tender_hour", "cap_mult", "fuel_pct"]},
    }


def ask_gemini(api_key, model, question, summary):
    prompt = (
        "You are a senior parcel network analyst at a national carrier. Using ONLY the KPI summary below, "
        "answer the question concisely for an operations leader. Tie every recommendation to a step of the "
        "one-day clock (cutoff, pick/pack, linehaul/air, hub sort, last mile) and quantify it from the data. "
        "If the data cannot support an answer, say so.\n\n"
        f"KPI SUMMARY (JSON):\n{summary}\n\nQUESTION: {question}"
    )
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        json={"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.3}},
        timeout=60,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Gemini API error {r.status_code}: {r.text[:300]}")
    parts = r.json().get("candidates", [{}])[0].get("content", {}).get("parts", [])
    return "".join(pt.get("text", "") for pt in parts) or "No answer returned."


def page_ai():
    header("AI Network Analyst", "Ask questions about today's plan in plain English (optional, Gemini)")
    one_day_lens([
        "Turns the plan into a briefing: <b>why</b> next-day attainment is where it is, and <b>which lever</b> to pull first.",
        "Privacy: only the aggregated KPI summary below is sent. No order IDs, addresses or customer data leave the app.",
    ])
    key = st.session_state.get("gemini_key") or os.environ.get("GEMINI_API_KEY", "")
    try:
        key = key or st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        pass
    a, b = st.columns([2, 1])
    typed = a.text_input("Gemini API key", type="password", value="" if key else "", placeholder="Uses GEMINI_API_KEY if set" if key else "Paste key (kept only for this session)")
    if typed:
        st.session_state["gemini_key"] = typed; key = typed
    model = b.text_input("Model", value=st.session_state.get("gemini_model", "gemini-2.5-flash"))
    st.session_state["gemini_model"] = model
    summary = kpi_summary()
    with st.expander("Data sent to the model"):
        st.json(summary)
    presets = ["Why are Next-Day orders missing, and what is the first fix?",
               "Which node should get more capacity or labor tonight?",
               "Is our air mix too high? How do we reduce it without losing one-day?",
               "Write a 5-bullet morning briefing for the network operations call."]
    q = st.selectbox("Suggested questions", presets)
    q = st.text_area("Your question", value=q, height=80)
    if st.button("Ask analyst", type="primary", disabled=not key):
        try:
            with st.spinner("Analyzing…"):
                st.markdown(ask_gemini(key, model, q, summary))
        except Exception as e:
            st.error(str(e))
    if not key:
        st.info("Add a Gemini API key to enable this page. The rest of the tool works without it.")


def page_data():
    c = ctx(); p = c["p"]
    header("Data & Assumptions", "Load your own orders and network, review the rate card and methodology")
    one_day_lens(["Every one-day number in this tool comes from the assumptions on this page. Replace them with your network's actuals to make the plan operational."])
    t1, t2, t3, t4 = st.tabs(["Orders", "Facilities", "Rate card & assumptions", "Methodology & glossary"])
    with t1:
        st.write("Upload a CSV of orders to replace the simulated sample day. Required columns:")
        st.code(", ".join(ORDER_COLUMNS))
        st.download_button("Download template (simulated sample)", simulate_orders(200).to_csv(index=False).encode(), "orders_template.csv", "text/csv")
        up = st.file_uploader("Orders CSV", type="csv")
        if up is not None:
            df, err = validate_orders(pd.read_csv(up))
            if err:
                st.error(err)
            else:
                st.session_state["uploaded_orders"] = df
                st.success(f"Loaded {len(df):,} orders. All pages now use your data.")
        if "uploaded_orders" in st.session_state and st.button("Revert to simulated orders"):
            del st.session_state["uploaded_orders"]; st.rerun()
    with t2:
        st.write("Edit nodes, capacity, stocked categories (comma-separated A,B,C), cutoffs and departures. Types containing 'Air Hub' act as the air sort hub.")
        edited = st.data_editor(st.session_state["facilities"], num_rows="dynamic", width="stretch", hide_index=True, key="fac_editor")
        if st.button("Apply network changes", type="primary"):
            st.session_state["facilities"] = edited.dropna(subset=["Facility_ID", "Lat", "Lon"]).reset_index(drop=True); st.rerun()
        if st.button("Reset to default network"):
            st.session_state["facilities"] = DEFAULT_FACILITIES.copy(); st.rerun()
    with t3:
        rc = pd.DataFrame({"Zone": ZONES, "Ground base $": GROUND_BASE, "Ground $/lb": GROUND_PER_LB, "Air base $": AIR_BASE, "Air $/lb": AIR_PER_LB,
                           "Miles up to": [str(b) for b in ZONE_BREAKS] + ["1800+"]})
        st.dataframe(rc, width="stretch", hide_index=True)
        st.caption(f"Billable weight = max(actual, L×W×H / {DIM_DIVISOR}), rounded up. Fuel surcharge {pct(p['fuel_pct'])}, residential fee ${p['resi_fee']:.2f}, rate multiplier {p['rate_mult']:.2f}.")
        st.json({k: v for k, v in p.items()})
    with t4:
        st.markdown("""
**Optimization.** Linear program (HiGHS). Decision variables: one per order × stocking node × mode, plus an "unfulfilled" option.
Objective: minimise transport + handling + packaging + (days late × penalty). Constraints: every order assigned once; node outbound
capacity; nightly air-lift capacity. Fractional LP splits are rounded to the highest-weight option.

**Time-in-transit.** Ship day from node cutoff (node local time). Ground arrival = departure + road miles / effective speed +
sort hours × (1 + one extra sort per 600 mi) + time-zone shift; delivered the first day the station's out-for-delivery time is met.
Air: tender at gateway by the earlier of the tender hour and the hub inbound deadline less flight time; delivered the day after tender.

**Promise.** Promise day = service days + 1 if ordered after the checkout promise cutoff (customer local).

**Glossary.** *Zone* — distance band that sets the rate. *Cutoff* — last order time for today's departure. *Linehaul* — node-to-hub trailer
move. *Air lift* — nightly aircraft capacity. *FSL* — forward stocking location near dense demand. *OFD* — out for delivery.
*Cost to serve* — landed cost per package excluding penalties. *Slack* — hours between planned delivery and promise deadline.
""")


# ─────────────────────────────────────────────────────────────────────────────
# App shell
# ─────────────────────────────────────────────────────────────────────────────
def sidebar_params():
    sb = st.sidebar
    sb.markdown(f"### {APP_TITLE}")
    src = "Uploaded orders" if "uploaded_orders" in st.session_state else "Simulated sample day"
    sb.caption(f"Data: **{src}**")
    with sb.expander("Volume & capacity", expanded=True):
        n = st.slider("Orders (simulated)", 500, 5000, 3000, 250, disabled="uploaded_orders" in st.session_state)
        cap_mult = st.slider("Node capacity multiplier", 0.5, 1.5, 1.0, 0.05)
        air_lift = st.slider("Nightly air lift (packages)", 0, 1500, 450, 50)
        air_enabled = st.toggle("Next-day air available", True)
        fsl = st.toggle("Use forward stocking locations", True)
    with sb.expander("One-day clock"):
        promise_cutoff = st.slider("Checkout promise cutoff (customer local)", 12.0, 23.0, 19.0, 0.5)
        ground_speed = st.slider("Effective ground speed (mph)", 35, 65, 52)
        sort_hours = st.slider("Hours per hub sort", 1.0, 4.0, 2.0, 0.5)
        ofd_hour = st.slider("Out-for-delivery (local)", 6.0, 10.0, 8.0, 0.5)
        air_tender = st.slider("Air gateway tender cutoff (local)", 19.0, 23.5, 22.5, 0.5)
    with sb.expander("Cost"):
        fuel = st.slider("Fuel surcharge", 0.0, 0.35, 0.18, 0.01)
        rate_mult = st.slider("Rate card multiplier", 0.6, 1.4, 1.0, 0.05)
        packaging = st.slider("Packaging $/order", 0.0, 3.0, 0.75, 0.25)
        penalty_mult = st.slider("Late-penalty multiplier", 0.0, 3.0, 1.0, 0.25)
        unf_cost = st.slider("Unfulfilled order cost ($)", 25, 300, 120, 5)
        margin = st.slider("Gross margin %", 0.10, 0.60, 0.35, 0.05)
    with sb.expander("Disruptions"):
        disrupted = st.multiselect("Weather-impacted regions", sorted(MARKETS["Region"].unique()))
    p = dict(cap_mult=cap_mult, air_lift=air_lift, air_enabled=air_enabled, promise_cutoff=promise_cutoff, ground_speed=float(ground_speed),
             sort_hours=sort_hours, ofd_hour=ofd_hour, air_tender_hour=air_tender, hub_inbound_et=26.5, circuity=1.2, fuel_pct=fuel,
             rate_mult=rate_mult, packaging=packaging, penalty_mult=penalty_mult, unfulfilled_cost=float(unf_cost), margin_pct=margin, resi_fee=1.20)
    return n, fsl, disrupted, p


def build_context():
    n, fsl, disrupted, p = sidebar_params()
    orders = st.session_state.get("uploaded_orders")
    if orders is None:
        orders = simulate_orders(n)
    fac = st.session_state["facilities"]
    if not fsl:
        fac = fac[~fac["Type"].str.contains("Forward", na=False)].reset_index(drop=True)
    plan, opts = solve_plan(orders, fac, p)
    util = pd.DataFrame({"Assigned": plan["Assigned_Node"].value_counts().reindex(fac["Facility_ID"]).fillna(0),
                         "Capacity": np.floor(fac.set_index("Facility_ID")["Daily_Capacity"] * p["cap_mult"])})
    util["Utilization"] = (util["Assigned"] / util["Capacity"].replace(0, np.nan)).fillna(0)
    actuals = simulate_actuals(plan, util["Utilization"].to_dict(), tuple(disrupted))
    risk, imp = score_risk(actuals)
    actuals = actuals.assign(Risk=risk)
    st.session_state["ctx"] = dict(orders=orders, fac=fac, p=p, plan=plan, opts=opts, util=util, base=baseline_plan(orders, fac, p),
                                   actuals=actuals, importance=imp, disrupted=list(disrupted))


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon=":material/local_shipping:", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.session_state.setdefault("facilities", DEFAULT_FACILITIES.copy())
    pages = {
        "Command Center": [
            st.Page(page_overview, title="Executive Overview", icon=":material/dashboard:", url_path="overview", default=True),
            st.Page(page_playbook, title="One-Day Delivery Playbook", icon=":material/schedule:", url_path="playbook"),
        ],
        "Plan": [
            st.Page(page_network, title="Network Map", icon=":material/hub:", url_path="network"),
            st.Page(page_sourcing, title="Fulfillment Optimizer", icon=":material/alt_route:", url_path="optimizer"),
            st.Page(page_cutoffs, title="Cutoffs & Time-in-Transit", icon=":material/timer:", url_path="cutoffs"),
            st.Page(page_capacity, title="Capacity & Inventory", icon=":material/warehouse:", url_path="capacity"),
        ],
        "Analyze": [
            st.Page(page_cost, title="Cost-to-Serve", icon=":material/payments:", url_path="cost"),
            st.Page(page_risk, title="SLA Risk & Exceptions", icon=":material/warning:", url_path="risk"),
            st.Page(page_ai, title="AI Network Analyst", icon=":material/smart_toy:", url_path="ai-analyst"),
        ],
        "Configure": [
            st.Page(page_data, title="Data & Assumptions", icon=":material/settings:", url_path="data"),
        ],
    }
    nav = st.navigation(pages)
    try:
        build_context()
    except Exception as e:
        st.error(f"Could not build the plan: {e}")
        st.stop()
    nav.run()


if __name__ == "__main__":
    main()
