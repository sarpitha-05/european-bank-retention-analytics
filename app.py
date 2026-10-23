import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Customer Retention Analytics", page_icon="🏦", layout="wide")

PROFILE_ORDER = ["Active Engaged", "Active Low-Product", "Inactive Disengaged", "Inactive High-Balance"]
PROFILE_COLORS = {"Active Engaged": "#2e7d32", "Active Low-Product": "#f9a825",
                  "Inactive Disengaged": "#ef6c00", "Inactive High-Balance": "#c62828"}


@st.cache_data
def load_data():
    return pd.read_csv("bank_features.csv")


def churn_rate(d):
    return d["Exited"].mean() * 100 if len(d) else 0.0


def bar_churn(table, x, title, color=None, cmap=None):
    fig = px.bar(table, x=x, y="ChurnRate", text="ChurnRate", color=color,
                 color_discrete_map=cmap, title=title,
                 labels={"ChurnRate": "Churn rate (%)"})
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(showlegend=False, yaxis_range=[0, max(table["ChurnRate"].max() * 1.2, 5)])
    return fig


def churn_by(d, col):
    t = d.groupby(col, observed=True)["Exited"].agg(Customers="count", ChurnRate="mean").reset_index()
    t["ChurnRate"] = t["ChurnRate"] * 100
    return t


df = load_data()
overall_churn_full = churn_rate(df)

# =============================== SIDEBAR FILTERS ===============================
st.sidebar.header("Filters")
activity = st.sidebar.multiselect("Activity status", ["Active", "Inactive"], default=["Active", "Inactive"])
profiles = st.sidebar.multiselect("Engagement profile", PROFILE_ORDER, default=PROFILE_ORDER)
geos = st.sidebar.multiselect("Geography", sorted(df["Geography"].unique()),
                              default=sorted(df["Geography"].unique()))
prod_min, prod_max = st.sidebar.slider("Number of products", 1, 4, (1, 4))
bal_lo, bal_hi = st.sidebar.slider("Balance range", 0, int(df["Balance"].max()) + 1,
                                   (0, int(df["Balance"].max()) + 1), step=5000)
sal_lo, sal_hi = st.sidebar.slider("Estimated salary range", 0, int(df["EstimatedSalary"].max()) + 1,
                                   (0, int(df["EstimatedSalary"].max()) + 1), step=5000)

act_map = {"Active": 1, "Inactive": 0}
f = df[
    df["IsActiveMember"].isin([act_map[a] for a in activity])
    & df["EngagementProfile"].isin(profiles)
    & df["Geography"].isin(geos)
    & df["NumOfProducts"].between(prod_min, prod_max)
    & df["Balance"].between(bal_lo, bal_hi)
    & df["EstimatedSalary"].between(sal_lo, sal_hi)
]

st.title("🏦 Customer Engagement & Product Utilization Analytics")
st.caption("Retention strategy dashboard - European bank, 10,000 customers. "
           "Use the sidebar filters to explore behaviour segments.")

if f.empty:
    st.warning("No customers match the current filters. Please widen the filters.")
    st.stop()

# =============================== KPI ROW ===============================
act_churn = churn_rate(f[f.IsActiveMember == 1])
ina_churn = churn_rate(f[f.IsActiveMember == 0])
err = (ina_churn / act_churn) if act_churn > 0 and len(f[f.IsActiveMember == 0]) else np.nan
hb = f[f.HighBalance == 1]
hb_dis = (hb["IsActiveMember"].eq(0).mean() * 100) if len(hb) else np.nan
card_ret = 100 - churn_rate(f[f.HasCrCard == 1]) if len(f[f.HasCrCard == 1]) else np.nan
nocard_ret = 100 - churn_rate(f[f.HasCrCard == 0]) if len(f[f.HasCrCard == 0]) else np.nan
ccs = card_ret / nocard_ret if (pd.notna(card_ret) and pd.notna(nocard_ret) and nocard_ret > 0) else np.nan

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Customers", f"{len(f):,}")
k2.metric("Churn rate", f"{churn_rate(f):.1f}%", f"{churn_rate(f) - overall_churn_full:+.1f} pts vs all",
          delta_color="inverse")
k3.metric("Engagement Retention Ratio", "n/a" if pd.isna(err) else f"{err:.2f}x")
k4.metric("High-Balance Disengagement", "n/a" if pd.isna(hb_dis) else f"{hb_dis:.1f}%")
k5.metric("Credit Card Stickiness", "n/a" if pd.isna(ccs) else f"{ccs:.3f}")
k6.metric("Avg Relationship Strength", f"{f['RSI'].mean():.1f}")

tab1, tab2, tab3, tab4 = st.tabs(["Engagement vs Churn", "Product Utilization",
                                  "High-Value Disengaged Detector", "Retention Strength"])

# =============================== TAB 1 ===============================
with tab1:
    c1, c2 = st.columns(2)
    t = churn_by(f, "IsActiveMember")
    t["Status"] = t["IsActiveMember"].map({1: "Active", 0: "Inactive"})
    c1.plotly_chart(bar_churn(t, "Status", "Churn rate: Active vs Inactive", color="Status",
                              cmap={"Active": "#2e7d32", "Inactive": "#c62828"}))
    tp = churn_by(f, "EngagementProfile")
    tp["EngagementProfile"] = pd.Categorical(tp["EngagementProfile"], PROFILE_ORDER, ordered=True)
    tp = tp.sort_values("EngagementProfile")
    c2.plotly_chart(bar_churn(tp, "EngagementProfile", "Churn rate by engagement profile",
                              color="EngagementProfile", cmap=PROFILE_COLORS))
    st.subheader("Engagement profile summary")
    summ = f.groupby("EngagementProfile").agg(
        Customers=("Exited", "count"), ChurnRate=("Exited", lambda s: round(s.mean() * 100, 1)),
        AvgBalance=("Balance", "mean"), AvgProducts=("NumOfProducts", "mean"),
        AvgTenure=("Tenure", "mean")).round(1).reindex(PROFILE_ORDER).dropna(how="all")
    st.dataframe(summ)

# =============================== TAB 2 ===============================
with tab2:
    c1, c2 = st.columns(2)
    tpr = churn_by(f, "NumOfProducts")
    c1.plotly_chart(bar_churn(tpr, "NumOfProducts", "Churn rate by number of products"))
    hm = f.pivot_table(index="NumOfProducts", columns="IsActiveMember", values="Exited",
                       aggfunc="mean") * 100
    hm.columns = ["Inactive" if c == 0 else "Active" for c in hm.columns]
    fig = px.imshow(hm, text_auto=".1f", color_continuous_scale="Reds", aspect="auto",
                    title="Churn % : products x activity", labels={"color": "Churn %"})
    c2.plotly_chart(fig)

    s, m = f[f.NumOfProducts == 1], f[f.NumOfProducts >= 2]
    a, b, c = st.columns(3)
    a.metric("Single-product churn", "n/a" if s.empty else f"{churn_rate(s):.1f}%")
    b.metric("Multi-product (2+) churn", "n/a" if m.empty else f"{churn_rate(m):.1f}%")
    two = f[f.NumOfProducts == 2]
    c.metric("2-product churn", "n/a" if two.empty else f"{churn_rate(two):.1f}%")

    st.subheader("Product Depth Index")
    overall_ret = 100 - churn_rate(f)
    pdi = tpr.copy()
    pdi["ProductDepthIndex"] = ((100 - pdi["ChurnRate"]) / overall_ret).round(2) if overall_ret > 0 else np.nan
    pdi["ChurnRate"] = pdi["ChurnRate"].round(1)
    st.dataframe(pdi.set_index("NumOfProducts"))
    st.caption("Index above 1 = retention better than the filtered-group average. "
               "Customers with 3-4 products are few but churn heavily - review how these bundles are sold.")

    st.subheader("Credit card ownership")
    cc = churn_by(f, "HasCrCard")
    cc["HasCrCard"] = cc["HasCrCard"].map({1: "Has credit card", 0: "No credit card"})
    st.dataframe(cc.set_index("HasCrCard").round(1))

# =============================== TAB 3 ===============================
with tab3:
    st.markdown("Find **inactive customers with high balances** - high-value relationships "
                "at risk of silent churn. This detector uses the full dataset and its own thresholds "
                "(only the Geography filter applies).")
    d1, d2, d3 = st.columns(3)
    min_bal = d1.slider("Minimum balance", 0, int(df["Balance"].max()), 100000, step=5000)
    min_sal = d2.slider("Minimum estimated salary", 0, int(df["EstimatedSalary"].max()), 0, step=5000)
    only_current = d3.checkbox("Only customers who have NOT churned yet (retention targets)", value=True)

    base = df[df["Geography"].isin(geos)]
    risk = base[(base.IsActiveMember == 0) & (base.Balance >= min_bal) & (base.EstimatedSalary >= min_sal)]
    peers = base[(base.IsActiveMember == 1) & (base.Balance >= min_bal) & (base.EstimatedSalary >= min_sal)]

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Disengaged high-value customers", f"{len(risk):,}")
    r2.metric("Their churn rate", "n/a" if risk.empty else f"{churn_rate(risk):.1f}%")
    r3.metric("Active peers' churn rate", "n/a" if peers.empty else f"{churn_rate(peers):.1f}%")
    r4.metric("Balance held by this group", f"{risk['Balance'].sum() / 1e6:,.1f} M")

    shown = risk[risk.Exited == 0] if only_current else risk
    cols = ["CustomerId", "Geography", "Age", "Tenure", "NumOfProducts", "HasCrCard",
            "Balance", "EstimatedSalary", "RSI", "Exited"]
    shown = shown.sort_values("Balance", ascending=False)[cols]
    st.write(f"Showing {len(shown):,} customers, highest balance first")
    st.dataframe(shown.head(500))
    st.download_button("Download this list (CSV)", shown.to_csv(index=False).encode("utf-8"),
                       file_name="high_value_disengaged_customers.csv", mime="text/csv")

    if not risk.empty:
        sc = px.scatter(risk.sample(min(len(risk), 1500), random_state=1), x="EstimatedSalary", y="Balance",
                        color=risk.sample(min(len(risk), 1500), random_state=1)["Exited"].map({0: "Retained", 1: "Churned"}),
                        opacity=0.6, title="Disengaged high-value customers: salary vs balance",
                        labels={"color": "Status"})
        st.plotly_chart(sc)

# =============================== TAB 4 ===============================
with tab4:
    c1, c2 = st.columns(2)
    tt = churn_by(f, "RSITier")
    tt["RSITier"] = pd.Categorical(tt["RSITier"], ["Weak", "Moderate", "Strong"], ordered=True)
    tt = tt.sort_values("RSITier")
    c1.plotly_chart(bar_churn(tt, "RSITier", "Churn rate by Relationship Strength tier", color="RSITier",
                              cmap={"Weak": "#c62828", "Moderate": "#f9a825", "Strong": "#2e7d32"}))
    g = f.copy()
    g["RSIBand"] = pd.cut(g["RSI"], bins=range(0, 101, 10), include_lowest=True).astype(str)
    tb = g.groupby("RSIBand", sort=False)["Exited"].agg(Customers="count", ChurnRate="mean").reset_index()
    tb["ChurnRate"] = tb["ChurnRate"] * 100
    tb["lo"] = tb["RSIBand"].str.extract(r"([\d.]+),").astype(float)
    tb = tb.sort_values("lo")
    fig = px.line(tb, x="RSIBand", y="ChurnRate", markers=True,
                  title="Churn rate vs Relationship Strength Index (engagement threshold)",
                  labels={"ChurnRate": "Churn rate (%)", "RSIBand": "RSI band"})
    c2.plotly_chart(fig)

    st.subheader("Sticky customer profile")
    st.markdown("**Sticky** = active member + exactly 2 products + RSI of 70 or more.")
    sticky = f[(f.IsActiveMember == 1) & (f.NumOfProducts == 2) & (f.RSI >= 70)]
    rest = f.drop(sticky.index)
    s1, s2, s3 = st.columns(3)
    s1.metric("Sticky customers", f"{len(sticky):,}", f"{len(sticky) / len(f) * 100:.1f}% of selection", delta_color="off")
    s2.metric("Sticky churn rate", "n/a" if sticky.empty else f"{churn_rate(sticky):.1f}%")
    s3.metric("All other customers", "n/a" if rest.empty else f"{churn_rate(rest):.1f}%")

    h = px.histogram(f.assign(Status=f["Exited"].map({0: "Retained", 1: "Churned"})), x="RSI",
                     color="Status", barmode="overlay", opacity=0.65, nbins=25,
                     color_discrete_map={"Retained": "#2e7d32", "Churned": "#c62828"},
                     title="RSI distribution: retained vs churned")
    st.plotly_chart(h)

st.markdown("---")
st.caption("Churn rate = share of customers with Exited = 1. High balance = top 25% of balances (>= 127,644). "
           "RSI = Relationship Strength Index (0-100): activity 40, product depth 25, tenure 15, "
           "credit card 10, holds a balance 10.")