import streamlit as st
import altair as alt
import pandas as pd
import duckdb
import numpy as np
import warnings
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix, roc_curve
)
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Olist Analytics",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Minimal CSS: bigger fonts only, no theme override ─────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        font-size: 16px;
    }

    /* Slightly larger axis/label text in charts */
    .vega-embed text { font-size: 13px !important; }

    .insight-box {
        background: #f0f4ff;
        border-left: 4px solid #5b6ef5;
        border-radius: 0 8px 8px 0;
        padding: 0.9rem 1.2rem;
        margin: 0.5rem 0 1rem 0;
        color: #374151;
        font-size: 0.97rem;
        line-height: 1.7;
    }
    .insight-box strong { color: #1e293b; }

    /* Larger metric values */
    [data-testid="stMetricValue"] { font-size: 1.7rem !important; }
    [data-testid="stMetricLabel"] { font-size: 0.9rem !important; }

    /* Tab font size */
    .stTabs [data-baseweb="tab"] { font-size: 0.95rem !important; }

    h1 { font-size: 2.1rem !important; font-weight: 700 !important; }
    h2 { font-size: 1.4rem !important; }
    h3 { font-size: 1.15rem !important; }
    p, li { font-size: 1rem; line-height: 1.7; }
</style>
""", unsafe_allow_html=True)

# ── Palette & Altair theme ────────────────────────────────────
PALETTE  = ['#5b6ef5', '#0ea5e9', '#10b981', '#f97316', '#ef4444']

def insight(text):
    st.markdown(f'<div class="insight-box">💡 {text}</div>', unsafe_allow_html=True)


# ── Load data ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    base = "data"
    orders    = pd.read_csv(f"{base}/olist_orders_dataset.csv")
    payments  = pd.read_csv(f"{base}/olist_order_payments_dataset.csv")
    items     = pd.read_csv(f"{base}/olist_order_items_dataset.csv")
    customers = pd.read_csv(f"{base}/olist_customers_dataset.csv")
    reviews   = pd.read_csv(f"{base}/olist_order_reviews_dataset.csv")

    # Source CSVs store dates as DD-MM-YY (e.g. "07-08-18 15:27"), an Excel-style
    # re-export. Parsing without an explicit format lets pandas fall back to
    # month-first, which silently swaps day/month for every date where the day is
    # <= 12 (~40% of rows) and corrupts every downstream time calculation.
    for col in ['order_purchase_timestamp','order_approved_at',
                'order_delivered_carrier_date','order_delivered_customer_date',
                'order_estimated_delivery_date']:
        orders[col] = pd.to_datetime(orders[col], format='%d-%m-%y %H:%M', errors='coerce')

    return orders, payments, items, customers, reviews

orders, payments, items, customers, reviews = load_data()

con = duckdb.connect()
con.register('orders_tbl',    orders)
con.register('payments_tbl',  payments)
con.register('items_tbl',     items)
con.register('customers_tbl', customers)
con.register('reviews_tbl',   reviews)


# ── Pre-compute KPIs ──────────────────────────────────────────
total_revenue = payments['payment_value'].sum()
avg_review    = reviews['review_score'].mean()
delivered_df  = orders[orders['order_status'] == 'delivered'].dropna(
    subset=['order_delivered_customer_date','order_estimated_delivery_date'])
late_count    = (delivered_df['order_delivered_customer_date'] >
                 delivered_df['order_estimated_delivery_date']).sum()
late_pct_kpi  = round(100 * late_count / len(delivered_df), 1)

cust_kpi = con.execute("""
    SELECT customer_unique_id, COUNT(DISTINCT o.order_id) AS total_orders
    FROM orders_tbl o JOIN customers_tbl c ON o.customer_id = c.customer_id
    WHERE o.order_status = 'delivered'
    GROUP BY customer_unique_id
""").df()
returning_rate = round(100 * (cust_kpi['total_orders'] > 1).sum() / len(cust_kpi), 1)


# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════
with st.container(horizontal_alignment="center"):
    st.title("📦 Olist E-Commerce Analytics")
    st.markdown(
        "An analysis of the Brazilian Olist marketplace covering **payment behavior**, "
        "**delivery performance**, and **customer retention** — "
        "using data from 2016 to 2018."
    )

st.space()

# ── KPI row ───────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Orders",       f"{len(orders):,}")
k2.metric("Total Revenue",      f"R$ {total_revenue:,.0f}")
k3.metric("Avg Review Score",   f"{avg_review:.2f} / 5.00")
k4.metric("Late Delivery Rate", f"{late_pct_kpi}%")
k5.metric("Retention Rate",     f"{returning_rate}%")

st.space("large")

# ── Tabs ──────────────────────────────────────────────────────
tab_eda, tab_pay, tab_del, tab_ret, tab_rfm, tab_ml, tab_rec = st.tabs([
    "📈  EDA", "💳  Payment Behavior", "🚚  Delivery & Satisfaction",
    "🔁  Customer Retention", "🎯  RFM Segmentation",
    "🤖  Churn Prediction", "💼  Recommendations"
])


# ═══════════════════════════════════════════════════════════════
# TAB 1 — EDA
# ═══════════════════════════════════════════════════════════════
with tab_eda:
    st.space()

    # ── Monthly trend ─────────────────────────────────────────
    st.subheader("Order Volume Over Time")
    st.caption("Use the slider to zoom into any time window.")

    orders_cp              = orders.copy()
    orders_cp['month']     = orders_cp['order_purchase_timestamp'].dt.to_period('M')
    monthly                = orders_cp.groupby('month').size().reset_index(name='order_count')
    monthly['month_str']   = monthly['month'].astype(str)
    monthly['month_dt']    = pd.to_datetime(monthly['month_str'])

    min_d = monthly['month_dt'].min().to_pydatetime()
    max_d = monthly['month_dt'].max().to_pydatetime()

    date_range = st.slider("Date range", min_value=min_d, max_value=max_d,
                           value=(min_d, max_d), format="MMM YYYY")
    mf = monthly[(monthly['month_dt'] >= date_range[0]) & (monthly['month_dt'] <= date_range[1])]

    area = (
        alt.Chart(mf, title="Monthly Order Volume")
        .mark_area(
            line={"color": PALETTE[0], "strokeWidth": 2.5},
            color=alt.Gradient(
                gradient="linear",
                stops=[alt.GradientStop(color=PALETTE[0], offset=0),
                       alt.GradientStop(color="#e0e7ff", offset=1)],
                x1=1, x2=1, y1=1, y2=0,
            )
        )
        .encode(
            alt.X("month_dt:T", title="Month", axis=alt.Axis(format="%b %Y", labelAngle=-40, labelFontSize=13)),
            alt.Y("order_count:Q", title="Orders", axis=alt.Axis(format=",d", labelFontSize=13)),
            tooltip=[alt.Tooltip("month_str:N", title="Month"),
                     alt.Tooltip("order_count:Q", title="Orders", format=",")]
        )
        .properties(height=320)
    )
    st.altair_chart(area, use_container_width=True)

    peak       = monthly.loc[monthly['order_count'].idxmax()]
    peak_label = peak['month_dt'].strftime('%B %Y')
    peak_val   = int(peak['order_count'])
    base_val   = int(monthly[monthly['order_count'] >= 100]['order_count'].iloc[0])
    growth_x   = peak_val / base_val
    nov17      = monthly.loc[monthly['month_str'] == '2017-11', 'order_count']
    oct17      = monthly.loc[monthly['month_str'] == '2017-10', 'order_count']
    bf_jump    = (nov17.iloc[0] / oct17.iloc[0] - 1) * 100 if len(nov17) and len(oct17) else None
    bf_txt     = (f", jumping ~{bf_jump:.0f}% over the prior month" if bf_jump else "")
    insight(
        f"Order volume grew more than <strong>{growth_x:.0f}× in under two years</strong> — from a few hundred "
        f"orders a month in late 2016 to a peak of <strong>~{peak_val:,} in {peak_label}</strong>. "
        f"That peak is the platform's <strong>Black Friday spike</strong>{bf_txt}, before volume settled back to trend. "
        "Seasonality like this means logistics and support have to scale <em>ahead</em> of peaks, not react to them."
    )

    st.space()

    # ── YoY ───────────────────────────────────────────────────
    st.subheader("Year-over-Year Comparison (Jan – Aug)")
    st.caption("2018 data is capped at August as the dataset ends mid-year.")

    yoy = orders_cp[orders_cp['order_purchase_timestamp'].dt.year.isin([2017, 2018])].copy()
    yoy['year']       = yoy['order_purchase_timestamp'].dt.year.astype(str)
    yoy['month_num']  = yoy['order_purchase_timestamp'].dt.month
    yoy['month_name'] = yoy['order_purchase_timestamp'].dt.strftime('%b')
    yoy = yoy[yoy['month_num'].between(1, 8)]
    yoy_agg    = yoy.groupby(['year','month_num','month_name']).size().reset_index(name='order_count')
    month_sort = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug']

    yoy_chart = (
        alt.Chart(yoy_agg, title="Monthly Orders: 2017 vs 2018")
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            alt.X("month_name:N", title=None, sort=month_sort,
                  axis=alt.Axis(labelAngle=0, labelFontSize=13)),
            alt.Y("order_count:Q", title="Orders", axis=alt.Axis(format=",d", labelFontSize=13)),
            alt.Color("year:N", title="Year",
                      scale=alt.Scale(domain=['2017','2018'], range=[PALETTE[0], PALETTE[2]])),
            alt.XOffset("year:N"),
            tooltip=[alt.Tooltip("month_name:N", title="Month"),
                     alt.Tooltip("year:N", title="Year"),
                     alt.Tooltip("order_count:Q", title="Orders", format=",")]
        )
        .properties(height=320)
    )
    st.altair_chart(yoy_chart, use_container_width=True)

    piv       = yoy_agg.pivot(index='month_num', columns='year', values='order_count')
    all_up    = bool((piv['2018'] > piv['2017']).all())
    jan_ratio = piv.loc[1, '2018'] / piv.loc[1, '2017']
    aug_pct   = (piv.loc[8, '2018'] / piv.loc[8, '2017'] - 1) * 100
    insight(
        f"2018 outsold 2017 in <strong>{'every month' if all_up else 'most months'} shown</strong>. "
        f"The gap was widest early in the year — January 2018 handled roughly <strong>{jan_ratio:.0f}× the "
        f"January 2017 volume</strong> — because the platform was still ramping up in early 2017. "
        f"It narrowed to about <strong>+{aug_pct:.0f}%</strong> by August as the 2017 base caught up. "
        "The takeaway is broad-based year-on-year growth rather than a single seasonal blip."
    )

    st.space()

    # ── Order status ──────────────────────────────────────────
    st.subheader("Order Status Breakdown")

    status_df = orders['order_status'].value_counts().reset_index()
    status_df.columns = ['Status', 'Count']
    status_df['% of Total'] = (status_df['Count'] / status_df['Count'].sum() * 100).round(1)

    cols = st.columns(2, border=True)
    with cols[0]:
        st.subheader("Distribution by status")
        status_chart = (
            alt.Chart(status_df, title=alt.TitleParams("", anchor="start"))
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                alt.X("Status:N", sort="-y", axis=alt.Axis(labelAngle=-30, labelFontSize=13)),
                alt.Y("Count:Q", axis=alt.Axis(format=",d", labelFontSize=13), title="Order Count"),
                alt.Color("Status:N", scale=alt.Scale(range=PALETTE), legend=None),
                tooltip=["Status:N",
                         alt.Tooltip("Count:Q", format=","),
                         alt.Tooltip("% of Total:Q", format=".1f", title="% of Total")]
            )
            .properties(height=280)
        )
        st.altair_chart(status_chart, use_container_width=True)

    with cols[1]:
        st.subheader("Summary table")
        st.dataframe(
            status_df,
            use_container_width=True, hide_index=True, height=280,
            column_config={
                "Count":      st.column_config.NumberColumn(format="localized"),
                "% of Total": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
            }
        )


# ═══════════════════════════════════════════════════════════════
# TAB 2 — PAYMENT BEHAVIOR
# ═══════════════════════════════════════════════════════════════
with tab_pay:
    st.space()
    st.subheader("Payment Method Overview")
    st.caption("Which payment methods do customers prefer, and how does order value differ across them?")

    pay_sum = con.execute("""
        SELECT
            payment_type,
            COUNT(DISTINCT order_id)                          AS order_count,
            ROUND(AVG(payment_value), 2)                      AS avg_order_value,
            ROUND(SUM(payment_value), 2)                      AS total_revenue,
            ROUND(100.0 * COUNT(DISTINCT order_id)
                  / SUM(COUNT(DISTINCT order_id)) OVER (), 1) AS pct_of_orders
        FROM payments_tbl
        GROUP BY payment_type
        ORDER BY order_count DESC
    """).df()

    metric_choice = st.segmented_control(
        "Visualise by",
        ["% of Orders", "Avg Order Value (BRL)", "Total Revenue (BRL)"],
        default="% of Orders",
    )
    col_map = {
        "% of Orders":           "pct_of_orders",
        "Avg Order Value (BRL)": "avg_order_value",
        "Total Revenue (BRL)":   "total_revenue",
    }
    mcol = col_map[metric_choice] if metric_choice else "pct_of_orders"

    cols = st.columns(2, border=True)
    with cols[0]:
        st.subheader(f"By {metric_choice}")
        pay_chart = (
            alt.Chart(pay_sum, title=alt.TitleParams("", anchor="start"))
            .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
            .encode(
                alt.Y("payment_type:N", title=None,
                      sort=alt.EncodingSortField(field=mcol, order="descending"),
                      axis=alt.Axis(labelFontSize=13)),
                alt.X(f"{mcol}:Q", title=metric_choice, axis=alt.Axis(labelFontSize=13)),
                alt.Color("payment_type:N", scale=alt.Scale(range=PALETTE), legend=None),
                tooltip=[alt.Tooltip("payment_type:N", title="Method"),
                         alt.Tooltip(f"{mcol}:Q", title=metric_choice, format=",.2f"),
                         alt.Tooltip("order_count:Q", title="Orders", format=",")]
            )
            .properties(height=240)
        )
        st.altair_chart(pay_chart, use_container_width=True)

    with cols[1]:
        st.subheader("Summary table")
        st.dataframe(
            pay_sum.rename(columns={
                'payment_type':'Method','order_count':'Orders',
                'avg_order_value':'Avg Value (R$)','total_revenue':'Revenue (R$)',
                'pct_of_orders':'% Share'
            }),
            use_container_width=True, hide_index=True, height=240,
            column_config={
                "Orders":        st.column_config.NumberColumn(format="localized"),
                "Avg Value (R$)":st.column_config.NumberColumn(format="R$ %.2f"),
                "Revenue (R$)":  st.column_config.NumberColumn(format="R$ %.0f"),
                "% Share":       st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
            }
        )

    cc_pct = pay_sum.set_index('payment_type').loc['credit_card', 'pct_of_orders']
    insight(
        f"<strong>Credit card dominates</strong> — about <strong>{cc_pct:.0f}% of orders</strong>, and the highest "
        "average order value of any method. That pairing suggests larger baskets lean on its installment option. "
        "<strong>Boleto</strong> (a Brazilian cash voucher) is a distant second with a lower average ticket — more "
        "price-sensitive buyers — while vouchers and debit are a small tail. "
        "Protecting the credit-card checkout is therefore protecting the highest-value orders."
    )

    st.space()

    # ── Installments ──────────────────────────────────────────
    st.subheader("Credit Card Installment Behavior")
    st.caption("Bars show order count; the dashed line shows average order value per installment tier.")

    installments = con.execute("""
        SELECT payment_installments,
               COUNT(*) AS order_count,
               ROUND(AVG(payment_value), 2) AS avg_value
        FROM payments_tbl
        WHERE payment_type = 'credit_card'
          AND payment_installments BETWEEN 1 AND 12
        GROUP BY payment_installments
        ORDER BY payment_installments
    """).df()

    inst_range    = st.slider("Installment range", 1, 12, (1, 12))
    inst_filtered = installments[installments['payment_installments'].between(inst_range[0], inst_range[1])]

    cols = st.columns(2, border=True)
    with cols[0]:
        st.subheader("Orders vs Avg Value by installment")
        inst_bars = (
            alt.Chart(inst_filtered, title=alt.TitleParams("", anchor="start"))
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color=PALETTE[0])
            .encode(
                alt.X("payment_installments:O", title="Installments",
                      axis=alt.Axis(labelAngle=0, labelFontSize=13)),
                alt.Y("order_count:Q", title="Number of Orders",
                      axis=alt.Axis(format=",d", labelFontSize=13)),
                tooltip=[alt.Tooltip("payment_installments:O", title="Installments"),
                         alt.Tooltip("order_count:Q", title="Orders", format=","),
                         alt.Tooltip("avg_value:Q", title="Avg Value (R$)", format=",.2f")]
            )
            .properties(height=300)
        )
        avg_line = (
            alt.Chart(inst_filtered)
            .mark_line(color=PALETTE[3], strokeDash=[5,3], strokeWidth=2.5,
                       point=alt.OverlayMarkDef(color=PALETTE[3], filled=True, size=60))
            .encode(
                alt.X("payment_installments:O"),
                alt.Y("avg_value:Q", title="Avg Order Value (R$)",
                      axis=alt.Axis(labelFontSize=13)),
                tooltip=[alt.Tooltip("payment_installments:O", title="Installments"),
                         alt.Tooltip("avg_value:Q", title="Avg Value (R$)", format=",.2f")]
            )
        )
        st.altair_chart(
            alt.layer(inst_bars, avg_line).resolve_scale(y='independent'),
            use_container_width=True
        )

    with cols[1]:
        st.subheader("Summary table")
        st.dataframe(
            inst_filtered.rename(columns={
                'payment_installments':'Installments',
                'order_count':'Orders',
                'avg_value':'Avg Value (R$)'
            }),
            use_container_width=True, hide_index=True, height=300,
            column_config={
                "Orders":        st.column_config.NumberColumn(format="localized"),
                "Avg Value (R$)":st.column_config.NumberColumn(format="R$ %.2f"),
            }
        )

    inst_total = installments['order_count'].sum()
    pct_1      = 100 * installments.loc[installments['payment_installments'] == 1, 'order_count'].iloc[0] / inst_total
    avg_1      = installments.loc[installments['payment_installments'] == 1, 'avg_value'].iloc[0]
    hi_grp     = installments[installments['payment_installments'] >= 6]
    avg_hi     = (hi_grp['avg_value'] * hi_grp['order_count']).sum() / hi_grp['order_count'].sum()
    insight(
        f"About <strong>{pct_1:.0f}% of credit-card orders are paid in a single installment</strong> (full amount "
        f"upfront). But order value climbs steeply as installments rise: ~R$ {avg_1:,.0f} at one installment versus "
        f"~R$ {avg_hi:,.0f} across six-plus. Customers only split payments when the ticket is large — so flexible "
        "6–12-month terms on high-value categories can unlock bigger baskets they wouldn't commit to paying upfront."
    )


# ═══════════════════════════════════════════════════════════════
# TAB 3 — DELIVERY & SATISFACTION
# ═══════════════════════════════════════════════════════════════
with tab_del:
    st.space()
    st.subheader("Delivery Performance vs Review Scores")
    st.caption("Do late deliveries hurt review scores? By how much?")

    delivery_df = con.execute("""
        SELECT
            o.order_id,
            CASE
                WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date
                THEN 'Late' ELSE 'On Time'
            END AS delivery_status,
            DATEDIFF('day',
                o.order_estimated_delivery_date,
                o.order_delivered_customer_date
            ) AS days_late,
            r.review_score
        FROM orders_tbl o
        JOIN reviews_tbl r ON o.order_id = r.order_id
        WHERE o.order_status = 'delivered'
          AND o.order_delivered_customer_date IS NOT NULL
          AND o.order_estimated_delivery_date IS NOT NULL
    """).df()

    del_sum  = delivery_df.groupby('delivery_status').agg(
        order_count=('order_id','count'), avg_review_score=('review_score','mean')
    ).round(2).reset_index()
    late_pct = round(100*(delivery_df['delivery_status']=='Late').sum()/len(delivery_df), 1)
    on_score = del_sum[del_sum['delivery_status']=='On Time']['avg_review_score'].values[0]
    lt_score = del_sum[del_sum['delivery_status']=='Late']['avg_review_score'].values[0]

    m1, m2, m3 = st.columns(3)
    m1.metric("Late Deliveries",     f"{late_pct}%")
    m2.metric("Avg Score — On Time", f"{on_score:.2f} ★")
    m3.metric("Avg Score — Late",    f"{lt_score:.2f} ★",
              delta=f"{lt_score-on_score:.2f}", delta_color="inverse")

    st.space()

    status_filter = st.multiselect(
        "Filter by delivery status",
        options=["On Time","Late"], default=["On Time","Late"]
    )
    fd = delivery_df[delivery_df['delivery_status'].isin(status_filter)]
    score_dist = fd.groupby(['delivery_status','review_score']).size().reset_index(name='count')
    score_dist['pct'] = score_dist.groupby('delivery_status')['count'].transform(
        lambda x: x / x.sum() * 100
    )

    cols = st.columns(2, border=True)
    with cols[0]:
        st.subheader("Avg review score")
        bar_sum = (
            alt.Chart(del_sum[del_sum['delivery_status'].isin(status_filter)],
                      title=alt.TitleParams("", anchor="start"))
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                alt.X("delivery_status:N", title=None, axis=alt.Axis(labelAngle=0, labelFontSize=14)),
                alt.Y("avg_review_score:Q", title="Avg Review Score",
                      scale=alt.Scale(domain=[0, 5.5]),
                      axis=alt.Axis(labelFontSize=13)),
                alt.Color("delivery_status:N",
                          scale=alt.Scale(domain=["On Time","Late"],
                                          range=[PALETTE[2], PALETTE[4]]), legend=None),
                tooltip=[alt.Tooltip("delivery_status:N", title="Status"),
                         alt.Tooltip("avg_review_score:Q", title="Avg Score", format=".2f"),
                         alt.Tooltip("order_count:Q", title="Orders", format=",")]
            )
            .properties(height=300)
        )
        labels = bar_sum.mark_text(dy=-12, fontSize=16, fontWeight=700).encode(
            text=alt.Text("avg_review_score:Q", format=".2f"),
            color=alt.value("#1e293b")
        )
        st.altair_chart(bar_sum + labels, use_container_width=True)

    with cols[1]:
        st.subheader("Score distribution")
        line_dist = (
            alt.Chart(score_dist, title=alt.TitleParams("", anchor="start"))
            .mark_line(point=alt.OverlayMarkDef(filled=True, size=100), strokeWidth=2.5)
            .encode(
                alt.X("review_score:O", title="Review Score",
                      axis=alt.Axis(labelAngle=0, labelFontSize=13)),
                alt.Y("pct:Q", title="% of Orders", axis=alt.Axis(format=".1f", labelFontSize=13)),
                alt.Color("delivery_status:N",
                          scale=alt.Scale(domain=["On Time","Late"],
                                          range=[PALETTE[2], PALETTE[4]]), title="Status"),
                tooltip=[alt.Tooltip("delivery_status:N", title="Status"),
                         alt.Tooltip("review_score:O", title="Score"),
                         alt.Tooltip("pct:Q", title="% of Orders", format=".1f"),
                         alt.Tooltip("count:Q", title="Count", format=",")]
            )
            .properties(height=300)
        )
        st.altair_chart(line_dist, use_container_width=True)

    insight(
        f"Late deliveries hurt satisfaction measurably. On-time orders average <strong>{on_score:.2f} stars</strong> "
        f"versus just <strong>{lt_score:.2f}</strong> for late ones — a gap of about "
        f"<strong>{on_score - lt_score:.1f} points</strong> on a 5-point scale. "
        "And it isn't a mild dip: the distribution on the right shows late deliveries produce a "
        "<strong>spike at 1 star</strong>, while on-time orders skew hard toward 5. A late delivery frequently "
        "triggers an angry review — and those customers rarely come back."
    )

    st.space()

    # ── Lateness buckets ──────────────────────────────────────
    st.subheader("Score vs Lateness Severity")
    st.caption("The longer the delay, the steeper the satisfaction drop.")

    dls = con.execute("""
        SELECT
            CASE
                WHEN days_late <= 0  THEN '0 – On Time'
                WHEN days_late <= 3  THEN '1–3 days late'
                WHEN days_late <= 7  THEN '4–7 days late'
                WHEN days_late <= 14 THEN '8–14 days late'
                ELSE '15+ days late'
            END AS lateness_bucket,
            COUNT(*) AS order_count,
            ROUND(AVG(review_score), 2) AS avg_score
        FROM (
            SELECT r.review_score,
                   DATEDIFF('day', o.order_estimated_delivery_date,
                       o.order_delivered_customer_date) AS days_late
            FROM orders_tbl o
            JOIN reviews_tbl r ON o.order_id = r.order_id
            WHERE o.order_status = 'delivered'
              AND o.order_delivered_customer_date IS NOT NULL
              AND o.order_estimated_delivery_date IS NOT NULL
              AND o.order_delivered_customer_date >= o.order_purchase_timestamp
        )
        GROUP BY lateness_bucket
    """).df()

    bucket_order = ['0 – On Time','1–3 days late','4–7 days late','8–14 days late','15+ days late']
    dls['lateness_bucket'] = pd.Categorical(dls['lateness_bucket'], categories=bucket_order, ordered=True)
    dls = dls.sort_values('lateness_bucket')
    dls['color'] = dls['lateness_bucket'].apply(lambda x: PALETTE[2] if x=='0 – On Time' else PALETTE[4])

    cols = st.columns(2, border=True)
    with cols[0]:
        st.subheader("Avg score by lateness bucket")
        bucket_chart = (
            alt.Chart(dls, title=alt.TitleParams("", anchor="start"))
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                alt.X("lateness_bucket:N", title=None, sort=bucket_order,
                      axis=alt.Axis(labelAngle=-20, labelFontSize=12)),
                alt.Y("avg_score:Q", title="Avg Review Score",
                      scale=alt.Scale(domain=[0, 5.5]),
                      axis=alt.Axis(labelFontSize=13)),
                alt.Color("color:N", scale=None, legend=None),
                tooltip=[alt.Tooltip("lateness_bucket:N", title="Lateness"),
                         alt.Tooltip("avg_score:Q", title="Avg Score", format=".2f"),
                         alt.Tooltip("order_count:Q", title="Orders", format=",")]
            )
            .properties(height=300)
        )
        labels = bucket_chart.mark_text(dy=-12, fontSize=14, fontWeight=700).encode(
            text=alt.Text("avg_score:Q", format=".2f"),
            color=alt.value("#1e293b")
        )
        st.altair_chart(bucket_chart + labels, use_container_width=True)

    with cols[1]:
        st.subheader("Summary table")
        st.dataframe(
            dls[['lateness_bucket','order_count','avg_score']].rename(columns={
                'lateness_bucket':'Lateness','order_count':'Orders','avg_score':'Avg Score'
            }),
            use_container_width=True, hide_index=True, height=300,
            column_config={
                "Orders":    st.column_config.NumberColumn(format="localized"),
                "Avg Score": st.column_config.ProgressColumn(min_value=0, max_value=5, format="%.2f"),
            }
        )

    # ── Anomaly note for 15+ days ──────────────────────────────
    score_15plus = dls[dls['lateness_bucket'] == '15+ days late']['avg_score'].values
    score_8_14   = dls[dls['lateness_bucket'] == '8–14 days late']['avg_score'].values
    if len(score_15plus) > 0 and len(score_8_14) > 0:
        if score_15plus[0] > score_8_14[0]:
            st.info(
                f"⚠️ **Data note — why does the 15+ days bucket show a higher score ({score_15plus[0]:.2f}) "
                f"than 8–14 days ({score_8_14[0]:.2f})?** "
                "This is a known pattern in e-commerce review data called **survivorship bias combined with resolution effect**. "
                "When a delivery is extremely late (15+ days), many customers either: "
                "(a) receive a refund or replacement before leaving a review — and then leave a positive review "
                "because the *resolution* was good, not the original delivery; or "
                "(b) simply give up and never leave a review at all, meaning only the less-angry customers are counted. "
                "The 8–14 day bucket captures customers who are upset enough to complain but haven't yet been resolved. "
                "This does not mean extreme lateness is acceptable — it means the review score alone understates the damage "
                "for very late orders. Churn rate (not returning to buy again) is a better measure of harm in this bucket."
            )

    sc_on  = dls['avg_score'].iloc[0]
    sc_814 = dls['avg_score'].iloc[-2]
    insight(
        f"The satisfaction drop steepens with delay. On-time orders sit at <strong>{sc_on:.2f} stars</strong>; "
        f"by 8–14 days late the average collapses to <strong>{sc_814:.2f}</strong> — well under half the scale. "
        "The damage is non-linear: going from 1 to 7 days late costs far more than going from 0 to 1. "
        "<strong>Every extra day compounds.</strong>"
    )

    st.space()

    # ── Statistical significance test ─────────────────────────
    st.subheader("Statistical Test: Is the Difference in Review Scores Real?")

    st.markdown("""
    The charts above show that late deliveries get lower scores. But before acting on this — investing in 
    faster logistics, renegotiating carrier contracts — a sensible question is: **could this difference just 
    be random noise in the data?** Maybe we happened to collect a sample where late orders got unlucky with 
    dissatisfied customers for unrelated reasons.

    A **statistical significance test** answers this question formally. It asks: if there were truly *no* 
    difference between late and on-time deliveries in reality, how likely is it that we'd still see a gap 
    this large just by chance? If that probability is very low, we can be confident the difference is real.
    """)

    with st.expander("📖 Why Mann-Whitney U — and not a simpler test?", expanded=True):
        st.markdown("""
        The most commonly known test for comparing two groups is the **t-test**. But the t-test has an 
        important assumption: it requires your data to be continuous and roughly normally distributed — 
        like heights, weights, or temperatures, where values can be any decimal and the data forms a 
        bell curve.

        **Review scores don't meet this requirement.** They are 1, 2, 3, 4, or 5 — discrete whole numbers. 
        They are *ordered* (5 is better than 4) but not truly numeric in the way a t-test expects. 
        The gap between a 1-star and a 2-star review doesn't necessarily represent the same emotional 
        difference as between a 4-star and a 5-star. This type of data is called **ordinal data**.

        The **Mann-Whitney U test** is specifically designed for ordinal data. Instead of comparing averages 
        directly, it works by *ranking* all the scores together (from lowest to highest across both groups), 
        then checking whether one group's scores tend to rank higher than the other's. It makes no assumption 
        about what the data distribution looks like, making it the correct and honest choice here.

        In short: using a t-test on review scores would be applying the wrong tool to the wrong data. 
        Mann-Whitney U is the statistically rigorous choice.
        """)

    ontime_scores = delivery_df[delivery_df['delivery_status'] == 'On Time']['review_score']
    late_scores   = delivery_df[delivery_df['delivery_status'] == 'Late']['review_score']
    u_stat, p_val = stats.mannwhitneyu(ontime_scores, late_scores, alternative='greater')
    n1, n2        = len(ontime_scores), len(late_scores)
    effect_size   = (2 * u_stat) / (n1 * n2) - 1  # rank-biserial r (+ = on-time ranks higher)
    abs_r         = abs(effect_size)
    eff_label     = ("large" if abs_r >= 0.5 else "medium" if abs_r >= 0.3
                     else "small" if abs_r >= 0.1 else "negligible")
    p_disp        = "< 1e-300" if p_val == 0 else f"{p_val:.2e}"

    col_t1, col_t2, col_t3 = st.columns(3)
    col_t1.metric("U Statistic",      f"{u_stat:,.0f}")
    col_t2.metric("p-value",          p_disp)
    col_t3.metric("Effect Size (r)",  f"{effect_size:.3f}")

    st.markdown("""
    **What do these three numbers mean?**

    - **U Statistic** is the raw output of the Mann-Whitney calculation. On its own it's not very 
      interpretable — it's used internally to compute the p-value and effect size. A larger U (relative 
      to the maximum possible) indicates on-time scores tend to rank higher than late scores.

    - **p-value** is the key number. It represents the probability that you'd see a score gap this large 
      purely by chance, if late and on-time deliveries were actually identical in reality. 
      A p-value of 0.05 means 5% chance — borderline. A p-value of 0.001 means 0.1% chance — very strong. 
      Our result here is astronomically small, essentially **zero**. This gap is not noise.

    - **Effect size (r)** tells you not just *whether* the difference is real, but *how big* it is in
      practical terms. The rank-biserial r runs from -1 to +1; a positive value means on-time orders
      rank higher. Conventional thresholds on its magnitude: 0.1 = small, 0.3 = medium, 0.5 = large.
      Our value sits in the **large** range — not a technically-significant-but-trivial difference,
      but a genuinely big one.
    """)

    if p_val < 0.001:
        insight(
            f"The Mann-Whitney U test confirms with near-certainty (p = {p_disp}) that the gap in review scores "
            f"between on-time and late deliveries is <strong>statistically real, not random</strong>. "
            f"The effect size r = {effect_size:.3f} makes this a <strong>{eff_label}</strong> effect — the damage "
            "from late deliveries is both statistically provable and practically significant, "
            "a strong, data-backed case for investing in faster logistics."
        )



# ═══════════════════════════════════════════════════════════════
# TAB 4 — CUSTOMER RETENTION
# ═══════════════════════════════════════════════════════════════
with tab_ret:
    st.space()
    st.subheader("Customer Retention")
    st.caption("What share of customers return for a second order, and how long does it take? "
               "Measured on customers with at least one *delivered* order, matching the RFM and churn tabs.")

    cust_orders = con.execute("""
        SELECT c.customer_unique_id, COUNT(DISTINCT o.order_id) AS total_orders
        FROM orders_tbl o JOIN customers_tbl c ON o.customer_id = c.customer_id
        WHERE o.order_status = 'delivered'
        GROUP BY c.customer_unique_id
    """).df()

    one_time  = int((cust_orders['total_orders'] == 1).sum())
    returning = int((cust_orders['total_orders'] > 1).sum())
    total     = len(cust_orders)

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Unique Customers", f"{total:,}")
    m2.metric("One-time Buyers",        f"{one_time:,}", delta=f"{100*one_time/total:.1f}% of base")
    m3.metric("Returning Buyers",       f"{returning:,}", delta=f"{100*returning/total:.1f}% of base")

    st.space()

    repeat_gap = con.execute("""
        WITH ranked AS (
            SELECT c.customer_unique_id, o.order_purchase_timestamp,
                   ROW_NUMBER() OVER (PARTITION BY c.customer_unique_id
                                      ORDER BY o.order_purchase_timestamp) AS order_rank
            FROM orders_tbl o JOIN customers_tbl c ON o.customer_id = c.customer_id
            WHERE o.order_status = 'delivered'
        ),
        first_second AS (
            SELECT r1.customer_unique_id,
                   DATEDIFF('day', r1.order_purchase_timestamp, r2.order_purchase_timestamp) AS days_to_return
            FROM ranked r1
            JOIN ranked r2
              ON r1.customer_unique_id = r2.customer_unique_id
             AND r1.order_rank = 1 AND r2.order_rank = 2
        )
        SELECT * FROM first_second WHERE days_to_return > 0
    """).df()

    median_days = int(repeat_gap['days_to_return'].median())
    mean_days   = int(repeat_gap['days_to_return'].mean())

    freq_df = pd.DataFrame({
        'Segment': ['One-time','Returning'],
        'Count':   [one_time, returning],
        'pct':     [round(100*one_time/total,1), round(100*returning/total,1)]
    })

    cols = st.columns(2, border=True)
    with cols[0]:
        st.subheader("Purchase frequency breakdown")
        donut = (
            alt.Chart(freq_df, title=alt.TitleParams("", anchor="start"))
            .mark_arc(innerRadius=80, outerRadius=150)
            .encode(
                alt.Theta("Count:Q"),
                alt.Color("Segment:N",
                          scale=alt.Scale(domain=['One-time','Returning'],
                                          range=[PALETTE[1], PALETTE[0]])),
                tooltip=[alt.Tooltip("Segment:N", title="Segment"),
                         alt.Tooltip("Count:Q", title="Customers", format=","),
                         alt.Tooltip("pct:Q", title="% of Base", format=".1f")]
            )
            .properties(height=340)
        )
        st.altair_chart(donut, use_container_width=True)

    with cols[1]:
        st.subheader("When do returning customers come back?")
        st.markdown(
            "The chart below shows **how many days pass** between a customer's first and second purchase. "
            "Each bar represents a group of customers who returned within that time window. "
            "The orange dashed line marks the median — half of all returning customers come back before this point, "
            "half come back after."
        )
        clip_val = st.slider(
            "Show returns within (days)",
            90, 730, 365, step=30,
            help="Drag to focus on a shorter or longer window. Customers who took longer than this are hidden to keep the chart readable."
        )
        clipped = repeat_gap[repeat_gap['days_to_return'] <= clip_val].copy()

        hist = (
            alt.Chart(clipped, title=alt.TitleParams("", anchor="start"))
            .mark_bar(color=PALETTE[0], opacity=0.8, binSpacing=1)
            .encode(
                alt.X("days_to_return:Q", title="Days Between 1st and 2nd Order",
                      bin=alt.Bin(maxbins=40), axis=alt.Axis(labelFontSize=13)),
                alt.Y("count():Q", title="Number of Customers", axis=alt.Axis(format=",d", labelFontSize=13)),
                tooltip=[alt.Tooltip("days_to_return:Q", title="Days (bin start)", bin="binned"),
                         alt.Tooltip("count():Q", title="Customers", format=",")]
            )
            .properties(height=260)
        )
        med_rule = (
            alt.Chart(pd.DataFrame({'median': [median_days]}))
            .mark_rule(color=PALETTE[3], strokeDash=[6,3], strokeWidth=2.5)
            .encode(x="median:Q")
        )
        st.altair_chart(hist + med_rule, use_container_width=True)
        st.caption(
            f"Median return gap: **{median_days} days** · "
            f"Mean: **{mean_days} days** · "
            f"Customers shown: **{len(clipped):,}** (out of {len(repeat_gap):,} total returners)"
        )

    st.space()

    # ── Cumulative return curve ───────────────────────────────
    st.subheader("Cumulative Return Rate Over Time")
    st.markdown(
        "This chart answers a more actionable question: **by day X after their first purchase, "
        "what percentage of all returning customers have already come back?** "
        "It's read like a progress bar — if the line hits 50% at day 120, it means half of all "
        "customers who will ever return have done so within 120 days. "
        "This is critical for deciding *when* to send re-engagement campaigns — too early and the "
        "customer wasn't thinking about buying again yet; too late and they've already moved on."
    )

    sorted_gaps = np.sort(repeat_gap['days_to_return'].values)
    cumulative_pct = np.arange(1, len(sorted_gaps) + 1) / len(sorted_gaps) * 100
    cum_df = pd.DataFrame({'days': sorted_gaps, 'cumulative_pct': cumulative_pct})
    cum_df = cum_df[cum_df['days'] <= 500]

    pct_30  = (repeat_gap['days_to_return'] <= 30).sum()  / len(repeat_gap) * 100
    pct_90  = (repeat_gap['days_to_return'] <= 90).sum()  / len(repeat_gap) * 100
    pct_180 = (repeat_gap['days_to_return'] <= 180).sum() / len(repeat_gap) * 100

    cum_line = (
        alt.Chart(cum_df)
        .mark_line(color=PALETTE[0], strokeWidth=2.5)
        .encode(
            alt.X("days:Q", title="Days After First Purchase", axis=alt.Axis(labelFontSize=13)),
            alt.Y("cumulative_pct:Q", title="% of Returning Customers Who Have Returned",
                  axis=alt.Axis(format=".0f", labelFontSize=13)),
            tooltip=[
                alt.Tooltip("days:Q", title="Day"),
                alt.Tooltip("cumulative_pct:Q", title="Cumulative % returned", format=".1f"),
            ]
        )
        .properties(height=300)
    )

    milestone_df = pd.DataFrame({
        'days': [30, 90, 180],
        'label': [
            f"{pct_30:.0f}% by day 30",
            f"{pct_90:.0f}% by day 90",
            f"{pct_180:.0f}% by day 180",
        ],
        'cumulative_pct': [pct_30, pct_90, pct_180]
    })
    milestones = (
        alt.Chart(milestone_df)
        .mark_point(size=120, filled=True, color=PALETTE[3])
        .encode(
            alt.X("days:Q"),
            alt.Y("cumulative_pct:Q"),
            tooltip=["label:N"]
        )
    )
    milestone_labels = milestones.mark_text(dy=-14, fontSize=12, fontWeight=600, color=PALETTE[3]).encode(
        text="label:N"
    )
    st.altair_chart(cum_line + milestones + milestone_labels, use_container_width=True)

    insight(
        f"<strong>{100*one_time/total:.0f}% of Olist customers never make a second purchase</strong> — this is the "
        f"single biggest commercial problem visible in this dataset. Among the small minority who do return, "
        f"the behaviour is spread out: only <strong>{pct_30:.0f}%</strong> come back within 30 days, "
        f"<strong>{pct_90:.0f}%</strong> within 90 days, and <strong>{pct_180:.0f}%</strong> within 180 days. "
        f"The median gap is <strong>{median_days} days</strong>. This means if you want to intervene and bring "
        "a customer back, you have a narrow window — most customers who will ever return do so within the "
        "first 3–6 months. After that, the probability of re-engagement drops sharply. "
        "A structured re-engagement email or discount sent at day 14, day 30, and day 60 after the first "
        "purchase would target the window where customers are still warm and most likely to respond."
    )

# ═══════════════════════════════════════════════════════════════
# TAB 5 — RFM SEGMENTATION
# ═══════════════════════════════════════════════════════════════
with tab_rfm:
    st.space()
    st.subheader("RFM Customer Segmentation")

    st.markdown("""
    **RFM** stands for **Recency**, **Frequency**, and **Monetary value** — three dimensions that together 
    describe how valuable a customer is to the business.

    - **Recency**: How recently did this customer last make a purchase? A customer who bought last week is 
      more likely to buy again than one who bought two years ago.
    - **Frequency**: How many times has this customer bought in total? Someone who has ordered 5 times 
      is more engaged than someone who ordered once.
    - **Monetary**: How much has this customer spent in total? High spenders are worth more to the business 
      and may deserve different treatment.

    RFM is one of the most widely used frameworks in e-commerce and retail analytics because it is simple, 
    interpretable, and directly actionable — each segment leads naturally to a different marketing strategy.
    """)

    with st.expander("📖 How are the scores and segments calculated?", expanded=True):
        st.markdown("""
        **Step 1 — Scoring each customer 1 to 5:**

        Every customer is ranked on each of the three dimensions using **quintile scoring**. 
        Quintile means we divide all customers into 5 equal-sized groups (20% each). The top 20% 
        spenders get an M score of 5; the bottom 20% get 1. The same applies to Frequency and Recency. 
        For Recency, the score is inverted — a *lower* number of days since last purchase is *better*, 
        so we flip the scale so that 5 = most recent.

        **Step 2 — Assigning segments using business rules:**

        The three scores are then combined using a set of hand-written rules to assign each customer 
        to a named segment. These rules are based on standard industry logic:

        | Segment | Rule |
        |---|---|
        | Champions | R ≥ 4, F ≥ 4, M ≥ 4 — bought recently, often, and a lot |
        | Loyal Customers | R ≥ 3, F ≥ 3 — consistent buyers, still active |
        | New Customers | R ≥ 4, F ≤ 2 — bought recently but not yet a habit |
        | Potential Loyalists | Everything in between |
        | At Risk | R ≤ 2, F ≥ 3 — used to buy often but haven't recently |
        | Lost | R ≤ 2, F ≤ 2 — low engagement, not active |

        **Important methodological note:** This is a **rule-based** segmentation, not a machine learning 
        clustering algorithm. The boundaries (e.g. "R ≥ 4") are chosen by the analyst based on business 
        logic, not discovered automatically by the data. An alternative approach would be **K-means clustering**, 
        which would let the data decide where the natural groupings are. The tradeoff is interpretability: 
        rule-based RFM gives you segments with clear, actionable names; K-means might give you more 
        statistically natural clusters but harder-to-explain labels. For a business audience, rule-based 
        RFM almost always wins.

        **Why not PCA?** PCA (Principal Component Analysis) is used to *compress* many dimensions into fewer. 
        RFM only has 3 dimensions — there's nothing to compress. PCA would also destroy the interpretability 
        of the axes, replacing "Recency" and "Monetary" with abstract mathematical components. Not useful here.
        """)

    @st.cache_data
    def compute_rfm():
        rfm_raw = con.execute("""
            SELECT
                c.customer_unique_id,
                DATEDIFF('day',
                    MAX(o.order_purchase_timestamp),
                    (SELECT MAX(order_purchase_timestamp) FROM orders_tbl)
                ) AS recency_days,
                COUNT(DISTINCT o.order_id)  AS frequency,
                SUM(p.payment_value)         AS monetary
            FROM orders_tbl o
            JOIN customers_tbl c ON o.customer_id = c.customer_id
            JOIN payments_tbl  p ON o.order_id    = p.order_id
            WHERE o.order_status = 'delivered'
            GROUP BY c.customer_unique_id
        """).df()

        for col, ascending in [('recency_days', False), ('frequency', True), ('monetary', True)]:
            label = col[0].upper()
            _, bins = pd.qcut(rfm_raw[col], q=5, retbins=True, duplicates='drop')
            n_bins = len(bins) - 1
            bin_labels = list(range(1, n_bins + 1))
            rfm_raw[f'{label}_score'] = pd.qcut(
                rfm_raw[col], q=5, labels=bin_labels, duplicates='drop'
            ).astype(int)
            if col == 'recency_days':
                max_score = rfm_raw['R_score'].max()
                rfm_raw['R_score'] = max_score + 1 - rfm_raw['R_score']

        rfm_raw['RFM_score'] = rfm_raw[['R_score','F_score','M_score']].sum(axis=1)

        def segment(row):
            r, f, m = row['R_score'], row['F_score'], row['M_score']
            if r >= 4 and f >= 4 and m >= 4:
                return 'Champions'
            elif r >= 3 and f >= 3:
                return 'Loyal Customers'
            elif r >= 4 and f <= 2:
                return 'New Customers'
            elif r <= 2 and f >= 3:
                return 'At Risk'
            elif r <= 2 and f <= 2:
                return 'Lost'
            else:
                return 'Potential Loyalists'

        rfm_raw['Segment'] = rfm_raw.apply(segment, axis=1)
        return rfm_raw

    rfm = compute_rfm()

    seg_summary = rfm.groupby('Segment').agg(
        Customers    =('customer_unique_id', 'count'),
        Avg_Recency  =('recency_days', 'mean'),
        Avg_Frequency=('frequency', 'mean'),
        Avg_Monetary =('monetary', 'mean'),
    ).round(1).reset_index()
    seg_summary['% of Base'] = (seg_summary['Customers'] / seg_summary['Customers'].sum() * 100).round(1)

    SEGMENT_COLORS = {
        'Champions':           '#10b981',
        'Loyal Customers':     '#5b6ef5',
        'New Customers':       '#0ea5e9',
        'Potential Loyalists': '#f97316',
        'At Risk':             '#ef4444',
        'Lost':                '#94a3b8',
    }

    cols = st.columns(2, border=True)
    with cols[0]:
        st.subheader("How many customers are in each segment?")
        seg_bar = (
            alt.Chart(seg_summary, title=alt.TitleParams("", anchor="start"))
            .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
            .encode(
                alt.Y("Segment:N", sort="-x", axis=alt.Axis(labelFontSize=13)),
                alt.X("Customers:Q", axis=alt.Axis(format=",d", labelFontSize=13)),
                alt.Color("Segment:N",
                          scale=alt.Scale(domain=list(SEGMENT_COLORS.keys()),
                                          range=list(SEGMENT_COLORS.values())), legend=None),
                tooltip=[alt.Tooltip("Segment:N"),
                         alt.Tooltip("Customers:Q", format=","),
                         alt.Tooltip("% of Base:Q", format=".1f", title="% of Customers")]
            )
            .properties(height=300)
        )
        st.altair_chart(seg_bar, use_container_width=True)

    with cols[1]:
        st.subheader("How much does each segment spend on average?")
        mon_bar = (
            alt.Chart(seg_summary, title=alt.TitleParams("", anchor="start"))
            .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
            .encode(
                alt.Y("Segment:N", sort="-x", axis=alt.Axis(labelFontSize=13)),
                alt.X("Avg_Monetary:Q", title="Avg Total Spend (R$)",
                      axis=alt.Axis(format=",.0f", labelFontSize=13)),
                alt.Color("Segment:N",
                          scale=alt.Scale(domain=list(SEGMENT_COLORS.keys()),
                                          range=list(SEGMENT_COLORS.values())), legend=None),
                tooltip=[alt.Tooltip("Segment:N"),
                         alt.Tooltip("Avg_Monetary:Q", title="Avg Spend (R$)", format=",.2f"),
                         alt.Tooltip("Avg_Recency:Q",  title="Avg Days Since Last Purchase", format=".0f")]
            )
            .properties(height=300)
        )
        st.altair_chart(mon_bar, use_container_width=True)

    st.space()

    # ── Scatter: Recency vs Monetary coloured by segment ──────
    st.subheader("Customer Map: Recency vs. Spend by Segment")
    st.markdown(
        "Each dot below represents one customer, plotted by how recently they last purchased (x-axis) "
        "and how much they spent in total (y-axis), coloured by their RFM segment. "
        "This gives you a visual intuition for where the segments sit relative to each other. "
        "**New Customers** (blue) sit toward the left — recent buyers, almost all with just a single order. "
        "**Potential Loyalists** (orange) fall in the middle, and **Lost** customers (grey) sit to the right: "
        "haven't bought in a long time. Because nearly every customer has exactly one order, the segments "
        "separate left-to-right by *recency* rather than by frequency. "
        "A random sample of 3,000 customers is shown to keep the chart readable."
    )

    rfm_sample = rfm.sample(min(3000, len(rfm)), random_state=42)
    scatter = (
        alt.Chart(rfm_sample)
        .mark_circle(size=25, opacity=0.5)
        .encode(
            alt.X("recency_days:Q", title="Days Since Last Purchase (lower = more recent)",
                  axis=alt.Axis(labelFontSize=12)),
            alt.Y("monetary:Q", title="Total Spend (R$)",
                  scale=alt.Scale(type='log'),
                  axis=alt.Axis(labelFontSize=12, format=",.0f")),
            alt.Color("Segment:N",
                      scale=alt.Scale(domain=list(SEGMENT_COLORS.keys()),
                                      range=list(SEGMENT_COLORS.values())),
                      legend=alt.Legend(title="Segment", labelFontSize=12)),
            tooltip=[alt.Tooltip("Segment:N"),
                     alt.Tooltip("recency_days:Q", title="Days Since Last Purchase"),
                     alt.Tooltip("monetary:Q",     title="Total Spend (R$)", format=",.2f"),
                     alt.Tooltip("frequency:Q",    title="Total Orders")]
        )
        .properties(height=420)
    )
    st.altair_chart(scatter, use_container_width=True)

    st.space()
    st.subheader("Segment Summary Table")
    st.dataframe(
        seg_summary.sort_values('Customers', ascending=False),
        use_container_width=True, hide_index=True,
        column_config={
            "Customers":      st.column_config.NumberColumn(format="localized"),
            "% of Base":      st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
            "Avg_Recency":    st.column_config.NumberColumn("Avg Days Since Last Purchase", format="%.0f days"),
            "Avg_Frequency":  st.column_config.NumberColumn("Avg Orders", format="%.1f"),
            "Avg_Monetary":   st.column_config.NumberColumn("Avg Total Spend (R$)", format="R$ %.0f"),
        }
    )

    seg_pct       = dict(zip(seg_summary['Segment'], seg_summary['% of Base']))
    present_segs  = set(seg_summary['Segment'])
    empty_segs    = [s for s in ('Champions', 'Loyal Customers', 'At Risk')
                     if s not in present_segs]
    empty_names   = ', '.join(empty_segs) if empty_segs else 'the high-frequency tiers'
    one_order_pct = 100 * (rfm['frequency'] == 1).mean()
    new_pct  = seg_pct.get('New Customers', 0.0)
    pot_pct  = seg_pct.get('Potential Loyalists', 0.0)
    lost_pct = seg_pct.get('Lost', 0.0)

    insight(
        f"On this dataset the rules produce only three populated segments: "
        f"<strong>New Customers</strong> ({new_pct:.1f}%), <strong>Potential Loyalists</strong> ({pot_pct:.1f}%), "
        f"and <strong>Lost</strong> ({lost_pct:.1f}%). "
        f"The frequency-based tiers — <strong>{empty_names}</strong> — come out empty, and that is a property of "
        f"the data rather than a bug: about {one_order_pct:.0f}% of customers place exactly one order, so the "
        f"Frequency score is essentially constant for everyone and can never clear the F ≥ 3 / F ≥ 4 thresholds "
        f"those segments require. With frequency carrying almost no signal, the only dimension that meaningfully "
        f"separates customers here is <strong>Recency</strong>. "
        f"<br><br>"
        f"That reshapes the playbook. There is no genuine 'used to buy often, now lapsed' group to win back, so the "
        f"priority is recency-driven: reach <strong>New Customers</strong> while they are still warm — a "
        f"day-14 / 30 / 60 re-engagement nudge — to convert first-time buyers into repeat ones before they drift "
        f"into <strong>Lost</strong>. Winning back the Lost segment is far more expensive and rarely pays off, so it "
        f"is the lowest-priority group, not the highest."
    )


# ═══════════════════════════════════════════════════════════════
# TAB 6 — CHURN PREDICTION (ML)
# ═══════════════════════════════════════════════════════════════
with tab_ml:
    st.space()
    st.subheader("Churn Prediction Model")

    st.markdown("""
    This section uses **machine learning** to predict whether a customer will ever make a second purchase — 
    or whether they will churn (leave after just one order). Being able to predict this *before* a customer 
    churns gives the business time to intervene: send a targeted discount, a re-engagement email, or a 
    personalised product recommendation at exactly the right moment.
    """)

    with st.expander("📖 How was this prediction problem set up?", expanded=True):
        st.markdown("""
        **Step 1 — Defining 'churn':**

        The Olist dataset doesn't come with a "churned" column. We had to define it ourselves. 
        The definition used here is: **a customer churned if they never made a second purchase**. 
        Label 0 = churned (one-time buyer). Label 1 = retained (came back at least once).

        We look only at each customer's *first order*, and we score the model **once that first order has been
        delivered and reviewed** — the natural moment to decide whether to spend on winning the customer back.
        Every feature below is known by that point; none of it peeks at any later order (which would leak the answer).

        **Step 2 — Features used to predict churn:**

        | Feature | What it captures |
        |---|---|
        | Order value (R$) | How much they spent on their first purchase |
        | Installments | Whether they split the payment (signals commitment) |
        | Review score | How satisfied they were — intuitive, but a *weak* predictor here (see importances below) |
        | Payment type | Credit card vs. other (credit card users tend to be higher-value) |
        | Days late | Whether the first delivery was late, and by how much |

        **Step 3 — The class imbalance problem:**

        Approximately **97% of customers never return**. This creates a severe imbalance:
        for every 1 retained customer, there are roughly 32 churned customers. A naive model will simply
        predict "churned" for everyone and be ~97% accurate — but completely useless, because it never
        identifies anyone worth targeting. This is the same problem that appears in fraud detection
        (99% of transactions are legitimate) and disease diagnosis (most patients are healthy).

        We test **two strategies** to fix this, explained below.
        """)

    with st.expander("⚖️ How we handle class imbalance: two approaches compared", expanded=True):
        st.markdown("""
        **Approach 1 — Class Weighting (`class_weight='balanced'`)**

        Instead of changing the data, we change how the model *learns from it*. We tell the model: 
        "a mistake on a retained customer is ~32× more costly than a mistake on a churned customer."
        The model then works harder to correctly identify the minority class. No data is thrown away — 
        the model just penalises minority-class errors more heavily during training.

        ✅ Keeps all data  
        ✅ Simple to implement  
        ⚠️ Doesn't add new information — just reweights existing data

        ---

        **Approach 2 — Undersampling the majority class**

        We randomly remove churned customers from the training set until the classes are balanced 
        (50/50). The model then trains on a balanced dataset and doesn't develop a bias toward 
        predicting "churned" for everyone.

        ✅ Creates a truly balanced training set  
        ✅ Sometimes produces better recall on the minority class  
        ⚠️ Throws away real data — we discard ~97% of churned customer records
        ⚠️ Works best when the majority class is very large (millions of records). In our case, 
        after undersampling we only train on ~4,000–6,000 total records, which may limit model quality.

        **The fraud detection comparison:** In fraud datasets with millions of transactions, 
        undersampling is extremely effective because even after discarding 90% of non-fraud cases, 
        you still have tens of thousands of them — more than enough. In our dataset, we have ~93,000
        customers but only ~2,800 returning ones. Undersampling brings us down to ~5,600 total
        training records, which is quite small. Class weighting tends to win here, but we show both
        so you can compare directly.
        """)

    @st.cache_data
    def build_ml_dataset():
        df = con.execute("""
            WITH pay AS (
                SELECT
                    order_id,
                    SUM(payment_value)                                           AS payment_value,
                    MAX(payment_installments)                                    AS payment_installments,
                    MAX(CASE WHEN payment_type = 'credit_card' THEN 1 ELSE 0 END) AS is_credit_card
                FROM payments_tbl
                GROUP BY order_id
            ),
            rev AS (
                SELECT order_id, AVG(review_score) AS review_score
                FROM reviews_tbl
                GROUP BY order_id
            ),
            customer_orders AS (
                SELECT
                    c.customer_unique_id,
                    o.order_id,
                    o.order_purchase_timestamp,
                    o.order_delivered_customer_date,
                    o.order_estimated_delivery_date,
                    pay.payment_value,
                    pay.payment_installments,
                    pay.is_credit_card,
                    rev.review_score,
                    EXTRACT(month FROM o.order_purchase_timestamp) AS purchase_month,
                    ROW_NUMBER() OVER (
                        PARTITION BY c.customer_unique_id
                        ORDER BY o.order_purchase_timestamp
                    ) AS order_rank,
                    COUNT(DISTINCT o.order_id) OVER (PARTITION BY c.customer_unique_id) AS total_orders
                FROM orders_tbl o
                JOIN customers_tbl c  ON o.customer_id = c.customer_id
                JOIN pay              ON o.order_id    = pay.order_id
                LEFT JOIN rev         ON o.order_id    = rev.order_id
                WHERE o.order_status = 'delivered'
            )
            SELECT
                customer_unique_id,
                payment_value,
                payment_installments,
                COALESCE(review_score, 3)                    AS review_score,
                is_credit_card,
                CASE
                    WHEN order_delivered_customer_date > order_estimated_delivery_date
                    THEN DATEDIFF('day', order_estimated_delivery_date, order_delivered_customer_date)
                    ELSE 0
                END                                          AS days_late,
                purchase_month,
                CASE WHEN total_orders > 1 THEN 1 ELSE 0 END AS label
            FROM customer_orders
            WHERE order_rank = 1
        """).df()
        return df

    ml_df = build_ml_dataset()

    features = ['payment_value', 'payment_installments', 'review_score',
                'is_credit_card', 'days_late', 'purchase_month']
    target   = 'label'

    X = ml_df[features].values
    y = ml_df[target].values

    feature_labels = ['Order Value (R$)', 'Installments', 'Review Score',
                      'Credit Card', 'Days Late', 'Purchase Month']

    @st.cache_data
    def train_all_models(X_data, y_data):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_data, y_data, test_size=0.2, random_state=42, stratify=y_data
        )
        scaler   = StandardScaler()
        X_tr_s   = scaler.fit_transform(X_tr)
        X_te_s   = scaler.transform(X_te)

        # ── Undersampled training set ─────────────────────────
        tr_df    = pd.DataFrame(X_tr, columns=features)
        tr_df['label'] = y_tr
        majority = tr_df[tr_df['label'] == 0]
        minority = tr_df[tr_df['label'] == 1]
        maj_down = resample(majority, replace=False, n_samples=len(minority), random_state=42)
        balanced = pd.concat([maj_down, minority]).sample(frac=1, random_state=42)
        X_tr_us  = balanced[features].values
        y_tr_us  = balanced['label'].values
        X_tr_us_s = scaler.transform(X_tr_us)

        results = {}

        # 1. LR with class_weight
        lr_cw = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        lr_cw.fit(X_tr_s, y_tr)
        _store(results, 'Logistic Regression\n(Class Weighting)', lr_cw, X_tr_s, X_te_s, y_te)

        # 2. LR with undersampling
        lr_us = LogisticRegression(max_iter=1000, random_state=42)
        lr_us.fit(X_tr_us_s, y_tr_us)
        _store(results, 'Logistic Regression\n(Undersampling)', lr_us, X_tr_us_s, X_te_s, y_te)

        # 3. GB with class_weight
        gb_cw = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                            learning_rate=0.05, subsample=0.8, random_state=42)
        # GB doesn't have class_weight; use sample_weight instead
        sw = np.where(y_tr == 1, len(y_tr) / (2 * (y_tr == 1).sum()),
                                  len(y_tr) / (2 * (y_tr == 0).sum()))
        gb_cw.fit(X_tr, y_tr, sample_weight=sw)
        _store(results, 'Gradient Boosting\n(Class Weighting)', gb_cw, X_tr, X_te, y_te)

        # 4. GB with undersampling
        gb_us = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                            learning_rate=0.05, subsample=0.8, random_state=42)
        gb_us.fit(X_tr_us, y_tr_us)
        _store(results, 'Gradient Boosting\n(Undersampling)', gb_us, X_tr_us, X_te, y_te)

        fi = pd.DataFrame({
            'Feature':    feature_labels,
            'Importance': gb_cw.feature_importances_
        }).sort_values('Importance', ascending=False)

        return results, fi, scaler, gb_cw

    def _store(results, name, model, Xtr, Xte, y_te):
        y_pred = model.predict(Xte)
        y_prob = model.predict_proba(Xte)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, y_prob)
        rep = classification_report(y_te, y_pred, output_dict=True, zero_division=0)
        results[name] = {
            'auc':    roc_auc_score(y_te, y_prob),
            'report': rep,
            'fpr':    fpr,
            'tpr':    tpr,
            'y_te':   y_te,
            'y_pred': y_pred,
        }

    results, feat_imp, scaler, gb_model = train_all_models(X, y)

    # ── Model comparison table ────────────────────────────────
    st.subheader("Model Performance Comparison")
    st.markdown(
        "Four models are trained — two algorithms × two imbalance strategies. "
        "Use the table and selector below to compare them and understand the tradeoffs."
    )

    comp_rows = []
    for name, res in results.items():
        rep = res['report']
        comp_rows.append({
            'Model': name.replace('\n', ' '),
            'ROC-AUC': round(res['auc'], 3),
            'Precision (Retained)': round(rep.get('1', {}).get('precision', 0), 3),
            'Recall (Retained)':    round(rep.get('1', {}).get('recall', 0), 3),
            'F1 (Retained)':        round(rep.get('1', {}).get('f1-score', 0), 3),
        })
    comp_df = pd.DataFrame(comp_rows)
    st.dataframe(comp_df, use_container_width=True, hide_index=True,
                 column_config={
                     'ROC-AUC':               st.column_config.NumberColumn(format="%.3f"),
                     'Precision (Retained)':  st.column_config.NumberColumn(format="%.3f"),
                     'Recall (Retained)':     st.column_config.NumberColumn(format="%.3f"),
                     'F1 (Retained)':         st.column_config.NumberColumn(format="%.3f"),
                 })

    with st.expander("📖 What do these metrics mean?", expanded=False):
        st.markdown("""
        - **ROC-AUC**: Overall model quality on a 0.5–1.0 scale. 0.5 = random guessing. 1.0 = perfect. 
          Above 0.65 is meaningful for this kind of imbalanced problem.
        - **Precision (Retained)**: Of all customers the model *predicted* would return, what fraction 
          actually did? High precision = fewer false alarms (fewer wasted discount coupons sent to 
          customers who wouldn't have returned anyway).
        - **Recall (Retained)**: Of all customers who *actually* returned, what fraction did the model 
          correctly identify? High recall = fewer missed opportunities (fewer returners wrongly written off).
        - **F1 (Retained)**: The harmonic mean of Precision and Recall. A single score that balances both. 
          Use this as your primary metric when you care about both false alarms and missed opportunities.
        """)

    # ── Model selector ────────────────────────────────────────
    model_names = list(results.keys())
    display_names = [n.replace('\n', ' ') for n in model_names]
    model_choice_display = st.segmented_control(
        "Inspect model in detail", display_names, default=display_names[2]
    )
    if not model_choice_display:
        model_choice_display = display_names[2]
    model_choice = model_names[display_names.index(model_choice_display)]

    res = results[model_choice]
    rep = res['report']

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROC-AUC",              f"{res['auc']:.3f}")
    m2.metric("Precision (Retained)", f"{rep.get('1',{}).get('precision',0):.3f}")
    m3.metric("Recall (Retained)",    f"{rep.get('1',{}).get('recall',0):.3f}")
    m4.metric("F1 (Retained)",        f"{rep.get('1',{}).get('f1-score',0):.3f}")

    st.space()

    cols = st.columns(2, border=True)
    with cols[0]:
        st.subheader("ROC Curve")
        st.markdown(
            "The ROC curve shows the tradeoff between catching real returners (True Positive Rate, y-axis) "
            "and falsely flagging churned customers as returners (False Positive Rate, x-axis). "
            "The diagonal dashed line = random guessing. The more the curve bows toward the top-left corner, "
            "the better the model."
        )
        roc_df = pd.DataFrame({'FPR': res['fpr'], 'TPR': res['tpr']})
        roc_line = (
            alt.Chart(roc_df)
            .mark_line(color=PALETTE[0], strokeWidth=2.5)
            .encode(
                alt.X("FPR:Q", title="False Positive Rate", axis=alt.Axis(format=".1f")),
                alt.Y("TPR:Q", title="True Positive Rate",  axis=alt.Axis(format=".1f")),
                tooltip=[alt.Tooltip("FPR:Q", format=".3f"), alt.Tooltip("TPR:Q", format=".3f")]
            )
            .properties(height=300)
        )
        diag = (
            alt.Chart(pd.DataFrame({'x': [0,1], 'y': [0,1]}))
            .mark_line(strokeDash=[5,3], color='#94a3b8', strokeWidth=1.5)
            .encode(alt.X("x:Q"), alt.Y("y:Q"))
        )
        st.altair_chart(roc_line + diag, use_container_width=True)
        st.caption(f"AUC = {res['auc']:.3f}")

    with cols[1]:
        st.subheader("Feature Importance (Gradient Boosting — Class Weighting)")
        st.markdown(
            "This shows which features the Gradient Boosting model relies on most when deciding "
            "whether a customer will return. A higher bar = more influential. "
            "These are derived from the class-weighted model, which is the best overall performer."
        )
        fi_chart = (
            alt.Chart(feat_imp)
            .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
            .encode(
                alt.Y("Feature:N", sort="-x", axis=alt.Axis(labelFontSize=13)),
                alt.X("Importance:Q", axis=alt.Axis(format=".3f", labelFontSize=13)),
                alt.Color("Feature:N", scale=alt.Scale(range=PALETTE), legend=None),
                tooltip=[alt.Tooltip("Feature:N"), alt.Tooltip("Importance:Q", format=".4f")]
            )
            .properties(height=300)
        )
        st.altair_chart(fi_chart, use_container_width=True)

    st.space()

    # ── Confusion matrix ──────────────────────────────────────
    st.subheader("Confusion Matrix")
    st.markdown(
        "A confusion matrix shows exactly where the model is right and wrong. "
        "Each cell tells you how many customers fell into that combination of actual vs. predicted outcome. "
        "The ideal matrix has large numbers in the top-left (correctly predicted churned) and "
        "bottom-right (correctly predicted retained), and small numbers in the off-diagonal cells."
    )
    cm = confusion_matrix(res['y_te'], res['y_pred'])
    cm_df = pd.DataFrame(
        cm,
        index=['Actual: Churned','Actual: Retained'],
        columns=['Predicted: Churned','Predicted: Retained']
    ).reset_index().melt(id_vars='index', var_name='Predicted', value_name='Count')
    cm_df.columns = ['Actual','Predicted','Count']

    cm_chart = (
        alt.Chart(cm_df).mark_rect()
        .encode(
            alt.X("Predicted:N", axis=alt.Axis(labelAngle=-20, labelFontSize=13)),
            alt.Y("Actual:N",    axis=alt.Axis(labelFontSize=13)),
            alt.Color("Count:Q", scale=alt.Scale(scheme="blues")),
            tooltip=["Actual:N","Predicted:N", alt.Tooltip("Count:Q", format=",")]
        )
        .properties(height=220, width=420)
    )
    cm_text = cm_chart.mark_text(fontSize=18, fontWeight=700).encode(
        text=alt.Text("Count:Q", format=","),
        color=alt.condition(alt.datum.Count > cm.max()/2, alt.value("white"), alt.value("#1e293b"))
    )
    st.altair_chart(cm_chart + cm_text)

    st.space()

    # ── Live predictor ────────────────────────────────────────
    st.subheader("🔮 Try It: Will This Customer Return?")
    st.markdown(
        "Use the sliders below to describe a hypothetical customer's first order. "
        "The best model (Gradient Boosting with class weighting) will instantly estimate "
        "the probability that this customer will make a second purchase."
    )

    p1, p2, p3 = st.columns(3)
    with p1:
        inp_value        = st.slider("Order Value (R$)", 10, 1000, 150, step=10)
        inp_installments = st.slider("Installments",     1, 12,    1)
    with p2:
        inp_review       = st.slider("Review Score",     1, 5,     4)
        inp_credit       = st.selectbox("Payment Type", ["Credit Card", "Other"])
    with p3:
        inp_late         = st.slider("Days Late",        0, 30,    0)
        inp_month        = st.slider("Purchase Month",   1, 12,    6)

    inp_vec = np.array([[
        inp_value, inp_installments, inp_review,
        1 if inp_credit == "Credit Card" else 0,
        inp_late, inp_month
    ]])
    prob_return = gb_model.predict_proba(inp_vec)[0][1]
    prob_churn  = 1 - prob_return

    rc1, rc2 = st.columns(2)
    rc1.metric("Probability of Returning", f"{prob_return:.1%}")
    rc2.metric("Probability of Churning",  f"{prob_churn:.1%}")

    if prob_return >= 0.4:
        insight(
            f"This customer profile has a <strong>{prob_return:.1%} likelihood of returning</strong>. "
            "They are relatively likely to come back on their own — but a light-touch follow-up "
            "(e.g. a product recommendation email at day 30) could still accelerate their return."
        )
    else:
        insight(
            f"This customer profile carries a <strong>{prob_churn:.1%} churn risk</strong>. "
            "The model flags them as unlikely to return without intervention. "
            "A targeted re-engagement offer — a personalised discount, free shipping on the next order, "
            "or a curated product recommendation — sent within 14–30 days of their first purchase "
            "would be the highest-priority action for this type of customer."
        )

    best_auc = max(v['auc'] for v in results.values())
    best_model_name = [k for k, v in results.items() if v['auc'] == best_auc][0].replace('\n', ' ')
    top_feats   = feat_imp['Feature'].tolist()
    retain_rate = 100 * ml_df['label'].mean()
    insight(
        f"The best-performing model is <strong>{best_model_name}</strong> with an AUC of <strong>{best_auc:.3f}</strong>. "
        f"It relies most on <strong>{top_feats[0]}</strong>, followed by {top_feats[1]} and {top_feats[2]}. "
        "The signals available at the moment of a first purchase are weak, so predictive power is limited — "
        "review score, often assumed to drive repeat buying, actually carries relatively little weight here. "
        "<br><br>"
        f"Note on limitations: this model uses 6 features available at the time of first order, and only "
        f"~{retain_rate:.0f}% of customers ever return — an inherently hard, low-signal problem. "
        "In a production setting, additional signals — product category, seller rating, customer's city, "
        "whether a review was left at all — could improve performance. "
        "The model should be retrained monthly as new order data accumulates."
    )


# ═══════════════════════════════════════════════════════════════
# TAB 7 — BUSINESS RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════
with tab_rec:
    st.space()
    st.subheader("Business Recommendations")
    st.caption("Actionable strategies derived from the data analysis and predictive model.")

    rec_data = [
        {
            "Priority": "🔴 High",
            "Area": "Logistics & Delivery",
            "Finding": f"Late deliveries hit {late_pct_kpi}% of orders and cut review scores by about "
                       f"{on_score - lt_score:.1f} points ({on_score:.2f} → {lt_score:.2f}) — the single biggest, "
                       f"statistically confirmed satisfaction lever in the data.",
            "Action": "Partner with faster regional carriers for top-revenue states, and hold internal delivery "
                      "targets a couple of days ahead of the customer-facing estimate to build a buffer.",
            "KPI": "Late delivery rate < 5% | On-time review score > 4.2",
        },
        {
            "Priority": "🔴 High",
            "Area": "Customer Retention",
            "Finding": f"About {100 * one_time / total:.0f}% of customers never place a second delivered order; "
                       f"those who do return take a median of {median_days} days.",
            "Action": "Run a post-purchase re-engagement series (day 14 / 30 / 60) with a personalized offer, "
                      "concentrated in the first 90 days while the customer is still warm.",
            "KPI": "Second-purchase conversion rate | 90-day repeat rate",
        },
        {
            "Priority": "🟡 Medium",
            "Area": "Churn Prediction & Targeting",
            "Finding": f"The best model reaches only AUC ≈ {best_auc:.2f} from first-order features — a weak signal, "
                       f"because {100 * one_time / total:.0f}% of customers never return and the top driver is simply "
                       f"order value, not satisfaction.",
            "Action": "Use the churn score to *prioritize* a limited re-engagement budget (target the highest-risk "
                      "decile), not as a precise filter. A/B test whether the coupon beats an untargeted send.",
            "KPI": "Incremental repeat rate vs. control | Coupon redemption rate",
        },
        {
            "Priority": "🟡 Medium",
            "Area": "Payment Optimisation",
            "Finding": f"Credit card ({cc_pct:.0f}% of orders) carries the highest average ticket, and larger baskets "
                       f"correlate with more installments — customers split payment when the total is big.",
            "Action": "Promote installment options on high-value categories to nudge 1-installment buyers toward "
                      "2–3 installments and lift basket size.",
            "KPI": "Avg order value | Installment adoption rate",
        },
        {
            "Priority": "🟢 Low",
            "Area": "New-Customer Onboarding (Recency Focus)",
            "Finding": f"RFM frequency carries almost no signal here — ~{one_order_pct:.0f}% of customers have exactly "
                       f"one order, so the Champions / Loyal / At-Risk tiers are empty. There is no loyal base to "
                       f"protect yet; only recency separates customers.",
            "Action": "Redirect spend from loyalty perks (no population) to first-to-second-purchase conversion: "
                      "an onboarding series and category cross-sell aimed at New Customers before they go cold.",
            "KPI": "New → repeat conversion | Time-to-second-order",
        },
    ]

    rec_df = pd.DataFrame(rec_data)

    for _, row in rec_df.iterrows():
        with st.expander(f"{row['Priority']}  ·  **{row['Area']}**", expanded=True):
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown("**📊 Finding**")
                st.write(row['Finding'])
            with c2:
                st.markdown("**🎯 Recommended Action**")
                st.write(row['Action'])
            st.markdown(f"**📏 Success Metrics:** `{row['KPI']}`")

    st.space()
    st.markdown("---")
    st.markdown(
        "##### Methodology Note\n"
        "All findings are derived from the Olist public dataset (2016–2018); customer-level analysis "
        "(retention, RFM, churn) is measured on delivered orders. "
        "Statistical claims are supported by Mann-Whitney U tests. "
        "The predictive model uses Gradient Boosting with 6 first-order features; "
        "production deployment would benefit from additional features (product category, seller region, seasonality). "
        "RFM segmentation uses quintile-based scoring."
    )