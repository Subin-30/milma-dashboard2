import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Milma Dairy Analytics",
    page_icon="🥛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# THEME
# ─────────────────────────────────────────────
CAT_COLORS = {
    'Curd':     '#2ecc71',
    'Cheese':   '#3498db',
    'Sambaram': '#e67e22',
    'Paneer':   '#9b59b6',
    'Yogurt':   '#e74c3c'
}
QUADRANT_COLORS = {
    'Star':               '#27ae60',
    'Push Item':          '#2980b9',
    'Premium Puzzle':     '#f39c12',
    'Efficiency Target':  '#e74c3c'
}

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    daily    = pd.read_csv('master_dataset.csv', parse_dates=['Date'])
    monthly  = pd.read_csv('master_monthly.csv')
    me_main  = pd.read_csv('menu_engineering_results.csv')
    me_seas  = pd.read_csv('menu_engineering_seasonal.csv')
    me_year  = pd.read_csv('menu_engineering_yearly.csv')
    forecast = pd.read_csv('sarimax_2026_forecasts.csv')
    perf     = pd.read_csv('sarimax_model_performance.csv')

    monthly['Date'] = pd.to_datetime(
        monthly['Year'].astype(str) + '-' + monthly['Month'].astype(str).str.zfill(2) + '-01'
    )
    forecast['Date'] = pd.to_datetime(
        forecast['Year'].astype(str) + '-' + forecast['Month'].astype(str).str.zfill(2) + '-01'
    )
    return daily, monthly, me_main, me_seas, me_year, forecast, perf

daily, monthly, me_main, me_seas, me_year, forecast, perf = load_data()

CATEGORIES = sorted(daily['Category'].unique())
PRODUCTS   = sorted(daily['Product'].unique())
YEARS      = sorted(daily['Year'].unique())

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/e/e0/Milma_logo.svg/200px-Milma_logo.svg.png",
             width=120)
    st.markdown("## 🥛 Milma Analytics")
    st.markdown("**Malabar KCMMF | 2021–2026**")
    st.divider()

    page = st.radio("Navigate", [
        "🏠 Overview",
        "📊 EDA & Trends",
        "⭐ Menu Engineering",
        "🔮 2026 Forecasts",
        "📋 Model Performance"
    ])

    st.divider()
    st.markdown("**Global Filters**")
    sel_cats = st.multiselect("Categories", CATEGORIES, default=CATEGORIES)
    sel_years = st.multiselect("Years", YEARS, default=YEARS)

    st.divider()
    st.caption("M.Sc. Data Analytics | Semester IV\nVaidyan Subin Thomas | 24203030\nGuide: Diljith K Benny\nRajagiri College of Social Sciences")

# ─────────────────────────────────────────────
# FILTER DATA
# ─────────────────────────────────────────────
if not sel_cats:  sel_cats  = CATEGORIES
if not sel_years: sel_years = YEARS

daily_f   = daily[daily['Category'].isin(sel_cats) & daily['Year'].isin(sel_years)]
monthly_f = monthly[monthly['Category'].isin(sel_cats) & monthly['Year'].isin(sel_years)]

# ═════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("🥛 Milma Fermented Dairy Sales Analytics")
    st.markdown("**Temporal Sales Behavior Analysis | Malabar KCMMF (Milma) | 2021–2025**")
    st.divider()

    # KPI row
    total_rev  = daily_f['Revenue'].sum()
    total_qty  = daily_f['Quantity'].sum()
    n_products = daily_f['Product'].nunique()
    avg_rate   = daily_f['Avg_Rate'].mean()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Revenue", f"Rs.{total_rev/1e6:.1f}M")
    k2.metric("Total Quantity", f"{total_qty/1e6:.2f}M units")
    k3.metric("Products", str(n_products))
    k4.metric("Avg Rate / Unit", f"Rs.{avg_rate:.2f}")

    st.divider()

    col1, col2 = st.columns(2)

    # Revenue by category donut
    with col1:
        st.subheader("Revenue Share by Category")
        cat_rev = daily_f.groupby('Category')['Revenue'].sum().reset_index()
        fig = px.pie(cat_rev, values='Revenue', names='Category',
                     color='Category',
                     color_discrete_map=CAT_COLORS,
                     hole=0.45)
        fig.update_traces(textposition='outside', textinfo='percent+label')
        fig.update_layout(height=380, showlegend=False, margin=dict(t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)

    # Annual revenue trend
    with col2:
        st.subheader("Annual Revenue Trend")
        yr_cat = daily_f.groupby(['Year','Category'])['Revenue'].sum().reset_index()
        fig2 = px.bar(yr_cat, x='Year', y='Revenue', color='Category',
                      color_discrete_map=CAT_COLORS, barmode='stack')
        fig2.update_layout(height=380, yaxis_tickformat=',.0f',
                           yaxis_title='Revenue (Rs)', margin=dict(t=20,b=20))
        st.plotly_chart(fig2, use_container_width=True)

    # Top 10 products
    st.subheader("Top 10 Products by Revenue")
    top10 = (daily_f.groupby(['Product','Category'])['Revenue']
             .sum().reset_index()
             .sort_values('Revenue', ascending=False).head(10))
    fig3 = px.bar(top10, x='Revenue', y='Product', color='Category',
                  color_discrete_map=CAT_COLORS, orientation='h')
    fig3.update_layout(height=380, xaxis_tickformat=',.0f',
                       yaxis={'categoryorder':'total ascending'},
                       margin=dict(t=20,b=20))
    st.plotly_chart(fig3, use_container_width=True)

    # Dataset info
    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.info(f"**Daily records:** {len(daily_f):,}\n\n**Date range:** {daily_f['Date'].min().date()} → {daily_f['Date'].max().date()}")
    c2.info(f"**Categories:** {', '.join(CATEGORIES)}\n\n**Filtered years:** {', '.join(map(str,sel_years))}")
    c3.info("**Model:** SARIMAX(p,d,q)(P,D,Q)[12]\n\n**Forecast horizon:** Jan–Dec 2026")

# ═════════════════════════════════════════════
# PAGE 2 — EDA & TRENDS
# ═════════════════════════════════════════════
elif page == "📊 EDA & Trends":
    st.title("📊 Exploratory Data Analysis & Trends")
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(["Monthly Trends", "Seasonal Heatmap", "YoY Growth", "Product Deep Dive"])

    # ── Tab 1: Monthly trends ──
    with tab1:
        st.subheader("Monthly Revenue by Category")
        agg = monthly_f.groupby(['Date','Category'])['Revenue'].sum().reset_index()
        fig = px.line(agg, x='Date', y='Revenue', color='Category',
                      color_discrete_map=CAT_COLORS, markers=True)

        # Festival bands
        for yr in sel_years:
            fig.add_vrect(x0=f"{yr}-04-01", x1=f"{yr}-04-30",
                          fillcolor="gold", opacity=0.08, line_width=0, annotation_text="Vishu")
            fig.add_vrect(x0=f"{yr}-08-25", x1=f"{yr}-09-10",
                          fillcolor="orange", opacity=0.08, line_width=0, annotation_text="Onam")

        fig.update_layout(height=420, yaxis_tickformat=',.0f',
                          yaxis_title='Revenue (Rs)', xaxis_title='Month')
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Gold = Vishu window | Orange = Onam window")

        # Quantity trend
        st.subheader("Monthly Quantity Sold")
        agg_qty = monthly_f.groupby(['Date','Category'])['Quantity'].sum().reset_index()
        fig2 = px.area(agg_qty, x='Date', y='Quantity', color='Category',
                       color_discrete_map=CAT_COLORS)
        fig2.update_layout(height=380, yaxis_title='Quantity (units)', xaxis_title='Month')
        st.plotly_chart(fig2, use_container_width=True)

    # ── Tab 2: Heatmap ──
    with tab2:
        st.subheader("Revenue Heatmap — Year × Month")
        sel_cat_heat = st.selectbox("Select Category", sel_cats, key='heat')
        heat_df = (monthly_f[monthly_f['Category']==sel_cat_heat]
                   .groupby(['Year','Month'])['Revenue'].sum()
                   .reset_index())
        pivot = heat_df.pivot(index='Year', columns='Month', values='Revenue').fillna(0)
        pivot.columns = ['Jan','Feb','Mar','Apr','May','Jun',
                         'Jul','Aug','Sep','Oct','Nov','Dec'][:len(pivot.columns)]

        fig = px.imshow(pivot, color_continuous_scale='YlOrRd',
                        aspect='auto', text_auto='.2s')
        fig.update_layout(height=350, title=f"{sel_cat_heat} — Revenue Heatmap (Rs)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Brighter = higher revenue. Confirms summer peak (Apr–Jun) for Curd & Sambaram.")

    # ── Tab 3: YoY Growth ──
    with tab3:
        st.subheader("Year-over-Year Revenue Growth")
        yoy = (monthly_f.groupby(['Year','Category'])['Revenue']
               .sum().reset_index())
        yoy['Prev'] = yoy.groupby('Category')['Revenue'].shift(1)
        yoy['YoY_Pct'] = (yoy['Revenue'] - yoy['Prev']) / yoy['Prev'] * 100

        fig = px.bar(yoy.dropna(), x='Year', y='YoY_Pct', color='Category',
                     barmode='group', color_discrete_map=CAT_COLORS)
        fig.add_hline(y=0, line_dash='dash', line_color='black')
        fig.update_layout(height=400, yaxis_title='YoY Growth (%)')
        st.plotly_chart(fig, use_container_width=True)

        # Summary table
        pivot_yoy = yoy.pivot(index='Category', columns='Year', values='YoY_Pct').round(1)
        st.dataframe(pivot_yoy.style.format("{:+.1f}%").background_gradient(cmap='RdYlGn', axis=None),
                     use_container_width=True)

    # ── Tab 4: Product Deep Dive ──
    with tab4:
        st.subheader("Product Deep Dive")
        sel_product = st.selectbox("Select Product",
                                   sorted(daily_f['Product'].unique()), key='prod_dd')
        prod_m = (monthly[monthly['Product']==sel_product]
                  .sort_values('Date'))

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Revenue", f"Rs.{prod_m['Revenue'].sum():,.0f}")
        c2.metric("Total Quantity", f"{prod_m['Quantity'].sum():,.0f}")
        c3.metric("Avg Rate", f"Rs.{prod_m['Avg_Rate'].mean():.2f}")

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=['Monthly Revenue', 'Monthly Quantity'])
        fig.add_trace(go.Scatter(x=prod_m['Date'], y=prod_m['Revenue'],
                                 mode='lines+markers', name='Revenue',
                                 line=dict(color=CAT_COLORS.get(prod_m['Category'].iloc[0],'#333'))),
                      row=1, col=1)
        fig.add_trace(go.Bar(x=prod_m['Date'], y=prod_m['Quantity'],
                             name='Quantity',
                             marker_color=CAT_COLORS.get(prod_m['Category'].iloc[0],'#333'),
                             opacity=0.6),
                      row=2, col=1)
        fig.update_layout(height=480, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ═════════════════════════════════════════════
# PAGE 3 — MENU ENGINEERING
# ═════════════════════════════════════════════
elif page == "⭐ Menu Engineering":
    st.title("⭐ Menu Engineering Framework")
    st.markdown("BCG-style portfolio classification: **Stars | Push Items | Premium Puzzles | Efficiency Targets**")
    st.divider()

    tab1, tab2, tab3 = st.tabs(["Overall Portfolio", "Year-wise Migration", "Seasonal Comparison"])

    # ── Tab 1: Overall ──
    with tab1:
        me_f = me_main[me_main['Category'].isin(sel_cats)]

        # Quadrant counts
        q_counts = me_f['Quadrant'].value_counts().reset_index()
        q_counts.columns = ['Quadrant','Count']

        c1, c2, c3, c4 = st.columns(4)
        for col, qname, emoji in zip(
            [c1,c2,c3,c4],
            ['Star','Push Item','Premium Puzzle','Efficiency Target'],
            ['⭐','📣','💎','🎯']
        ):
            n = len(me_f[me_f['Quadrant']==qname])
            col.metric(f"{emoji} {qname}", str(n) + " products")

        st.subheader("Portfolio Matrix — Quantity vs Revenue")
        fig = px.scatter(me_f,
                         x='Total_Quantity', y='Total_Revenue',
                         color='Quadrant', size='Revenue_Share',
                         color_discrete_map=QUADRANT_COLORS,
                         hover_data=['Product','Category','Recommendation'],
                         text='Product', size_max=50)

        # Threshold lines
        thresh_qty = me_f['Qty_Threshold'].iloc[0]
        thresh_rev = me_f['Rev_Threshold'].iloc[0]
        fig.add_vline(x=thresh_qty, line_dash='dash', line_color='gray', opacity=0.6)
        fig.add_hline(y=thresh_rev, line_dash='dash', line_color='gray', opacity=0.6)

        # Quadrant labels
        for label, xa, ya in [
            ('⭐ STARS',          0.75, 0.85),
            ('📣 PUSH ITEMS',     0.75, 0.15),
            ('💎 PREMIUM PUZZLES',0.25, 0.85),
            ('🎯 EFFICIENCY',     0.25, 0.15),
        ]:
            fig.add_annotation(xref='paper', yref='paper', x=xa, y=ya,
                               text=label, showarrow=False,
                               font=dict(size=11, color='gray'))

        fig.update_traces(textposition='top center', textfont_size=8)
        fig.update_layout(height=550, xaxis_title='Total Quantity Sold',
                          yaxis_title='Total Revenue (Rs)')
        st.plotly_chart(fig, use_container_width=True)

        # Recommendations table
        st.subheader("Product Recommendations")
        show_cols = ['Category','Product','Quadrant','Recommendation',
                     'Total_Revenue','Revenue_Share']
        st.dataframe(
            me_f[show_cols].sort_values('Total_Revenue', ascending=False)
            .style.applymap(
                lambda v: f"background-color: {QUADRANT_COLORS.get(v,'')};color:white"
                if v in QUADRANT_COLORS else '',
                subset=['Quadrant']
            ),
            use_container_width=True, height=400
        )

    # ── Tab 2: Year-wise migration ──
    with tab2:
        st.subheader("Quadrant Migration 2021 → 2025")
        me_yr_f = me_year[me_year['Category'].isin(sel_cats)]

        # Heatmap: product × year → quadrant as number
        quad_map = {'Star':4,'Push Item':3,'Premium Puzzle':2,'Efficiency Target':1}
        me_yr_f = me_yr_f.copy()
        me_yr_f['Quad_Num'] = me_yr_f['Quadrant'].map(quad_map)

        pivot_q = me_yr_f.pivot_table(index='Product', columns='Year',
                                       values='Quad_Num', aggfunc='first')
        pivot_label = me_yr_f.pivot_table(index='Product', columns='Year',
                                           values='Quadrant', aggfunc='first')

        fig = px.imshow(pivot_q,
                        color_continuous_scale=['#e74c3c','#f39c12','#2980b9','#27ae60'],
                        aspect='auto', text_auto=False,
                        zmin=1, zmax=4)

        # Add text labels
        for i, prod in enumerate(pivot_q.index):
            for j, yr in enumerate(pivot_q.columns):
                val = pivot_label.loc[prod, yr] if yr in pivot_label.columns else ''
                if pd.notna(val):
                    short = val[:2] if val else ''
                    fig.add_annotation(x=j, y=i, text=short,
                                       showarrow=False, font=dict(size=8, color='white'))

        fig.update_layout(height=max(400, len(pivot_q)*22),
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🟢 Star | 🔵 Push Item | 🟡 Premium Puzzle | 🔴 Efficiency Target")

        # Migration table
        st.dataframe(pivot_label, use_container_width=True)

    # ── Tab 3: Seasonal ──
    with tab3:
        st.subheader("Summer vs Non-Summer Portfolio Comparison")
        me_s_f = me_seas[me_seas['Category'].isin(sel_cats)]

        c1, c2 = st.columns(2)
        for col, season in zip([c1, c2], ['Summer', 'Non-Summer']):
            with col:
                st.markdown(f"**{'☀️' if season=='Summer' else '🌧️'} {season}**")
                s_df = me_s_f[me_s_f['Season']==season]
                fig = px.scatter(s_df,
                                 x='Total_Quantity', y='Total_Revenue',
                                 color='Quadrant',
                                 color_discrete_map=QUADRANT_COLORS,
                                 hover_data=['Product'],
                                 size='Revenue_Share', size_max=40)
                thresh_qty = s_df['Qty_Threshold'].iloc[0] if len(s_df) > 0 else 0
                thresh_rev = s_df['Rev_Threshold'].iloc[0] if len(s_df) > 0 else 0
                fig.add_vline(x=thresh_qty, line_dash='dash', line_color='gray')
                fig.add_hline(y=thresh_rev, line_dash='dash', line_color='gray')
                fig.update_layout(height=400, showlegend=(season=='Non-Summer'))
                st.plotly_chart(fig, use_container_width=True)

# ═════════════════════════════════════════════
# PAGE 4 — 2026 FORECASTS
# ═════════════════════════════════════════════
elif page == "🔮 2026 Forecasts":
    st.title("🔮 2026 SARIMAX Demand Forecasts")
    st.markdown("Festival-aware seasonal forecasts for Jan–Dec 2026 with 80% & 95% confidence intervals.")
    st.divider()

    fc_f = forecast[forecast['Category'].isin(sel_cats)]

    # KPI forecast totals
    k1, k2, k3, k4, k5 = st.columns(5)
    for col, cat in zip([k1,k2,k3,k4,k5], ['Curd','Cheese','Paneer','Sambaram','Yogurt']):
        if cat in sel_cats:
            total = fc_f[fc_f['Category']==cat]['Forecast_Revenue'].sum()
            col.metric(cat, f"Rs.{total/1e6:.2f}M")

    st.divider()
    tab1, tab2, tab3 = st.tabs(["Category Forecasts", "Product Forecast", "2025 vs 2026"])

    MONTH_LABELS = ['Jan','Feb','Mar','Apr','May','Jun',
                    'Jul','Aug','Sep','Oct','Nov','Dec']
    FESTIVAL_MONTHS = {3:'Eid', 4:'Vishu', 8:'Onam', 9:'Onam', 12:'Christmas'}

    # ── Tab 1: Category level ──
    with tab1:
        st.subheader("2026 Monthly Forecast by Category")

        cat_fc = fc_f.groupby(['Category','Month']).agg(
            Forecast_Revenue=('Forecast_Revenue','sum'),
            Lower_80=('Lower_80','sum'),
            Upper_80=('Upper_80','sum'),
            Lower_95=('Lower_95','sum'),
            Upper_95=('Upper_95','sum'),
        ).reset_index()

        fig = go.Figure()
        for cat in sorted(cat_fc['Category'].unique()):
            d     = cat_fc[cat_fc['Category']==cat].sort_values('Month')
            color = CAT_COLORS.get(cat,'#333')
            r,g,b = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)

            fig.add_trace(go.Scatter(
                x=d['Month'], y=d['Forecast_Revenue'],
                mode='lines+markers', name=cat,
                line=dict(color=color, width=2.5),
                marker=dict(size=8)
            ))
            fig.add_trace(go.Scatter(
                x=pd.concat([d['Month'], d['Month'][::-1]]),
                y=pd.concat([d['Upper_80'], d['Lower_80'][::-1]]),
                fill='toself',
                fillcolor=f'rgba({r},{g},{b},0.12)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False, name=f'{cat} 80% CI'
            ))

        # Festival markers
        for m, name in FESTIVAL_MONTHS.items():
            fig.add_vline(x=m, line_dash='dot', line_color='gray', opacity=0.5)
            fig.add_annotation(x=m, y=1, yref='paper', text=name,
                               showarrow=False, font=dict(size=9, color='gray'),
                               textangle=-45, yanchor='bottom')

        fig.update_layout(
            height=460,
            xaxis=dict(tickmode='array', tickvals=list(range(1,13)),
                       ticktext=MONTH_LABELS),
            yaxis_title='Forecast Revenue (Rs)',
            yaxis_tickformat=',.0f',
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Shaded bands = 80% confidence interval | Dotted lines = Kerala festival windows")

    # ── Tab 2: Product level ──
    with tab2:
        st.subheader("Individual Product 2026 Forecast")
        fc_products = sorted(fc_f['Product'].unique())
        sel_prod_fc = st.selectbox("Select Product", fc_products, key='fc_prod')

        p_fc   = fc_f[fc_f['Product']==sel_prod_fc].sort_values('Month')
        p_hist = monthly[monthly['Product']==sel_prod_fc].sort_values('Date')
        cat    = p_fc['Category'].iloc[0]
        color  = CAT_COLORS.get(cat,'#333')

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=p_hist['Date'], y=p_hist['Revenue'],
            mode='lines', name='Historical (2021–2025)',
            line=dict(color='#2c3e50', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=p_fc['Date'], y=p_fc['Forecast_Revenue'],
            mode='lines+markers', name='2026 Forecast',
            line=dict(color=color, width=2.5, dash='dot'),
            marker=dict(size=8)
        ))
        r,g,b = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)
        fig.add_trace(go.Scatter(
            x=pd.concat([p_fc['Date'], p_fc['Date'][::-1]]),
            y=pd.concat([p_fc['Upper_95'], p_fc['Lower_95'][::-1]]),
            fill='toself', fillcolor=f'rgba({r},{g},{b},0.08)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% CI'
        ))
        fig.add_trace(go.Scatter(
            x=pd.concat([p_fc['Date'], p_fc['Date'][::-1]]),
            y=pd.concat([p_fc['Upper_80'], p_fc['Lower_80'][::-1]]),
            fill='toself', fillcolor=f'rgba({r},{g},{b},0.18)',
            line=dict(color='rgba(255,255,255,0)'),
            name='80% CI'
        ))

        fig.update_layout(height=420, yaxis_title='Revenue (Rs)',
                          yaxis_tickformat=',.0f',
                          title=f"{sel_prod_fc} — Historical + 2026 Forecast")
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("2026 Total Forecast", f"Rs.{p_fc['Forecast_Revenue'].sum():,.0f}")
        c2.metric("Peak Month", MONTH_LABELS[p_fc.loc[p_fc['Forecast_Revenue'].idxmax(),'Month']-1])
        hist_2025 = p_hist[p_hist['Year']==2025]['Revenue'].sum() if 'Year' in p_hist.columns else 0
        fc_2026   = p_fc['Forecast_Revenue'].sum()
        yoy       = (fc_2026 - hist_2025) / hist_2025 * 100 if hist_2025 > 0 else 0
        c3.metric("YoY vs 2025", f"{yoy:+.1f}%")

    # ── Tab 3: 2025 vs 2026 ──
    with tab3:
        st.subheader("Annual Revenue: 2025 Actual vs 2026 Forecast")
        actual_2025 = (monthly[(monthly["Year"]==2025) & (monthly["Category"].isin(sel_cats))]
                       .groupby('Category')['Revenue'].sum().reset_index()
                       .rename(columns={'Revenue':'Rev_2025'}))
        fc_2026 = fc_f.groupby('Category')['Forecast_Revenue'].sum().reset_index()
        bar_df  = actual_2025.merge(fc_2026, on='Category')
        bar_df['YoY_Pct'] = (bar_df['Forecast_Revenue'] - bar_df['Rev_2025']) / bar_df['Rev_2025'] * 100

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='2025 Actual',
            x=bar_df['Category'], y=bar_df['Rev_2025'],
            marker_color=[CAT_COLORS.get(c) for c in bar_df['Category']],
            opacity=0.55
        ))
        fig.add_trace(go.Bar(
            name='2026 Forecast',
            x=bar_df['Category'], y=bar_df['Forecast_Revenue'],
            marker_color=[CAT_COLORS.get(c) for c in bar_df['Category']],
            opacity=1.0,
            text=[f"{p:+.1f}%" for p in bar_df['YoY_Pct']],
            textposition='outside'
        ))
        fig.update_layout(height=430, barmode='group',
                          yaxis_tickformat=',.0f',
                          yaxis_title='Annual Revenue (Rs)')
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            bar_df.assign(**{'YoY %': bar_df['YoY_Pct'].round(1)})
            [['Category','Rev_2025','Forecast_Revenue','YoY %']]
            .rename(columns={'Rev_2025':'2025 Actual','Forecast_Revenue':'2026 Forecast'})
            .style.format({'2025 Actual':'Rs.{:,.0f}','2026 Forecast':'Rs.{:,.0f}','YoY %':'{:+.1f}%'})
            .background_gradient(subset=['YoY %'], cmap='RdYlGn'),
            use_container_width=True
        )

# ═════════════════════════════════════════════
# PAGE 5 — MODEL PERFORMANCE
# ═════════════════════════════════════════════
elif page == "📋 Model Performance":
    st.title("📋 SARIMAX Model Performance")
    st.divider()

    perf_f = perf[perf['Category'].isin(sel_cats)]

    # Summary KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Products Modeled", str(len(perf_f)))
    k2.metric("Mean MAE",  f"Rs.{perf_f['MAE'].mean():,.0f}")
    k3.metric("Mean RMSE", f"Rs.{perf_f['RMSE'].mean():,.0f}")
    k4.metric("Mean AIC",  f"{perf_f['AIC'].mean():,.1f}")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("RMSE by Product")
        fig = px.bar(perf_f.sort_values('RMSE'),
                     x='RMSE', y='Product', color='Category',
                     color_discrete_map=CAT_COLORS, orientation='h')
        fig.update_layout(height=550,
                          yaxis={'categoryorder':'total ascending'},
                          xaxis_tickformat=',.0f')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("MAE by Category")
        cat_err = perf_f.groupby('Category')[['MAE','RMSE']].mean().reset_index()
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name='MAE',  x=cat_err['Category'], y=cat_err['MAE'],
                              marker_color='#3498db', opacity=0.8))
        fig2.add_trace(go.Bar(name='RMSE', x=cat_err['Category'], y=cat_err['RMSE'],
                              marker_color='#e74c3c', opacity=0.8))
        fig2.update_layout(height=380, barmode='group', yaxis_tickformat=',.0f',
                           yaxis_title='Error (Rs)')
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("SARIMAX Orders Used")
        order_df = perf_f[['Product','Category','Order','Seasonal','AIC']].copy()
        st.dataframe(order_df.sort_values('AIC'), use_container_width=True, height=280)

    # Full table
    st.subheader("Full Performance Table")
    st.dataframe(
        perf_f[['Category','Product','Order','Seasonal','MAE','RMSE','AIC']]
        .sort_values('RMSE')
        .style.format({'MAE':'Rs.{:,.0f}','RMSE':'Rs.{:,.0f}','AIC':'{:.1f}'})
        .background_gradient(subset=['RMSE'], cmap='RdYlGn_r'),
        use_container_width=True, height=500
    )

    st.divider()
    st.info("""
**About the Model**
- **Algorithm:** SARIMAX — Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors
- **Seasonal period:** 12 months (annual Kerala seasonal cycle)
- **Exogenous variables:** Is_Festival, Is_Summer, Is_Peak_Q2, Is_Monsoon
- **Training:** Jan 2021 – Jun 2025 | **Validation:** Jul – Dec 2025
- **Orders:** Auto-selected per product using AIC minimisation (pmdarima.auto_arima)
- **30 individual models** — one per product for maximum accuracy
    """)
