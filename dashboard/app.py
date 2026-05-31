"""
dashboard/app.py — CU Voice Coach Analytics Dashboard

Run:  streamlit run dashboard/app.py
      (or: python -m streamlit run dashboard/app.py)

Requires: streamlit, pandas, plotly, sqlalchemy, psycopg2-binary (or asyncpg)

This dashboard reads directly from the member_coaching_sessions table using a
synchronous SQLAlchemy connection (Streamlit is not async).  It is read-only
and never modifies session or member data.

Sections
--------
1. KPI strip       — Calls, Avg duration, Escalation rate, Completion rate
2. Sentiment gauge — Donut chart of sentiment distribution
3. Topics chart    — Horizontal bar of top 20 most-discussed topics
4. Questions table — Top 30 most frequently asked member questions
5. Escalations     — Breakdown by escalation type over time
6. Time-series     — Daily call volume for the selected period

The "Most common questions" table feeds directly into the RAG knowledge base
backlog: the P52 text coach team can review this list weekly and add/update
FAQ articles for any questions that appear ≥ threshold times.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine, text

# ─────────────────────────────────────────────────────────────────────────────
# Page config  (must be the first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title  = "Voice Coach Analytics",
    page_icon   = "📊",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# DB connection  (cached so it is created once per Streamlit session)
# ─────────────────────────────────────────────────────────────────────────────

DATABASE_URL = os.getenv("DATABASE_URL", "")

@st.cache_resource(show_spinner=False)
def _get_engine():
    if not DATABASE_URL:
        st.error(
            "**DATABASE_URL is not set.**\n\n"
            "Add it as an environment variable for this Streamlit service:\n"
            "```\nDATABASE_URL=postgresql://user:pass@host:5432/cu_coach\n```"
        )
        st.stop()
    # Replace asyncpg driver with psycopg2 for sync Streamlit use
    url = DATABASE_URL.replace("+asyncpg", "+psycopg2")
    return create_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": 5})


# ─────────────────────────────────────────────────────────────────────────────
# Data loading  (cached with TTL so the dashboard auto-refreshes)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=120, show_spinner="Loading call data…")
def load_sessions(start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """
    Load all session rows in [start_dt, end_dt] from Postgres.

    JSONB columns (topics_covered, member_questions, action_items) are
    returned as Python objects by psycopg2 and left as-is for downstream
    expansion.
    """
    query = text("""
        SELECT
            id,
            call_sid,
            member_id,
            started_at,
            ended_at,
            duration_s,
            topics_covered,
            member_questions,
            information_provided,
            action_items,
            member_sentiment,
            follow_up_required,
            escalated,
            escalation_type
        FROM member_coaching_sessions
        WHERE started_at >= :start
          AND started_at <  :end
        ORDER BY started_at DESC
    """)
    try:
        with _get_engine().connect() as conn:
            df = pd.read_sql(query, conn, params={"start": start_dt, "end": end_dt})
    except Exception as exc:
        st.error(f"Database error: {exc}")
        return pd.DataFrame()

    # Ensure JSON columns are lists (psycopg2 returns them as Python objects already)
    for col in ("topics_covered", "member_questions", "action_items"):
        if col in df.columns:
            df[col] = df[col].apply(lambda v: v if isinstance(v, list) else [])

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — date range filter
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🏦 Voice Coach")
    st.subheader("Analytics Dashboard")
    st.divider()

    preset = st.selectbox(
        "Date range",
        ["This week", "Last 7 days", "Last 30 days", "This month", "Custom"],
        index=0,
    )

    today    = datetime.now(tz=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = today - timedelta(days=today.weekday())

    if preset == "This week":
        start_dt, end_dt = week_start, today + timedelta(days=1)
    elif preset == "Last 7 days":
        start_dt, end_dt = today - timedelta(days=7), today + timedelta(days=1)
    elif preset == "Last 30 days":
        start_dt, end_dt = today - timedelta(days=30), today + timedelta(days=1)
    elif preset == "This month":
        start_dt = today.replace(day=1)
        end_dt   = today + timedelta(days=1)
    else:
        col1, col2 = st.columns(2)
        with col1:
            sd = st.date_input("From", value=today.date() - timedelta(days=7))
        with col2:
            ed = st.date_input("To",   value=today.date())
        start_dt = datetime.combine(sd, datetime.min.time()).replace(tzinfo=timezone.utc)
        end_dt   = datetime.combine(ed, datetime.max.time()).replace(tzinfo=timezone.utc)

    st.caption(f"Showing {start_dt.date()} → {end_dt.date()}")
    st.divider()

    min_question_freq = st.slider(
        "Min frequency for question list", min_value=1, max_value=20, value=2,
        help="Only show questions asked this many times or more",
    )

    st.divider()
    if st.button("🔄 Refresh data"):
        st.cache_data.clear()
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────

df = load_sessions(start_dt, end_dt)

# Previous period (same length) for delta comparisons
period_len   = end_dt - start_dt
prev_start   = start_dt - period_len
prev_end     = start_dt
df_prev      = load_sessions(prev_start, prev_end)

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.title("📊 Voice Financial Coach — Analytics")
st.caption(
    f"Data: {start_dt.strftime('%b %d')} – {end_dt.strftime('%b %d, %Y')}  "
    f"| Refreshes every 2 min  |  {len(df):,} calls loaded"
)

if df.empty:
    st.info("No calls found for the selected period.  Try a wider date range.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# KPI strip
# ─────────────────────────────────────────────────────────────────────────────

n_calls     = len(df)
n_prev      = len(df_prev)
avg_dur_s   = df["duration_s"].mean() if "duration_s" in df else 0
avg_dur_min = avg_dur_s / 60
esc_rate    = df["escalated"].mean() * 100 if "escalated" in df else 0
prev_esc    = df_prev["escalated"].mean() * 100 if (not df_prev.empty and "escalated" in df_prev) else 0
completed   = df[df["escalated"] == False]
completion  = len(completed) / n_calls * 100 if n_calls else 0
fu_rate     = df["follow_up_required"].mean() * 100 if "follow_up_required" in df else 0

def _delta(current, previous) -> str:
    if previous == 0:
        return ""
    d = current - previous
    arrow = "▲" if d > 0 else "▼"
    return f"{arrow} {abs(d):.0f}"

k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.metric(
        "📞 Total calls",
        f"{n_calls:,}",
        delta=_delta(n_calls, n_prev),
        delta_color="normal",
    )
with k2:
    mins = int(avg_dur_min)
    secs = int((avg_dur_min - mins) * 60)
    st.metric("⏱ Avg duration", f"{mins}m {secs:02d}s")
with k3:
    st.metric(
        "🚨 Escalation rate",
        f"{esc_rate:.1f}%",
        delta=f"{_delta(esc_rate, prev_esc)}%",
        delta_color="inverse",   # lower is better
    )
with k4:
    st.metric(
        "✅ Completion rate",
        f"{completion:.1f}%",
        help="Calls handled fully by AI (not escalated to human)",
    )
with k5:
    st.metric(
        "📋 Follow-up rate",
        f"{fu_rate:.1f}%",
        help="Calls that generated a CRM follow-up task",
    )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Row 1:  Sentiment distribution  |  Daily call volume
# ─────────────────────────────────────────────────────────────────────────────

col_sent, col_vol = st.columns([1, 2])

with col_sent:
    st.subheader("Member Sentiment")

    sentiment_order  = ["positive", "neutral", "concerned", "distressed"]
    sentiment_colors = {
        "positive":  "#22c55e",
        "neutral":   "#94a3b8",
        "concerned": "#f59e0b",
        "distressed":"#ef4444",
    }

    sent_counts = (
        df["member_sentiment"]
        .value_counts()
        .reindex(sentiment_order, fill_value=0)
        .reset_index()
    )
    sent_counts.columns = ["sentiment", "count"]

    fig_sent = px.pie(
        sent_counts,
        names  = "sentiment",
        values = "count",
        hole   = 0.55,
        color  = "sentiment",
        color_discrete_map = sentiment_colors,
        category_orders    = {"sentiment": sentiment_order},
    )
    fig_sent.update_traces(textposition="outside", textinfo="label+percent")
    fig_sent.update_layout(
        showlegend   = False,
        margin       = dict(t=10, b=10, l=10, r=10),
        height       = 280,
    )
    st.plotly_chart(fig_sent, use_container_width=True)

    # Satisfaction score: weight positive=100, neutral=60, concerned=30, distressed=0
    weights = {"positive": 100, "neutral": 60, "concerned": 30, "distressed": 0}
    if n_calls > 0:
        score = sum(
            weights.get(row["sentiment"], 60) * row["count"]
            for _, row in sent_counts.iterrows()
        ) / n_calls
        st.metric("Inferred satisfaction score", f"{score:.0f} / 100")

with col_vol:
    st.subheader("Daily Call Volume")

    if "started_at" in df.columns:
        vol_df = (
            df.assign(date=pd.to_datetime(df["started_at"]).dt.date)
            .groupby("date")
            .size()
            .reset_index(name="calls")
        )
        fig_vol = px.bar(
            vol_df,
            x      = "date",
            y      = "calls",
            labels = {"date": "", "calls": "Calls"},
            color_discrete_sequence = ["#3b82f6"],
        )
        fig_vol.update_layout(
            margin    = dict(t=10, b=10, l=0, r=0),
            height    = 280,
            xaxis     = dict(tickformat="%b %d"),
            yaxis     = dict(title="Calls", rangemode="tozero"),
            plot_bgcolor = "rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_vol, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Row 2:  Topics frequency  |  Escalation breakdown
# ─────────────────────────────────────────────────────────────────────────────

col_topics, col_esc = st.columns([3, 2])

with col_topics:
    st.subheader("🏷 Topics Discussed")
    st.caption("From AI-extracted session summaries — top 20")

    all_topics: list[str] = []
    for topics in df["topics_covered"]:
        if isinstance(topics, list):
            all_topics.extend(t.strip() for t in topics if t)

    if all_topics:
        topic_counts = (
            pd.Series(Counter(all_topics))
            .sort_values(ascending=True)
            .tail(20)
            .reset_index()
        )
        topic_counts.columns = ["topic", "count"]

        fig_topics = px.bar(
            topic_counts,
            x      = "count",
            y      = "topic",
            orientation = "h",
            labels = {"count": "Mentions", "topic": ""},
            color  = "count",
            color_continuous_scale = "Blues",
        )
        fig_topics.update_layout(
            margin            = dict(t=10, b=10, l=0, r=0),
            height            = 420,
            coloraxis_showscale = False,
            plot_bgcolor      = "rgba(0,0,0,0)",
            yaxis             = dict(tickfont=dict(size=12)),
        )
        st.plotly_chart(fig_topics, use_container_width=True)
    else:
        st.info("No topic data available for this period.")

with col_esc:
    st.subheader("🚨 Escalation Breakdown")

    esc_df = df[df["escalated"] == True]
    if not esc_df.empty:
        esc_type_counts = (
            esc_df["escalation_type"]
            .fillna("unspecified")
            .value_counts()
            .reset_index()
        )
        esc_type_counts.columns = ["type", "count"]

        esc_colors = {
            "crisis":             "#ef4444",
            "financial_hardship": "#f59e0b",
            "complaint":          "#8b5cf6",
            "investment_advice":  "#3b82f6",
            "escalation":         "#94a3b8",
            "unspecified":        "#e2e8f0",
        }

        fig_esc = px.pie(
            esc_type_counts,
            names  = "type",
            values = "count",
            hole   = 0.4,
            color  = "type",
            color_discrete_map = esc_colors,
        )
        fig_esc.update_traces(textposition="outside", textinfo="label+value")
        fig_esc.update_layout(
            showlegend = False,
            margin     = dict(t=10, b=10, l=10, r=10),
            height     = 300,
        )
        st.plotly_chart(fig_esc, use_container_width=True)

        # Trend: escalations over time
        esc_vol = (
            esc_df.assign(date=pd.to_datetime(esc_df["started_at"]).dt.date)
            .groupby(["date", "escalation_type"])
            .size()
            .reset_index(name="count")
        )
        fig_esc_vol = px.bar(
            esc_vol,
            x      = "date",
            y      = "count",
            color  = "escalation_type",
            labels = {"date": "", "count": "Escalations", "escalation_type": "Type"},
            color_discrete_map = esc_colors,
        )
        fig_esc_vol.update_layout(
            margin   = dict(t=10, b=10, l=0, r=0),
            height   = 180,
            xaxis    = dict(tickformat="%b %d"),
            plot_bgcolor = "rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_esc_vol, use_container_width=True)
    else:
        st.success("No escalations in the selected period. 🎉")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Row 3:  Most common member questions  (→ RAG knowledge base feed)
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("❓ Most Common Member Questions")
st.caption(
    f"Questions asked ≥ {min_question_freq}× — "
    "**Review weekly and update the P52 text coach FAQ / RAG knowledge base.**"
)

all_questions: list[str] = []
for qs in df["member_questions"]:
    if isinstance(qs, list):
        all_questions.extend(q.strip() for q in qs if q and len(q) > 8)

if all_questions:
    q_counts = Counter(all_questions)
    q_df     = (
        pd.DataFrame(q_counts.items(), columns=["Question", "Times Asked"])
        .query("`Times Asked` >= @min_question_freq")
        .sort_values("Times Asked", ascending=False)
        .reset_index(drop=True)
    )
    q_df.index = q_df.index + 1   # 1-based row numbers

    if not q_df.empty:
        col_q, col_export = st.columns([5, 1])
        with col_q:
            st.dataframe(
                q_df,
                use_container_width = True,
                height              = 400,
                column_config       = {
                    "Question":    st.column_config.TextColumn(width="large"),
                    "Times Asked": st.column_config.NumberColumn(format="%d"),
                },
            )
        with col_export:
            st.download_button(
                label     = "⬇ Export CSV",
                data      = q_df.to_csv(index=False).encode("utf-8"),
                file_name = f"voice_coach_questions_{start_dt.date()}_{end_dt.date()}.csv",
                mime      = "text/csv",
                help      = "Download question list for knowledge base review",
            )
    else:
        st.info(f"No questions asked ≥ {min_question_freq} times in this period.")
else:
    st.info("No question data available for this period.")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Row 4:  Follow-up action items overview
# ─────────────────────────────────────────────────────────────────────────────

with st.expander("📋 Pending Follow-Up Actions", expanded=False):
    fu_sessions = df[df["follow_up_required"] == True].copy()

    if fu_sessions.empty:
        st.info("No follow-up items in the selected period.")
    else:
        # Flatten action_items JSONB into rows
        rows = []
        for _, sess in fu_sessions.iterrows():
            for item in (sess["action_items"] or []):
                rows.append({
                    "Call date":    pd.to_datetime(sess["started_at"]).date(),
                    "Member ID":    sess["member_id"] or "guest",
                    "Task":         item.get("task", ""),
                    "Owner":        item.get("owner", ""),
                    "Sentiment":    sess["member_sentiment"],
                })
        if rows:
            ai_df = pd.DataFrame(rows).sort_values("Call date", ascending=False)
            st.dataframe(
                ai_df,
                use_container_width = True,
                column_config={
                    "Owner": st.column_config.SelectboxColumn(
                        options=["CU", "member", "specialist"]
                    ),
                },
            )
        else:
            st.info("Follow-up flagged but no structured action items extracted.")

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.caption(
    "Data sourced from `member_coaching_sessions` table.  "
    "Sentiment is AI-inferred and should not be used as sole basis for member outreach.  "
    "Questions table exports feed the P52 text coach RAG knowledge base review process."
)
