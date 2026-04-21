import streamlit as st
import sqlite3, json, os
import pandas as pd
import plotly.express as px
from PIL import Image

st.set_page_config(
    page_title = "Hy-ViS Surveillance Dashboard",
    page_icon  = "🛡️",
    layout     = "wide"
)

st.title("🛡️ Hy-ViS — Hybrid Video Intelligence System")
st.caption("Multi-modal AI surveillance pipeline")

# Resolve base paths relative to this script
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DB_PATH      = os.path.join(BASE_DIR, "surveillance.db")
FRAMES_DIR   = os.path.join(BASE_DIR, "saved_frames")
HEATMAP_PATH = os.path.join(BASE_DIR, "heatmap_final.png")
ZONES_PATH   = os.path.join(BASE_DIR, "zones.json")

# ── Sidebar navigation ────────────────────────────────────────────────────────
page = st.sidebar.radio("Navigation", [
    "📋 Alert Feed",
    "🗺️ Heatmap Viewer",
    "🔍 CLIP Search",
    "📊 Zone Overview",
    "⚙️ Settings",
])

# ── Load alerts from SQLite ───────────────────────────────────────────────────
@st.cache_data(ttl=5)
def load_alerts():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame(columns=["ID","Type","Severity","Frame","Timestamp","Details"])
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id,alert_type,severity,frame_idx,timestamp,details"
        " FROM alerts ORDER BY frame_idx"
    ).fetchall()
    conn.close()
    return pd.DataFrame(rows, columns=["ID","Type","Severity","Frame","Timestamp","Details"])

# ── Shared result grid helper ─────────────────────────────────────────────────
def _show_results(results, label):
    if not results:
        st.warning("No matching frames found.")
        return
    st.success(f"Top {len(results)} results for: **{label}**")
    st.divider()
    cols_per_row = 3
    for i in range(0, len(results), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, res in enumerate(results[i:i+cols_per_row]):
            with cols[j]:
                img = Image.open(res["frame_path"])
                st.image(img, use_container_width=True)
                st.caption(
                    f"⏱️ {res['timestamp']}  |  "
                    f"Frame {res['frame_num']}  |  "
                    f"Score: {res['similarity']:.3f}"
                )

# ════════════════════════════════════════════════════════════════════
# PAGE 1 — ALERT FEED
# ════════════════════════════════════════════════════════════════════
if page == "📋 Alert Feed":
    st.header("Alert Feed")

    df = load_alerts()

    if st.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()

    if df.empty:
        st.info("No alerts yet. Run pipeline.py first.")
    else:
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Total Alerts",  len(df))
        c2.metric("🔴 CRITICAL",   len(df[df.Severity=="CRITICAL"]))
        c3.metric("🟠 HIGH",       len(df[df.Severity=="HIGH"]))
        c4.metric("🔵 MEDIUM",     len(df[df.Severity=="MEDIUM"]))
        c5.metric("🟢 LOW",        len(df[df.Severity=="LOW"]))

        st.divider()

        col1, col2 = st.columns(2)
        sev_filter  = col1.multiselect(
            "Filter by Severity",
            ["CRITICAL","HIGH","MEDIUM","LOW"],
            default=["CRITICAL","HIGH","MEDIUM","LOW"]
        )
        type_filter = col2.multiselect(
            "Filter by Type",
            df.Type.unique().tolist(),
            default=df.Type.unique().tolist()
        )

        filtered = df[df.Severity.isin(sev_filter) & df.Type.isin(type_filter)]

        def colour_row(row):
            colours = {
                "CRITICAL" : "background-color:#FEE2E2",
                "HIGH"     : "background-color:#FEF3C7",
                "MEDIUM"   : "background-color:#DBEAFE",
                "LOW"      : "background-color:#F0FDF4",
            }
            return [colours.get(row.Severity, "")] * len(row)

        st.dataframe(
            filtered.style.apply(colour_row, axis=1),
            use_container_width=True,
            height=400
        )

        st.divider()
        st.subheader("Alert Timeline")
        fig = px.scatter(
            filtered, x="Frame", y="Type",
            color="Severity",
            color_discrete_map={"CRITICAL":"red","HIGH":"orange","MEDIUM":"royalblue","LOW":"green"},
            hover_data=["Timestamp","Details"],
            height=300, title="Alerts by Frame"
        )
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.pie(filtered, names="Type", title="Alert Type Distribution", height=350)
        st.plotly_chart(fig2, use_container_width=True)

# ════════════════════════════════════════════════════════════════════
# PAGE 2 — HEATMAP VIEWER
# ════════════════════════════════════════════════════════════════════
elif page == "🗺️ Heatmap Viewer":
    st.header("Person Density Heatmap")
    st.caption("Red = high activity  |  Blue = low activity")

    if os.path.exists(HEATMAP_PATH):
        img = Image.open(HEATMAP_PATH)
        st.image(img, caption="Accumulated person density across video",
                 use_container_width=True)
        with open(HEATMAP_PATH,"rb") as f:
            st.download_button("⬇️ Download Heatmap", f, "heatmap_final.png", "image/png")
    else:
        st.warning("No heatmap found.")
        st.info("Run pipeline.py first — heatmap_final.png is generated at the end.")

# ════════════════════════════════════════════════════════════════════
# PAGE 3 — CLIP SEARCH
# ════════════════════════════════════════════════════════════════════
elif page == "🔍 CLIP Search":
    st.header("CLIP Zero-Shot Frame Search")
    st.caption("Search video frames by describing what you see, or by uploading a reference person photo")

    # ── Load CLIP once and cache ──────────────────────────────────
    @st.cache_resource
    def load_clip_model():
        from modules.clip_search import CLIPSearch
        return CLIPSearch("saved_frames")

    # ── Check frames exist ────────────────────────────────────────
    if not os.path.exists(FRAMES_DIR) or len(os.listdir(FRAMES_DIR)) == 0:
        st.error("No saved frames found.")
        st.info("Run pipeline.py first. It saves frames to saved_frames/ automatically.")
        st.stop()

    frame_count = len(os.listdir(FRAMES_DIR))

    # ── Shared top bar ────────────────────────────────────────────
    col1, col2, col3 = st.columns([1, 1, 2])
    top_k = col1.slider("Results", 1, 10, 5)
    fps   = col2.number_input("Video FPS", value=25, min_value=1)
    with col3:
        st.write("")   # spacing
        if st.button("🔄 Rebuild Index"):
            with st.spinner("Rebuilding CLIP index..."):
                clip = load_clip_model()
                clip.rebuild_index()
                st.cache_resource.clear()
            st.success("Index rebuilt.")

    st.info(f"📂 {frame_count} frames indexed and ready for search")
    st.divider()

    # ── Two-tab layout ────────────────────────────────────────────
    tab_text, tab_person = st.tabs(["🔍 Text Search", "👤 Person Search"])

    # ──────────────────────────────────────────────────────────────
    # TAB 1 — TEXT SEARCH
    # ──────────────────────────────────────────────────────────────
    with tab_text:
        st.subheader("Search by Description")
        st.caption("Describe what you want to find in plain English")

        query = st.text_input(
            "Query",
            placeholder="e.g.  person in a red shirt  |  two people near the gate",
            label_visibility="collapsed",
            key="text_query_input"
        )

        if st.button("🔍 Search", type="primary", key="btn_text"):
            if query:
                with st.spinner(f'Searching for "{query}"...'):
                    clip    = load_clip_model()
                    results = clip.search(query, top_k=top_k, fps=fps)
                _show_results(results, query)
            else:
                st.warning("Please enter a search query first.")

        with st.expander("💡 Example queries to try"):
            examples = [
                "person in a red shirt",
                "person carrying a bag",
                "two people standing close together",
                "person running",
                "crowd gathering near a door",
                "person sitting on the floor",
                "empty walkway or corridor",
                "person pointing at something",
                "person in dark clothing",
                "multiple people walking",
            ]
            cols = st.columns(2)
            for i, ex in enumerate(examples):
                if cols[i % 2].button(ex, key=f"example_{i}"):
                    st.session_state["text_query_input"] = ex
                    st.rerun()

    # ──────────────────────────────────────────────────────────────
    # TAB 2 — PERSON SEARCH (image upload)
    # ──────────────────────────────────────────────────────────────
    with tab_person:
        st.subheader("Search by Person Photo")
        st.caption(
            "Upload a photo of the target person. "
            "CLIP will find frames where a visually similar person appears."
        )

        st.info(
            "💡 **Tips for best results:**\n"
            "- Crop the image tightly around the person\n"
            "- Use a clear, well-lit photo\n"
            "- Distinctive clothing (colors, patterns) improves accuracy"
        )

        uploaded = st.file_uploader(
            "Upload person image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            key="person_upload"
        )

        if uploaded:
            query_img = Image.open(uploaded).convert("RGB")

            c_img, c_info = st.columns([1, 2])
            with c_img:
                st.image(query_img, caption="Reference person", use_container_width=True)
            with c_info:
                w, h = query_img.size
                st.markdown(f"**File:** {uploaded.name}")
                st.markdown(f"**Size:** {w} × {h} px")
                st.markdown("---")
                if st.button("👤 Find This Person", type="primary", key="btn_person"):
                    with st.spinner("Scanning frames for this person..."):
                        clip    = load_clip_model()
                        results = clip.search_by_image(query_img, top_k=top_k, fps=fps)
                    _show_results(results, f"Person from {uploaded.name}")

# ════════════════════════════════════════════════════════════════════
# PAGE 4 — ZONE OVERVIEW
# ════════════════════════════════════════════════════════════════════
elif page == "📊 Zone Overview":
    st.header("Zone Incident Overview")

    df = load_alerts()
    zone_df = df[df.Type == "LOITERING"]

    if zone_df.empty:
        st.info("No zone incidents recorded yet.")
    else:
        st.metric("Total Zone Incidents", len(zone_df))
        st.dataframe(zone_df, use_container_width=True)

        fig = px.bar(
            zone_df, x="Frame", color="Severity",
            color_discrete_map={"CRITICAL":"red","HIGH":"orange","MEDIUM":"blue"},
            title="Zone Incidents by Frame", height=350
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Current Zone Configuration")
    if os.path.exists(ZONES_PATH):
        with open(ZONES_PATH) as f:
            zones = json.load(f)
        for z in zones:
            with st.expander(f"Zone: {z['name']}"):
                st.json(z)
    else:
        st.warning("zones.json not found.")

# ════════════════════════════════════════════════════════════════════
# PAGE 5 — SETTINGS
# ════════════════════════════════════════════════════════════════════
elif page == "⚙️ Settings":
    st.header("System Settings")

    st.subheader("Database")
    if os.path.exists(DB_PATH):
        size = os.path.getsize(DB_PATH) / 1024
        st.success(f"surveillance.db exists — {size:.1f} KB")
        df = load_alerts()
        st.metric("Total alerts in database", len(df))
        if st.button("🗑️ Clear all alerts", type="secondary"):
            conn = sqlite3.connect(DB_PATH)
            conn.execute("DELETE FROM alerts")
            conn.commit(); conn.close()
            st.cache_data.clear()
            st.success("All alerts cleared.")
    else:
        st.error("surveillance.db not found. Run init_db.py first.")

    st.divider()
    st.subheader("Saved Frames")
    if os.path.exists(FRAMES_DIR):
        count = len(os.listdir(FRAMES_DIR))
        st.info(f"{count} frames in saved_frames/")
        if st.button("🗑️ Clear saved frames"):
            import shutil
            shutil.rmtree(FRAMES_DIR)
            os.makedirs(FRAMES_DIR)
            INDEX_PATH = os.path.join(BASE_DIR, "clip_index.pt")
            if os.path.exists(INDEX_PATH):
                os.remove(INDEX_PATH)
            st.success("Frames and CLIP index cleared.")
    else:
        st.warning("saved_frames/ not found.")

    st.divider()
    st.subheader("CLIP Index")
    INDEX_PATH = os.path.join(BASE_DIR, "clip_index.pt")
    if os.path.exists(INDEX_PATH):
        size = os.path.getsize(INDEX_PATH) / 1024
        st.success(f"clip_index.pt exists — {size:.1f} KB")
        if st.button("🔄 Rebuild CLIP Index"):
            from modules.clip_search import CLIPSearch
            clip = CLIPSearch("saved_frames")
            clip.rebuild_index()
            st.success("CLIP index rebuilt.")
    else:
        st.info("No CLIP index yet. Will be built on first search.")
