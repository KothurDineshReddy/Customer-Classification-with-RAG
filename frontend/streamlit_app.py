"""Streamlit frontend for the Customer Classification Tool."""

import io
import os
import time

import httpx
import pandas as pd
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
POLL_INTERVAL = 3  # seconds between status polls

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Customer Classification Tool",
    page_icon="🏷️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

defaults = {
    "job_id": None,
    "upload_done": False,
    "run_clicked": False,
    "last_status": None,
    "output_bytes": None,
    "records_total": 0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def _api(method: str, path: str, **kwargs) -> httpx.Response:
    """Thin wrapper; raises on network error, returns response otherwise."""
    try:
        return httpx.request(method, f"{API_URL}{path}", timeout=30, **kwargs)
    except httpx.RequestError as exc:
        st.error(f"Could not reach the API at {API_URL}. Is the backend running?  \n`{exc}`")
        st.stop()


def _api_json(method: str, path: str, **kwargs) -> dict:
    resp = _api(method, path, **kwargs)
    if not resp.is_success:
        detail = resp.json().get("detail", resp.text)
        st.error(f"API error {resp.status_code}: {detail}")
        st.stop()
    return resp.json()


# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------

st.title("🏷️ Customer Classification Tool")
st.caption(
    "Upload a CSV of customer accounts, classify them against the SGWS taxonomy, "
    "and download the enriched output."
)

# ---------------------------------------------------------------------------
# Section 1 — Upload
# ---------------------------------------------------------------------------

st.header("1 · Upload Customer File")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
    help="Required columns: customer_id, customer_name, address, city, state, zip",
)

if uploaded_file is not None:
    # Preview
    try:
        df_preview = pd.read_csv(uploaded_file)
        uploaded_file.seek(0)  # reset for later upload
    except Exception as exc:
        st.error(f"Could not parse the file: {exc}")
        st.stop()

    n_rows = len(df_preview)
    st.caption(f"**{n_rows:,} rows** · {len(df_preview.columns)} columns")
    st.dataframe(df_preview.head(5), use_container_width=True)

    upload_btn = st.button("⬆️ Upload to API", type="primary", disabled=st.session_state.upload_done)

    if upload_btn:
        with st.spinner("Uploading…"):
            resp = _api(
                "POST",
                "/upload",
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")},
            )
        if resp.status_code == 200:
            body = resp.json()
            st.session_state.job_id = body["job_id"]
            st.session_state.upload_done = True
            st.session_state.run_clicked = False
            st.session_state.last_status = None
            st.session_state.output_bytes = None
            st.session_state.records_total = n_rows
            st.success(f"✅ Uploaded {n_rows:,} rows — Job ID: `{body['job_id']}`")
        else:
            detail = resp.json().get("detail", resp.text)
            st.error(f"Upload failed: {detail}")

# ---------------------------------------------------------------------------
# Section 2 — Configure (only after upload)
# ---------------------------------------------------------------------------

if st.session_state.upload_done and st.session_state.job_id:
    st.divider()
    st.header("2 · Configure & Run")

    col_left, col_right = st.columns([1, 2])
    with col_left:
        taxonomy_version = st.selectbox(
            "Taxonomy version",
            options=["v1"],
            index=0,
            help="The reference taxonomy used for classification.",
        )
    with col_right:
        confidence_threshold = st.slider(
            "Confidence threshold",
            min_value=0.50,
            max_value=0.95,
            value=0.75,
            step=0.05,
            help="Records below this score will be flagged for manual review.",
            format="%.2f",
        )

    st.caption(
        f"Records with confidence < **{confidence_threshold:.2f}** will be marked "
        "`needs_review = True` in the output."
    )

    already_running = st.session_state.last_status in ("running", "completed", "failed")
    run_btn = st.button(
        "▶️ Run Classification",
        type="primary",
        disabled=already_running,
    )

    if run_btn:
        with st.spinner("Starting pipeline…"):
            body = _api_json(
                "POST",
                "/run",
                json={
                    "job_id": st.session_state.job_id,
                    "taxonomy_version": taxonomy_version,
                    "confidence_threshold": confidence_threshold,
                },
            )
        st.session_state.run_clicked = True
        st.session_state.last_status = body.get("status", "running")
        st.rerun()

# ---------------------------------------------------------------------------
# Section 3 — Status / progress (only after run clicked)
# ---------------------------------------------------------------------------

if st.session_state.run_clicked and st.session_state.job_id:
    st.divider()
    st.header("3 · Classification Progress")

    status_container = st.empty()
    progress_container = st.empty()
    detail_container = st.empty()

    def _render_status(status_body: dict) -> None:
        status = status_body["status"]
        processed = status_body.get("records_processed", 0)
        total = status_body.get("records_total", 0) or st.session_state.records_total
        fraction = processed / total if total > 0 else 0.0

        icon = {"pending": "⏳", "running": "⚙️", "completed": "✅", "failed": "❌"}.get(
            status, "❓"
        )

        with status_container.container():
            st.markdown(f"**Status:** {icon} `{status.upper()}`")

        with progress_container.container():
            st.progress(
                min(fraction, 1.0),
                text=f"{processed:,} / {total:,} records processed",
            )

        if status == "failed":
            with detail_container.container():
                st.error(f"Pipeline failed: {status_body.get('error_message', 'unknown error')}")
        elif status == "completed":
            with detail_container.container():
                st.success("Classification complete — see Download section below.")

    # Poll loop
    terminal_statuses = {"completed", "failed"}
    current_status = st.session_state.last_status or "running"

    while current_status not in terminal_statuses:
        status_body = _api_json("GET", f"/status/{st.session_state.job_id}")
        current_status = status_body["status"]
        st.session_state.last_status = current_status
        _render_status(status_body)

        if current_status in terminal_statuses:
            break

        time.sleep(POLL_INTERVAL)
        st.rerun()

    # Final render after terminal state reached
    status_body = _api_json("GET", f"/status/{st.session_state.job_id}")
    _render_status(status_body)

# ---------------------------------------------------------------------------
# Section 4 — Download (only when completed)
# ---------------------------------------------------------------------------

if st.session_state.last_status == "completed" and st.session_state.job_id:
    st.divider()
    st.header("4 · Download Results")

    # Fetch output bytes once and cache in session state
    if st.session_state.output_bytes is None:
        with st.spinner("Fetching output file…"):
            resp = _api("GET", f"/download/{st.session_state.job_id}")
        if resp.is_success:
            st.session_state.output_bytes = resp.content
        else:
            st.error(f"Could not fetch output: {resp.json().get('detail', resp.text)}")

    if st.session_state.output_bytes:
        col_dl, col_meta = st.columns([1, 3])
        with col_dl:
            st.download_button(
                label="⬇️ Download Output CSV",
                data=st.session_state.output_bytes,
                file_name=f"classified_{st.session_state.job_id[:8]}.csv",
                mime="text/csv",
                type="primary",
            )

        # Preview
        try:
            df_out = pd.read_csv(io.BytesIO(st.session_state.output_bytes))
            with col_meta:
                st.caption(
                    f"**{len(df_out):,} rows classified** · "
                    f"{df_out['needs_review'].sum():,} flagged for review"
                )

            st.subheader("Output preview (first 10 rows)")

            # Highlight needs_review rows in amber
            def _highlight_review(row):
                color = "background-color: #fff3cd" if row.get("needs_review") else ""
                return [color] * len(row)

            display_cols = [
                c
                for c in [
                    "customer_id",
                    "customer_name",
                    "channel",
                    "establishment_type",
                    "premise",
                    "national_regional",
                    "unit_type",
                    "confidence",
                    "needs_review",
                ]
                if c in df_out.columns
            ]
            styled = (
                df_out[display_cols]
                .head(10)
                .style.apply(_highlight_review, axis=1)
                .format({"confidence": "{:.3f}"})
            )
            st.dataframe(styled, use_container_width=True)

        except Exception as exc:
            st.warning(f"Could not parse output for preview: {exc}")

    # Reset button to allow running another file
    st.divider()
    if st.button("🔄 Start over with a new file"):
        for k in defaults:
            st.session_state[k] = defaults[k]
        st.rerun()
