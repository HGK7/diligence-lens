from uuid import uuid4

import streamlit as st

from approval import request_approval, resume_approval
from risk_graph import run_review

st.set_page_config(page_title="Diligence Lens Approval", layout="wide")
st.title("Diligence Lens approval gate")
st.caption("Human-in-the-loop release path for diligence reports")

question = st.text_area(
    "Diligence question",
    value=(
        "Do any agreements conflict on liability caps or "
        "change-of-control consent?"
    ),
)


if st.button("Generate report for review"):
    try:
        with st.spinner("Generating the evidence-backed report..."):
            st.session_state["approval_review"] = run_review(question)
        st.session_state["approval_thread"] = f"atlas-{uuid4()}"
        request_approval(
            st.session_state["approval_review"]["report"],
            st.session_state["approval_thread"],
        )
        st.session_state["approval_status"] = "pending"
    except Exception as error:
        st.error(str(error))
        
review = st.session_state.get("approval_review")
status = st.session_state.get("approval_status")
thread_id = st.session_state.get("approval_thread")

if review:
    st.markdown(review["report"])
    
    
if status == "pending" and thread_id:
    st.warning("The report is paused and waiting for reviewer sign-off.")
    if st.button("Approve report"):
        result = resume_approval(thread_id, True)
        st.session_state["approval_status"] = (
            "approved" if result["approved"] else "rejected"
        )
    if st.button("Reject report"):
        result = resume_approval(thread_id, False)
        st.session_state["approval_status"] = (
            "approved" if result["approved"] else "rejected"
        )
        
if st.session_state.get("approval_status") == "approved" and review:
    st.success("Approved for release.")
    st.download_button(
        "Download approved report",
        data=review["report"],
        file_name="diligence-lens-approved-report.md",
        mime="text/markdown",
    )
elif st.session_state.get("approval_status") == "rejected":
    st.error("Report rejected. Generate a revised review before release.")
    