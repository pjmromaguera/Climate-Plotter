import streamlit as st

def make_card(label,value,sub=""):

    st.markdown(
        f"""
        <div style="
            background:#161b22;
            border:1px solid #30363d;
            padding:15px;
            border-radius:12px;
            min-height:90px;
        ">
        <div style="font-size:12px;color:#8b949e">{label}</div>
        <div style="font-size:22px;font-weight:bold">{value}</div>
        <div style="font-size:13px;color:#8b949e">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True
    )