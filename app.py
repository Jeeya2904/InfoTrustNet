"""Optional Streamlit web demo; train the model before launching."""
import streamlit as st
from infotrustnet import InfoTrustNet
st.set_page_config(page_title="InfoTrustNet",page_icon="🛡️")
st.title("🛡️ InfoTrustNet")
st.caption("Knowledge Graph + text classifier + reinforcement learning + explainable AI")
@st.cache_resource
def get_model(): return InfoTrustNet.load()
try: model=get_model()
except Exception: st.error("No trained model found. Run python main.py train first."); st.stop()
text=st.text_area("Claim / headline","Berlin is the capital of France.")
source=st.text_input("Source domain (optional)","viralbuzz24.info")
if st.button("Analyze") and text.strip():
    r=model.explain(text,source)
    (st.error if r["label"]=="MISINFORMATION" else st.success)(f"{r['label']} - P(misinformation) = {r['p_misinformation']:.2f}")
    st.subheader("Why?"); st.text(r["explanation"]); st.bar_chart(r["contributions"])
    st.write("Token influence:",dict(r["token_importance"])); st.write("KG paths:",r["kg_paths"])
