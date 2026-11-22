"""Lawzo-style dashboard. Run:  streamlit run app.py   (after python generate_data.py and python main.py)"""
import os
import pandas as pd
import streamlit as st
import core

st.set_page_config(page_title="Legal Case Classifier & Advocate Recommender", page_icon="⚖️", layout="wide")

os.chdir(os.path.dirname(os.path.abspath(__file__)))   # always work from the project folder

if not os.path.exists(core.MODEL_PATH) or not os.path.exists("data/advocates.csv"):
    # First run (or model files missing): build data and train the model automatically
    import subprocess, sys
    with st.spinner("First-time setup: creating data and training the model (1-3 minutes). Please wait..."):
        r1 = subprocess.run([sys.executable, "generate_data.py"], capture_output=True, text=True)
        r2 = subprocess.run([sys.executable, "main.py"], capture_output=True, text=True)
    if not os.path.exists(core.MODEL_PATH):
        st.error("Automatic setup failed. Run `python generate_data.py` and `python main.py` manually.")
        st.code((r1.stderr + "\n" + r2.stderr)[-2000:])
        st.stop()
    st.rerun()

core.init_db()
core.seed_demo_cases()
advocates = core.load_advocates()

st.title("⚖️ AI-Based Legal Case Classification & Advocate Recommendation")
st.caption("MTech Internship Project - Academy of Skill Development")
st.warning("Educational prototype using a synthetic dataset and fictional advocates. This is NOT legal advice.")

tab1, tab2, tab3, tab4 = st.tabs(["🧑 Client: New Case", "📂 Client: Track My Cases",
                                  "⚖️ Advocate Dashboard", "📊 Model Results"])

# ------------------------------------------------------------------ TAB 1
with tab1:
    st.subheader("Describe your legal problem in simple words")
    c1, c2, c3 = st.columns(3)
    name = c1.text_input("Your name", placeholder="e.g. Asha", key="name")
    city = c2.selectbox("Your city", sorted(advocates.city.unique()), key="city")
    lang = c3.selectbox("Preferred language", list(core.VOICE_LANGS), key="lang")

    st.markdown("##### 🎤 Option 1: Speak your case (voice input)")
    st.caption("Click the microphone, speak, stop, then press 'Convert voice to text'. Speak in the language selected above. Needs internet.")
    audio = st.audio_input("Record your case description", key="voice")
    if st.button("Convert voice to text", key="stt"):
        if audio is None:
            st.error("Please record your voice first.")
        else:
            with st.spinner("Converting speech to text..."):
                txt, err = core.transcribe_audio(audio.getvalue(), lang)
            if err:
                st.error(err)
            else:
                st.session_state["case_text"] = txt     # fills the text box below
                st.success("Voice converted. Please check the text below and edit if needed.")

    st.markdown("##### ⌨️ Option 2: Type (or edit the voice text here)")
    text = st.text_area("Case description", height=130, key="case_text",
                        placeholder="e.g. My landlord is not returning my security deposit of Rs 50,000 after I vacated the flat.")
    auto = st.checkbox("Automatically assign the best-matching advocate", value=True, key="auto")
    go = st.button("Analyse my case", type="primary", key="analyse")
    if go:
        if not name.strip() or len(text.split()) < 4:
            st.error("Please enter your name and a case description of at least 4 words.")
        else:
            preds = core.predict_category(text, 3)
            cat, conf = preds[0]
            st.success(f"Predicted legal category: **{cat}**  (confidence {conf*100:.1f}%)")
            st.write("Other possibilities: " + ", ".join(f"{c} ({p*100:.0f}%)" for c, p in preds[1:]))
            recs = core.recommend_advocates(cat, city, lang, 3)
            st.markdown("#### Top recommended advocates")
            show = recs[["advocate_id", "name", "primary_specialization", "experience_years", "rating", "city", "languages", "match_score"]]
            st.dataframe(show, width="stretch", hide_index=True)
            if auto and len(recs):
                best = recs.iloc[0]
                cid = core.add_case(name.strip(), city, lang, text, cat, conf, best.advocate_id, best["name"])
                st.info(f"Case #{cid} registered and assigned to **{best['name']}** ({best.advocate_id}). Track it in the next tab using your name.")
            else:
                cid = core.add_case(name.strip(), city, lang, text, cat, conf)
                st.info(f"Case #{cid} registered as Open. Matching advocates can pick it up from their dashboard.")

# ------------------------------------------------------------------ TAB 2
with tab2:
    st.subheader("Track your cases")
    who = st.text_input("Enter the name you used while filing", value="Demo Client", key="track_name")
    mine = core.get_cases(client_name=who.strip()) if who.strip() else pd.DataFrame()
    if mine.empty:
        st.info("No cases found for this name.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Total cases", len(mine)); m2.metric("Active", int((mine.status != "Closed").sum()))
        m3.metric("Closed", int((mine.status == "Closed").sum()))
        st.dataframe(mine[["case_id", "category", "advocate_name", "status", "created", "updated", "description"]],
                     width="stretch", hide_index=True)

# ------------------------------------------------------------------ TAB 3
with tab3:
    st.subheader("Advocate view")
    labels = {r.advocate_id: f"{r['name']} ({r.advocate_id}) - {r.primary_specialization}" for _, r in advocates.iterrows()}
    aid = st.selectbox("Select advocate (simulated login)", list(labels), format_func=lambda x: labels[x])
    adv = advocates[advocates.advocate_id == aid].iloc[0]
    st.caption(f"{adv.city} | {adv.experience_years} yrs experience | rating {adv.rating} | languages: {adv.languages.replace(';', ', ')}")

    st.markdown("#### My assigned cases")
    mine = core.get_cases(advocate_id=aid)
    if mine.empty:
        st.info("No cases assigned to this advocate yet.")
    else:
        st.dataframe(mine[["case_id", "client_name", "category", "status", "updated", "description"]],
                     width="stretch", hide_index=True)
        c1, c2, c3 = st.columns([1, 2, 1])
        cid = c1.selectbox("Case #", mine.case_id.tolist(), key="upd_case")
        new = c2.selectbox("New status", core.STATUSES, key="upd_status")
        if c3.button("Update status", key="upd_btn"):
            core.update_status(int(cid), new); st.success(f"Case #{cid} updated to '{new}'. Refresh the tab to see it."); st.rerun()

    st.markdown(f"#### Suitable open cases in my specialization ({adv.primary_specialization})")
    openc = core.get_cases(unassigned_category=adv.primary_specialization)
    if openc.empty:
        st.info("No open cases waiting in this category.")
    else:
        st.dataframe(openc[["case_id", "client_name", "city", "language", "description"]], width="stretch", hide_index=True)
        pick = st.selectbox("Take case #", openc.case_id.tolist(), key="take_case")
        if st.button("Accept this case", key="take_btn"):
            core.assign_case(int(pick), aid, adv["name"]); st.success("Case assigned to you."); st.rerun()

# ------------------------------------------------------------------ TAB 4
with tab4:
    st.subheader("Model performance and analysis")
    if os.path.exists("outputs/final_test_results.csv"):
        st.markdown("**Final test results**"); st.dataframe(pd.read_csv("outputs/final_test_results.csv"), hide_index=True)
        st.markdown("**Cross-validation comparison**"); st.dataframe(pd.read_csv("outputs/model_comparison_cv.csv"), hide_index=True)
        a, b = st.columns(2)
        a.image("outputs/04_confusion_matrix.png", caption="Confusion matrix"); b.image("outputs/05_per_class_f1.png", caption="Per-category F1")
        a.image("outputs/01_category_distribution.png", caption="Dataset distribution"); b.image("outputs/03_model_comparison.png", caption="Model comparison")
        st.image("outputs/06_top_words.png", caption="Most influential words per category")
    else:
        st.info("Run main.py to generate results.")
