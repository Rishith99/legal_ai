"""Automated smoke test: simulates a user filing a case and an advocate updating it."""
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("app.py", default_timeout=60).run()
assert not at.exception, at.exception
print("App loads OK. Tabs:", len(at.tabs))
at.text_input(key="name").set_value("TestUser")
at.text_area(key="case_text").set_value("Someone hacked my Instagram account and is blackmailing me with morphed photos.")
at.button(key="analyse").click().run()
assert not at.exception, at.exception
print("Prediction message:", at.success[0].value)
print("Info message:", at.info[0].value)
print("TEST PASSED")

# voice path without microphone: no recording -> friendly error, no crash
at.button(key="stt").click().run()
assert not at.exception, at.exception
print("Voice button without recording ->", at.error[0].value)
import core, wave, io
buf = io.BytesIO()
with wave.open(buf, "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(b"\x00\x00" * 16000)
print("Transcribe silent WAV ->", core.transcribe_audio(buf.getvalue(), "English"))
print("VOICE CODE PATH OK")
