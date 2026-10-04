import streamlit as stl
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_preprocess_input
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenet_v3_preprocess_input


MODEL_OPTIONS = {
	"MobileNetV2": ("black_pepper.keras", mobilenet_v2_preprocess_input),
	"MobileNetV3Small": ("baiMobileNetV3.keras", mobilenet_v3_preprocess_input),
}
CLASS_NAMES = ["Footrot", "Pollu_Disease", "Slow-Decline", "leaf blight"]
CLASS_INFO = {
	"Footrot": "โรคเน่าคอดิน/เน่าโคน",
	"Pollu_Disease": "โรคพอลลู",
	"Slow-Decline": "โรคเหี่ยวแห้งช้า",
	"leaf blight": "โรคใบไหม้",
}
IMAGE_SIZE = (128, 128)

stl.set_page_config(page_title="ตรวจโรคใบพริกไทย", page_icon="🌿", layout="wide")
stl.markdown(
	"""
	<style>
	@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Thai:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
	:root { --ink: #171914; --muted: #45483f; --olive: #52652b; --olive-dark: #3d4d20; --olive-soft: #eef1e7; --line: #cbd1bf; --white: #ffffff; }
	html, body, [class*="css"] { font-family: 'IBM Plex Sans Thai', sans-serif; color: var(--ink); }
	.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background: #f6f7f2 !important; color: var(--ink) !important; }
	[data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2, [data-testid="stMarkdownContainer"] h3,
	[data-testid="stHeadingWithActionElements"] h1, [data-testid="stHeadingWithActionElements"] h2, [data-testid="stHeadingWithActionElements"] h3,
	[data-testid="stMarkdownContainer"] p, [data-testid="stCaptionContainer"] p, label { color: var(--ink) !important; }
	.block-container { max-width: 1120px; padding-top: 3rem; padding-bottom: 3rem; }
	.eyebrow { color: var(--olive-dark) !important; font: 700 12px 'Manrope', sans-serif; letter-spacing: 1.4px; text-transform: uppercase; }
	.hero-title { color: var(--ink) !important; font: 800 46px 'IBM Plex Sans Thai', sans-serif; line-height: 1.16; margin: 10px 0 12px; }
	.hero-copy { font-size: 16px; color: var(--muted) !important; max-width: 620px; line-height: 1.8; }
	.panel { background: rgba(255,255,255,.82); border: 1px solid #dce5d8; border-radius: 8px; padding: 22px; }
	.result-label, .small-note { color: var(--muted) !important; }
	.result-label { font-size: 14px; margin-bottom: 4px; }
	.result-name { color: var(--olive-dark); font: 800 28px 'Manrope', 'IBM Plex Sans Thai', sans-serif; overflow-wrap: anywhere; }
	.small-note { font-size: 13px; line-height: 1.7; }
	[data-testid="stVerticalBlockBorderWrapper"] { background: var(--white); border-color: var(--line); border-radius: 8px; }
	[data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-diagnosis-results) { background: #171914 !important; border-color: #171914 !important; }
	.st-key-diagnosis-results { background: #171914 !important; color: var(--white) !important; }
	.st-key-diagnosis-results [data-testid="stMarkdownContainer"] h1,
	.st-key-diagnosis-results [data-testid="stMarkdownContainer"] h2,
	.st-key-diagnosis-results [data-testid="stMarkdownContainer"] h3,
	.st-key-diagnosis-results [data-testid="stMarkdownContainer"] p,
	.st-key-diagnosis-results [data-testid="stHeadingWithActionElements"] h3,
	.st-key-diagnosis-results [data-testid="stMetricLabel"],
	.st-key-diagnosis-results [data-testid="stMetricValue"],
	.st-key-diagnosis-results [data-testid="stCaptionContainer"],
	.st-key-diagnosis-results .result-label,
	.st-key-diagnosis-results .result-name,
	.st-key-diagnosis-results .small-note { color: var(--white) !important; -webkit-text-fill-color: var(--white) !important; }
	[data-testid="stFileUploader"] section { background: var(--white); border: 1px dashed var(--olive); border-radius: 8px; }
	[data-testid="stFileUploader"] section, [data-testid="stFileUploader"] section p { color: var(--muted) !important; }
	[data-testid="stFileUploader"] button, [data-testid="stFileUploader"] button * { background: var(--olive-dark) !important; border-color: var(--olive-dark) !important; color: var(--white) !important; }
	[data-testid="stFileUploader"] section button p { color: var(--white) !important; -webkit-text-fill-color: var(--white) !important; }
	[data-testid="stFileUploader"] button svg { fill: var(--white) !important; }
	[data-testid="stHeader"] { background: #171914 !important; }
	[data-testid="stHeader"] button, [data-testid="stHeader"] button span, [data-testid="stHeader"] svg { color: var(--white) !important; fill: var(--white) !important; }
	[data-testid="stProgressBar"] > div > div { background-color: var(--olive); }
	[data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-diagnosis-results) [data-testid="stProgressBar"] > div > div { background-color: var(--olive) !important; }
	@media (max-width: 640px) { .block-container { padding: 1.5rem 1rem; } .hero-title { font-size: 34px; } .panel { padding: 16px; } }
	</style>
	""",
	unsafe_allow_html=True,
)

@stl.cache_resource
def load_model(model_path: str):
	path = Path(__file__).with_name(model_path)
	if not path.is_file():
		raise FileNotFoundError(f"ไม่พบไฟล์โมเดล: {path.name}")
	model = tf.keras.models.load_model(path)
	if model.output_shape[-1] != len(CLASS_NAMES):
		raise ValueError("จำนวนผลลัพธ์ของโมเดลไม่ตรงกับจำนวนคลาสที่กำหนด")
	return model


def prepare_image(image: Image.Image, preprocess) -> np.ndarray:
	image = ImageOps.exif_transpose(image).convert("RGB")
	image = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
	pixels = np.asarray(image, dtype=np.float32)
	return preprocess(np.expand_dims(pixels, axis=0))


stl.markdown('<div class="eyebrow">BLACK PEPPER · LEAF HEALTH</div>', unsafe_allow_html=True)
stl.markdown('<h1 class="hero-title">ตรวจโรคจากใบพริกไทย</h1>', unsafe_allow_html=True)
stl.markdown(
	'<p class="hero-copy">อัปโหลดภาพใบพริกไทยเพื่อประเมินเบื้องต้นด้วยโมเดล AI '
	'พร้อมดูคะแนนความมั่นใจของแต่ละคลาส</p>',
	unsafe_allow_html=True,
)
stl.write("")

left, right = stl.columns([1, 1], gap="large")
with left:
	with stl.container(border=True):
		stl.subheader("ภาพใบพริกไทย")
		selected_model = stl.selectbox("Model", list(MODEL_OPTIONS))
		uploaded_file = stl.file_uploader(
			"เลือกภาพใบพริกไทย", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
		)
		if uploaded_file:
			image = Image.open(uploaded_file)
			stl.image(image, use_container_width=True)
		else:
			stl.markdown(
				'<p class="small-note">รองรับไฟล์ JPG หรือ PNG · แนะนำภาพที่เห็นใบชัดและมีแสงเพียงพอ</p>',
				unsafe_allow_html=True,
			)

with right:
	with stl.container(border=True, key="diagnosis-results"):
		stl.subheader("ผลการประเมิน")
		if uploaded_file:
			try:
				model_file, preprocess = MODEL_OPTIONS[selected_model]
				model = load_model(model_file)
				probabilities = model.predict(prepare_image(image, preprocess), verbose=0)[0]
				best_index = int(np.argmax(probabilities))
				best_class = CLASS_NAMES[best_index]
				stl.markdown('<div class="result-label">คลาสที่โมเดลประเมินได้สูงสุด</div>', unsafe_allow_html=True)
				stl.markdown(
					f'<div class="result-name">{best_class}</div>'
					f'<div class="small-note">{CLASS_INFO[best_class]}</div>',
					unsafe_allow_html=True,
				)
				stl.metric("คะแนนของคลาสนี้", f"{probabilities[best_index] * 100:.1f}%")
				stl.divider()
				stl.caption("คะแนนแยกตามคลาส")
				for class_name, probability in sorted(
					zip(CLASS_NAMES, probabilities), key=lambda item: item[1], reverse=True
				):
					stl.write(f"{class_name} · {CLASS_INFO[class_name]}  ·  {probability * 100:.1f}%")
					stl.progress(float(np.clip(probability, 0, 1)))
			except (OSError, ValueError, RuntimeError) as error:
				stl.error(f"ไม่สามารถประเมินภาพได้: {error}")
		else:
			stl.markdown(
				'<p class="small-note">ผลการประเมินจะแสดงที่นี่หลังเลือกภาพ</p>',
				unsafe_allow_html=True,
			)

stl.write("")
stl.caption(
	"คลาสที่รองรับ: Footrot · Pollu_Disease · Slow-Decline · leaf blight | "
	"ผลจาก AI ใช้คัดกรองเบื้องต้น ไม่แทนการวินิจฉัยจากผู้เชี่ยวชาญ"
)