import streamlit as stl
from pathlib import Path

import numpy as np
import streamlit as stl
import tensorflow as tf
from PIL import Image, ImageOps
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


MODEL_PATH = Path(__file__).with_name("black_pepper.keras")
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
	:root { --ink: #202b24; --muted: #65736a; --leaf: #2d6948; --lime: #d7e7a2; --paper: #f5f7f1; }
	html, body, [class*="css"] { font-family: 'IBM Plex Sans Thai', sans-serif; color: var(--ink); }
	.stApp { background: black, #e4edcf 0, transparent 32%), linear-gradient(135deg, #f5f7f1 0%, #edf2e8 100%); }
	.block-container { max-width: 1120px; padding-top: 3rem; padding-bottom: 3rem; }
	.eyebrow { color: var(--leaf); font: 700 12px 'Manrope', sans-serif; letter-spacing: 1.4px; text-transform: uppercase; }
	.hero-title { font: 800 46px 'IBM Plex Sans Thai', sans-serif; line-height: 1.16; margin: 10px 0 12px; }
	.hero-copy { font-size: 16px; color: var(--muted); max-width: 620px; line-height: 1.8; }
	.panel { background: rgba(255,255,255,.82); border: 1px solid #dce5d8; border-radius: 8px; padding: 22px; }
	.result-label { color: var(--muted); font-size: 14px; margin-bottom: 4px; }
	.result-name { color: var(--leaf); font: 800 28px 'Manrope', 'IBM Plex Sans Thai', sans-serif; overflow-wrap: anywhere; }
	.small-note { color: var(--muted); font-size: 13px; line-height: 1.7; }
	[data-testid="stFileUploader"] section { background: rgba(255,255,255,.62); border: 1px dashed #8ca78e; border-radius: 8px; }
	[data-testid="stProgressBar"] > div > div { background-color: var(--leaf); }
	@media (max-width: 640px) { .block-container { padding: 1.5rem 1rem; } .hero-title { font-size: 34px; } .panel { padding: 16px; } }
	</style>
	""",
	unsafe_allow_html=True,
)


@stl.cache_resource
def load_model():
	if not MODEL_PATH.is_file():
		raise FileNotFoundError(f"ไม่พบไฟล์โมเดล: {MODEL_PATH.name}")
	model = tf.keras.models.load_model(MODEL_PATH)
	if model.output_shape[-1] != len(CLASS_NAMES):
		raise ValueError("จำนวนผลลัพธ์ของโมเดลไม่ตรงกับจำนวนคลาสที่กำหนด")
	return model


def prepare_image(image: Image.Image) -> np.ndarray:
	image = ImageOps.exif_transpose(image).convert("RGB")
	image = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
	pixels = np.asarray(image, dtype=np.float32)
	return preprocess_input(np.expand_dims(pixels, axis=0))


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
	with stl.container(border=True):
		stl.subheader("ผลการประเมิน")
		if uploaded_file:
			try:
				model = load_model()
				probabilities = model.predict(prepare_image(image), verbose=0)[0]
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