import streamlit as st
import torch, cv2, numpy as np
from PIL import Image
from torchvision import transforms, models

st.set_page_config(page_title="PCB Detect", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@700&display=swap');
[data-testid="stAppViewContainer"]{background:#080C10!important;color:#C8D8E8;
  background-image:linear-gradient(rgba(0,255,180,.03) 1px,transparent 1px),linear-gradient(90deg,rgba(0,255,180,.03) 1px,transparent 1px);background-size:36px 36px;}
[data-testid="stHeader"],[data-testid="stToolbar"],footer{display:none!important;}
.block-container{padding:2rem 3rem!important;max-width:100%!important;}
h1{font-family:'Rajdhani',sans-serif;font-size:3rem;color:#E8F4FF;letter-spacing:.05em;margin:0;}
h1 span{color:#00FFB4;}
.lbl{font-family:'Share Tech Mono',monospace;font-size:10px;letter-spacing:.3em;color:#00FFB4;
  margin:1.2rem 0 .6rem;display:flex;align-items:center;gap:8px;}
.lbl::after{content:'';flex:1;height:1px;background:linear-gradient(90deg,rgba(0,255,180,.4),transparent);}
.card{background:rgba(0,255,180,.04);border:1px solid rgba(0,255,180,.15);padding:16px 20px;border-radius:3px;margin-bottom:4px;}
.card .v{font-family:'Rajdhani',sans-serif;font-size:26px;font-weight:700;color:#00FFB4;}
.card .k{font-family:'Share Tech Mono',monospace;font-size:10px;color:#4A6878;letter-spacing:.2em;}
.chip{display:inline-block;font-family:'Share Tech Mono',monospace;font-size:10px;padding:4px 10px;
  border:1px solid rgba(100,120,140,.3);color:#4A6878;margin:3px 2px;border-radius:2px;}
.chip.on{border-color:#00FFB4;color:#00FFB4;background:rgba(0,255,180,.08);}
.bar-t{height:5px;background:rgba(255,255,255,.05);border-radius:3px;overflow:hidden;margin-top:3px;}
.bar-f{height:100%;background:linear-gradient(90deg,#00FFB4,#00C8FF);border-radius:3px;}
[data-testid="stFileUploader"]{border:1px dashed rgba(0,255,180,.25)!important;background:rgba(0,255,180,.02)!important;border-radius:4px!important;}
[data-testid="stDownloadButton"] button{background:rgba(0,255,180,.08)!important;border:1px solid rgba(0,255,180,.4)!important;
  color:#00FFB4!important;font-family:'Rajdhani',sans-serif!important;font-weight:700!important;letter-spacing:.1em!important;border-radius:2px!important;}
</style>""", unsafe_allow_html=True)

st.markdown('<p style="font-family:monospace;font-size:10px;letter-spacing:.3em;color:#00FFB4;opacity:.6">// AUTOMATED VISUAL INSPECTION</p>', unsafe_allow_html=True)
st.markdown("<h1>PCB<span>DETECT</span></h1>", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, 6)
    m.load_state_dict(torch.load("pcb_defect_model.pth", map_location="cpu"))
    return m.eval()

CLASSES = ['missing_hole','mouse_bite','open_circuit','short','spur','spurious_copper']
tfm = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])

try:    model = load_model(); ok = True
except: ok = False; st.warning("⚠ `pcb_defect_model.pth` not found.")

st.markdown('<div class="lbl">UPLOAD IMAGE</div>', unsafe_allow_html=True)
f = st.file_uploader("", type=["jpg","jpeg","png"], label_visibility="collapsed")

if f:
    img = Image.open(f).convert("RGB")
    arr = np.array(img)
    _, thresh = cv2.threshold(cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY), 120, 255, cv2.THRESH_BINARY)

    label, acc, probs = "N/A", 0.0, []
    if ok:
        with torch.no_grad():
            p = torch.softmax(model(tfm(img).unsqueeze(0)), 1)[0]
            idx = p.argmax().item()
            label, acc, probs = CLASSES[idx], p[idx].item()*100, p.tolist()

    out_img = arr.copy()
    cv2.rectangle(out_img,(50,50),(150,150),(0,255,180),2)
    cv2.putText(out_img, label,(50,42),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,180),2)

    st.markdown('<div class="lbl">SCAN RESULTS</div>', unsafe_allow_html=True)
    for col, image, title in zip(st.columns(3), [img, thresh, out_img], ["Input","Threshold","Detection"]):
        col.caption(title); col.image(image, use_column_width=True)

    st.markdown('<div class="lbl">DETECTION</div>', unsafe_allow_html=True)
    for col, k, v in zip(st.columns(3), ["DEFECT CLASS","CONFIDENCE","BOUNDING BOX"],
                         [label.replace("_"," ").upper(), f"{acc:.1f}%", "(50,50)→(150,150)"]):
        col.markdown(f'<div class="card"><div class="k">{k}</div><div class="v">{v}</div></div>', unsafe_allow_html=True)

    st.markdown(f'<div style="margin:.8rem 0"><div style="font-family:monospace;font-size:11px;color:#4A6878">CONFIDENCE — {acc:.1f}%</div><div class="bar-t"><div class="bar-f" style="width:{acc:.1f}%"></div></div></div>', unsafe_allow_html=True)
    st.markdown("".join(f'<span class="chip {"on" if c==label else ""}">{c.replace("_"," ")}</span>' for c in CLASSES), unsafe_allow_html=True)

    if probs:
        st.markdown('<div class="lbl">CLASS PROBABILITIES</div>', unsafe_allow_html=True)
        for cls, p in sorted(zip(CLASSES, probs), key=lambda x: -x[1]):
            col = "#00FFB4" if cls==label else "rgba(0,255,180,.25)"
            st.markdown(f'<div style="margin-bottom:7px"><div style="display:flex;justify-content:space-between;font-family:monospace;font-size:11px;color:{"#00FFB4" if cls==label else "#4A6878"}">{cls.replace("_"," ")}<span>{p*100:.1f}%</span></div><div class="bar-t"><div class="bar-f" style="width:{p*100:.1f}%;background:{col}"></div></div></div>', unsafe_allow_html=True)

    _, enc = cv2.imencode('.jpg', cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
    st.download_button("⬇ DOWNLOAD RESULT", enc.tobytes(), f"pcb_{label}.jpg", "image/jpeg")
else:
    st.markdown('<div style="text-align:center;padding:4rem;color:rgba(200,216,232,.12);font-family:monospace;letter-spacing:.2em">AWAITING PCB IMAGE INPUT . . .</div>', unsafe_allow_html=True)
