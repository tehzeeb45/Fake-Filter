# 🚀 Complete Project Architecture & Execution Workflow Guide

This document provides a comprehensive code & architecture breakdown of the **Deepfake Detection System**. It includes the master end-to-end execution flow, file reading sequence, function code blocks, and detailed explanations of every component in the system.

---

## 🗺️ Master End-to-End System Flow Diagram

```
[ USER BROWSER ] ── (Click Scan) ──► [ Scanner.jsx ]
                                           │
                                           │ HTTP POST /api/upload
                                           ▼
                                    [ app.py ] ──► [ core/validator.py ]
                                           │             (Magic Bytes & Size Check)
                                           │ HTTP POST /api/detect
                                           ▼
                                  [ core/detector.py ]
                                           │
               ┌───────────────────────────┼───────────────────────────┐
               ▼                           ▼                           ▼
    [ core/preprocessing.py ]   [ core/models/*.py ]       [ core/gradcam.py ]
    (MTCNN Face Extraction)     (XceptionNet, EffNet, ViT) (Red/Yellow Proof Heatmap)
               │                           │                           │
               └───────────────────────────┼─────────────────────────┘
                                           ▼
                                 [ core/ensemble.py ]
                                 (Soft-Voting Algorithm)
                                           │
                                           ▼
                                    [ core/db.py ] ──► [ data/history.db ]
                                           │
                                           ▼
[ Scanner.jsx ] ◄── (JSON Response) ── [ app.py ]
(Renders REAL/FAKE Verdict & Heatmap Proof)
```

---

## 📋 Diagram Ke Har Box Ka Roman Urdu Mein Detail

### 📦 BOX 1 — `Scanner.jsx` (User Browser)
User website par **"Scan File"** button dabata hai. React component `Scanner.jsx` file ko `FormData` mein pack karta hai aur `POST /api/upload` request backend server (`app.py`) ko bhejta hai.

---

### 📦 BOX 2 — `app.py` → `core/validator.py` (Magic Bytes & Size Check)
`app.py` file receive karne ke baad usse `core/validator.py` ke paas check karne bhejta hai. Validator mein yeh 3 kaam hote hain:
- **File Size Check**: Image `10MB` se badi? Video `100MB` se badi? ❌ REJECT
- **Magic Bytes Check**: File ke pehle 12 bytes read kiye jate hain. `FF D8 FF` = JPEG ✅, `89 50 4E` = PNG ✅, `66 74 79` = MP4 ✅. Koi aur bytes mile toh ❌ REJECT
- **Extension Match**: File ka extension (`.jpg`, `.mp4`) aur actual content match karte hain?

✅ Sab pass hone par file `uploads/<session_id>/` mein save hoti hai aur unique `session_id` wapis bheja jata hai.

---

### 📦 BOX 3 — `core/detector.py` (Master AI Orchestrator)
`POST /api/detect` call aane par `app.py`, `detector.py` ko activate karta hai. Yeh poori AI pipeline ko coordinate karta hai — neechay ke teen modules is ke andar chalte hain:

---

### 📦 BOX 4 — `core/preprocessing.py` (MTCNN Face Extraction)
`detector.py` is module ko pehle call karta hai. Yahan yeh 5 steps hote hain:
1. **MTCNN** algorithm poori image scan karke chehra dhoondta hai
2. Confidence `>= 95%` hona zaruri hai warna error aata hai
3. Aankhon ko detect karke chehra **horizontal level** par rotate kiya jata hai
4. Face ke around **30% margin** add karke crop hoti hai
5. Image `224 × 224` pixels par resize hoti hai
6. **ImageNet normalization** se Tensor `[1, 3, 224, 224]` banta hai — yeh AI models ki input hai

---

### 📦 BOX 5 — `core/models/*.py` (XceptionNet, EffNet, ViT)
Preprocessed face tensor in 4 pre-loaded AI models mein jata hai:

| Model | File | Output |
|:---|:---|:---|
| XceptionNet | `cnn_xception.py` | P(Fake) = 0.97 |
| EfficientNet-B3 | `cnn_efficientnet.py` | P(Fake) = 0.99 |
| ViT | `vit_community.py` | P(Fake) = 0.95 |
| ViT-L/14 | `vit_lnclip.py` | P(Fake) = 0.98 |

Har model **Fake hone ki probability** (0.0 se 1.0) return karta hai.

---

### 📦 BOX 6 — `core/gradcam.py` (Red/Yellow Proof Heatmap)
Models ke sath parallel mein `gradcam.py` XceptionNet ke **conv4 layer** se gradients compute karta hai:
1. `compute_gradcam_heatmap()` → `[224, 224]` grayscale saliency map banta hai
2. `_overlay()` → JET colormap (red = most fake, blue = least fake) face image par blend hoti hai
3. `save_heatmap_overlay()` → Final heatmap `uploads/<session_id>/heatmap.png` save hoti hai

---

### 📦 BOX 7 — `core/ensemble.py` (Soft-Voting Algorithm)
Char models ke scores `ensemble.py` mein aate hain:
```
p_fake = (0.97 × w_cnn + 0.99 × w_effnet + 0.95 × w_vit + 0.98 × w_vitl)
         ÷ (w_cnn + w_effnet + w_vit + w_vitl)
       = 0.9725  →  >= 0.5  →  VERDICT = "FAKE"
       confidence = 0.9725 × 100 = 97.25%
```

---

### 📦 BOX 8 — `core/history.py` → `data/history.db` (Database Save)
Final result ko `add_scan()` function SQLite database mein permanent save karta hai:
- Verdict, Confidence, P(Fake), Model Scores (JSON), Heatmap URL, Created At timestamp

---

### 📦 BOX 9 — `app.py` → `Scanner.jsx` (JSON Response)
`app.py` final JSON response bhejta hai:
```json
{
  "verdict": "FAKE",
  "confidence": 97.25,
  "p_fake": 0.9725,
  "scores": {"cnn": 0.97, "efficientnet": 0.99, "vit": 0.95},
  "face_url": "/media/sid/face_crop.png",
  "heatmap_url": "/media/sid/heatmap.png"
}
```
`Scanner.jsx` yeh data receive karke screen par:
- 🔴 Glowing **FAKE** badge (ya 🟢 **REAL**) dikhata hai
- Confidence counter `0%` se `97.25%` animate hota hai
- 3 Tabs render hote hain: **Source**, **Face Crop**, **Heatmap Proof**

---

## 🛠️ Complete 5-Phase Execution Sequence

### 🟢 Phase 1: Frontend User Interaction (`frontend/src/`)
1. **Routing**: `App.jsx` app layout aur pages (`/`, `/history`, `/metrics`) manage karta hai.
2. **File Selection**: User `Scanner.jsx` upload box mein photo/video drop karta hai. `pickFile()` format (`JPEG/PNG/MP4/AVI`) aur size limit check karta hai.

---

### 🔵 Phase 2: API Gateway & File Security (`app.py` & `core/validator.py`)
1. **Upload Call**: "Scan" button dabane par `Scanner.jsx` `POST /api/upload` request `app.py` ko bhejta hai.
2. **Security Inspection**: `app.py` `core/validator.py` ko call karta hai. `sniff_extension()` file ke shuruati 64 binary bytes (**Magic Bytes**) scan karke virus/fake extension attack se bachata hai. File `uploads/` folder mein temporary save hoti hai aur unique `session_id` milti hai.

---

### 🟣 Phase 3: AI Preprocessing & Face Extraction (`core/preprocessing.py` & `core/detector.py`)
1. **Detection Trigger**: `Scanner.jsx` `POST /api/detect` request `app.py` ko bhejta hai.
2. **Orchestrator Trigger**: `app.py` main AI orchestrator `core/detector.py` ko call karta hai.
3. **MTCNN Face Crop**: `core/preprocessing.py` se MTCNN algorithm media mein se chehra (face) dhoond kar background noise remove karke crop kar leta hai.

---

### 🔴 Phase 4: Multi-Model AI Inference, Ensemble & Grad-CAM (`core/`)
1. **AI Models Feed**: Cropped face ko 4 AI Models (**XceptionNet**, **EfficientNet-B3**, **ViT**, **ViT-L/14**) mein feed kiya jata hai (`core/models/` + `models/*.pt`).
2. **Grad-CAM Proof**: `core/gradcam.py` chehre ke fake parts ko highlight karne ke liye **Red/Yellow Heatmap Image** (`heatmap.png`) generate karta hai.
3. **Ensemble Soft-Voting**: `core/ensemble.py` 4 models ke individual scores par **Weighted Soft-Voting Algorithm** chala kar 1 final **`REAL`** ya **`FAKE`** verdict aur confidence percentage score nikaalta hai.

---

### 🟡 Phase 5: Database Storage & UI Result Display
1. **Database Log**: Result `core/db.py` ke zariye SQLite database (`data/history.db`) mein save hota hai.
2. **JSON Response**: `app.py` final JSON response `Scanner.jsx` ko bhejta hai.
3. **UI Rendering**: `Scanner.jsx` screen par:
   * Glowing Red **`FAKE`** ya Green **`REAL`** Verdict Tag dikhata hai.
   * Animated confidence score (`0.0%` ➔ `99.4%`) counter chalata hai.
   * 3 Visual Tabs (**Source Media**, **MTCNN Cropped Face**, aur **Grad-CAM Heatmap Proof**) render kar deta hai!

---

## 🎯 Code Study Order (Konsi File Pehle Pardhein?):

1. **File 1**: [frontend/src/components/Scanner.jsx](file:///d:/Final%20Fyp/deepfake-detector/frontend/src/components/Scanner.jsx) ➔ Upload Card & User Action (`pickFile`, `detect`, `renderResult`)
2. **File 2**: [app.py](file:///d:/Final%20Fyp/deepfake-detector/app.py) ➔ Backend API Server Gateway (`/api/upload`, `/api/detect`)
3. **File 3**: [core/validator.py](file:///d:/Final%20Fyp/deepfake-detector/core/validator.py) ➔ Security & Magic Bytes Check (`sniff_extension`, `FileValidator`)
4. **File 4**: [core/preprocessing.py](file:///d:/Final%20Fyp/deepfake-detector/core/preprocessing.py) ➔ MTCNN Face Crop (`detect_face`)
5. **File 5**: [core/detector.py](file:///d:/Final%20Fyp/deepfake-detector/core/detector.py) ➔ AI Master Orchestrator (`detect_image`, `detect_video`)
6. **File 6**: [core/ensemble.py](file:///d:/Final%20Fyp/deepfake-detector/core/ensemble.py) & [core/gradcam.py](file:///d:/Final%20Fyp/deepfake-detector/core/gradcam.py) ➔ Voting & Heatmap Proof

---

## 📌 Executive Summary

The frontend is a modern React Single Page Application (SPA) designed for face-first deepfake detection:
1. **Client-Side Routing**: Navigates between the main **Landing** page, **History** page (`/history`), and **Developer Metrics** page (`/metrics`).
2. **Interactive Scroll Animations**: Uses modern `IntersectionObserver` APIs via the `useReveal` hook to reveal UI elements smoothly as the user scrolls.
3. **Animated Stat Counter**: Uses `requestAnimationFrame` via the `CountUp` component to animate statistics (e.g. 98% Accuracy, 99.9 ROC-AUC).
4. **Hero & Scanner Integration**: Renders the core `<Scanner />` component where users upload images/videos for face cropping and deepfake analysis.
5. **Academic & Project Metadata**: Renders FYP team details (Tehzeeb Ul Hassan & Muhammad Khubaib Khalid), supervisor information (M. Adeel Mustafa - IAC Lahore), and project FAQs.

---

## 🛠️ Complete Functions Breakdown & Explanations in `App.jsx`

---

### 1. `useReveal()` — Scroll Animation Hook ([App.jsx: L38-L55](file:///d:/Final%20Fyp/deepfake-detector/frontend/src/App.jsx#L38-L55))

#### Complete Function Code:
```javascript
function useReveal() {
  const ref = useRef(null);
  useEffect(() => {
    const els = ref.current ? ref.current.querySelectorAll(".reveal") : [];
    if (!("IntersectionObserver" in window)) {
      els.forEach(el => el.classList.add("visible"));
      return;
    }
    const obs = new IntersectionObserver(entries => {
      entries.forEach(en => {
        if (en.isIntersecting) { en.target.classList.add("visible"); obs.unobserve(en.target); }
      });
    }, { threshold: 0.12 });
    els.forEach(el => obs.observe(el));
    return () => obs.disconnect();
  }, []);
  return ref;
}
```

#### Function Ka Kaam (Explanation):
Yeh custom React hook web page ke scroll hone par cards aur sections par fade-in/slide-in visual animations apply karta hai. Yeh parent container ka DOM reference (`ref`) hold karta hai aur us ke andar maujood tamam `.reveal` CSS class wale elements par browser ki `IntersectionObserver` API attach karta hai. Jab user scroll karke kisi section par pohonchta hai (12% viewport coverage threshold par), toh yeh function automaticaly `.visible` class attach karke animation trigger kar deta hai aur observation stop kar deta hai taake browser memory aur scroll performance fast rahe.

---

### 2. `CountUp({ to, suffix })` — Animated Number Counter ([App.jsx: L57-L76](file:///d:/Final%20Fyp/deepfake-detector/frontend/src/App.jsx#L57-L76))

#### Complete Function Code:
```javascript
function CountUp({ to, suffix }) {
  const ref = useRef(null);
  const [val, setVal] = useState(0);
  useEffect(() => {
    if (!("IntersectionObserver" in window)) { setVal(to); return; }
    const obs = new IntersectionObserver(entries => {
      if (!entries[0].isIntersecting) return;
      obs.unobserve(entries[0].target);
      const start = performance.now(), dur = 1200;
      (function tick(ts) {
        const p = Math.min((ts - start) / dur, 1);
        setVal(Math.round(to * (1 - Math.pow(1 - p, 3))));
        if (p < 1) requestAnimationFrame(tick);
      })(start);
    }, { threshold: 0.4 });
    if (ref.current) obs.observe(ref.current);
    return () => obs.disconnect();
  }, [to]);
  return <b className="count" ref={ref}>{val}{suffix}</b>;
}
```

#### Function Ka Kaam (Explanation):
Yeh component statistics numbers (maslan 98% Accuracy, 0.999 ROC-AUC, 24 Frames) ko screen par scroll hone par zero se start karke target value tak smoothly animate karke dikhata hai. `IntersectionObserver` ke zariye jab counter card screen par visible hota hai, yeh 1.2 seconds (`dur = 1200`) ke dauran 60fps frame loop (`requestAnimationFrame`) chalta hai aur cubic easing mathematical formula (`1 - (1-p)^3`) se number zero se barha kar final target value par set karta hai.

---

### 3. `scrollToAnalyze()` — Smooth Scroll & Picker Handler ([App.jsx: L88-L95](file:///d:/Final%20Fyp/deepfake-detector/frontend/src/App.jsx#L88-L95))

#### Complete Function Code:
```javascript
  function scrollToAnalyze() {
    const el = document.getElementById("analyze");
    if (el) el.scrollIntoView({ behavior: "smooth" });
    setTimeout(() => {
      const inp = document.getElementById("file-input");
      if (inp) inp.click();
    }, 500);
  }
```

#### Function Ka Kaam (Explanation):
Yeh button click event handler hai jo user ko directly scanner upload section par pohanchata hai. Jab user header ya hero section mein "Analyze now" ya "Scan a file free" button par click karta hai, yeh function pehle page ko smooth animation ke saath `#analyze` DOM element tak scroll karta hai aur 500ms delay ke baad automatically browser ka file selector dialog open kar deta hai taake user foran file choose kar sake.

---

### 4. `Landing()` — Main Landing Page Component ([App.jsx: L78-L295](file:///d:/Final%20Fyp/deepfake-detector/frontend/src/App.jsx#L78-L295))

#### Complete Function Code:
```javascript
function Landing() {
  const reveal = useReveal();
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 10);
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  function scrollToAnalyze() {
    const el = document.getElementById("analyze");
    if (el) el.scrollIntoView({ behavior: "smooth" });
    setTimeout(() => {
      const inp = document.getElementById("file-input");
      if (inp) inp.click();
    }, 500);
  }

  return (
    <div>
      {/* Header, Hero, Scanner, How It Works, Features, FAQ, About, Footer */}
    </div>
  );
}
```

#### Function Ka Kaam (Explanation):
`Landing()` app ka primary UI layout component hai jo poore landing page ko compose karta hai. Is ke andar 2 main functions aur hooks kaam kar rahe hain:
* **Window Scroll Detection (`useEffect`)**: Page scroll position (`window.scrollY > 10`) ko monitor karta hai aur top navigation bar par dynamic glassmorphism CSS class (`topbar scrolled`) apply karta hai. Unmount hone par event listener cleanup kar deta hai.
* **JSX Page Rendering (`return`)**: Topbar Navigation, Hero Section, Core `<Scanner />` component, 3-Step Detection Process, Features Grid, Expandable FAQs list, aur FYP Academic Team Metadata (`Tehzeeb Ul Hassan` & `Muhammad Khubaib Khalid`, Supervisor `M. Adeel Mustafa`) render karta hai.

---

### 5. `App()` — Root Controller & Client-side Router ([App.jsx: L299-L307](file:///d:/Final%20Fyp/deepfake-detector/frontend/src/App.jsx#L299-L307))

#### Complete Function Code:
```javascript
export default function App() {
  const [path, setPath] = useState(window.location.pathname);

  useEffect(() => {
    const onPop = () => setPath(window.location.pathname);
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  return path === "/metrics" ? <Metrics /> : path === "/history" ? <History /> : <Landing />;
}
```

#### Function Ka Kaam (Explanation):
Yeh poori React application ka main Master Entry Point aur Router function hai. Yeh external routing library ke baghair URL path (`window.location.pathname`) track karta hai aur browser ke Back/Forward buttons (`popstate` event) listen karta hai. Conditional ternary logic se decide karta hai ke screen par konsa page render hoga:
* `/metrics` URL par -> Developer Metrics Dashboard (`<Metrics />`)
* `/history` URL par -> Detection History (`<History />`)
* Default URL `/` par -> Main Landing Page (`<Landing />`)

---

## 🌐 Core Detection Component Breakdown: `Scanner.jsx`

`Scanner.jsx` (`frontend/src/components/Scanner.jsx`) application ka main upload box UI card aur detection engine hai.

---

### 1. `useEffect()` Status Polling ([Scanner.jsx: L32-L57](file:///d:/Final%20Fyp/deepfake-detector/frontend/src/components/Scanner.jsx#L32-L57))

#### Complete Function Code:
```javascript
  useEffect(() => {
    let alive = true;
    (function poll() {
      fetch("/api/status")
        .then(r => r.json())
        .then(s => {
          if (!alive) return;
          setDevice(s.requested_device || s.device);
          if (s.ready) { setReady(true); setError(prev => prev === "LOADING" ? "" : prev); }
          else if (s.error) setError("Model load failed: " + s.error);
          else {
            setScanState(s.loading_stage ? "Loading: " + s.loading_stage : "Waiting for a file");
            setTimeout(poll, 2000);
          }
        })
        .catch(() => setTimeout(poll, 3000));
    })();
    return () => { alive = false; };
  }, []);
```

#### Function Ka Kaam (Explanation):
Yeh hook page load hone par background mein Flask backend (`/api/status`) se continuous polling karke dekhta hai ke kya Deep Learning AI Models (XceptionNet, EfficientNet-B3, ViT) RAM/GPU par fully load ho chuke hain ya nahi, aur current execution device (`CPU` ya `CUDA GPU`) ko sync rakhta hai.

---

### 2. `pickFile(f, type)` — Selection & Validation ([Scanner.jsx: L63-L100](file:///d:/Final%20Fyp/deepfake-detector/frontend/src/components/Scanner.jsx#L63-L100))

#### Complete Function Code:
```javascript
  function pickFile(f, type) {
    setError(""); setVerdict(null); setActiveTab(null);
    if (!f) {
      const inp = document.getElementById("file-input");
      if (inp) {
        inp.accept = type === "image" ? ".jpeg,.jpg,.png" : ".mp4,.avi";
        inp.click();
      }
      return;
    }
    if (!isSupported(f.name)) {
      setError("Unsupported format. Please use JPEG, PNG, MP4 or AVI.");
      return;
    }
    const isImg = isImage(f.name);
    if (f.size > maxMB * 1048576) {
      setError("File too large."); return;
    }
    setFile(f);
    setUploadType(isImg ? "image" : "video");
    const url = URL.createObjectURL(f);
    setPreviewUrl(url);
    setScanState("Ready to scan");
  }
```

#### Function Ka Kaam (Explanation):
Yeh function user ki select ki hui file ki validation karta hai. Yeh verify karta hai ke file supported extension (JPEG/PNG for images, MP4/AVI for videos) mein hai ya nahi, aur size limits (Image max 10MB, Video max 100MB) exceed toh nahi kar rahi. Validation pass hone par browser mein Instant File Preview generate karke UI ko `"Ready to scan"` state mein le ata hai.

---

### 3. `detect()` — Detection Pipeline Execution ([Scanner.jsx: L130-L168](file:///d:/Final%20Fyp/deepfake-detector/frontend/src/components/Scanner.jsx#L130-L168))

#### Complete Function Code:
```javascript
  async function detect() {
    if (!file || scanning) return;
    setScanning(true); setError(""); setScanState("Scanning…");
    try {
      const fd = new FormData();
      fd.append("file", file);
      const upRes = await fetch("/api/upload", { method: "POST", body: fd });
      const up = await upRes.json();
      setSessionId(up.session_id);
      
      const detRes = await fetch("/api/detect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: up.session_id })
      });
      const det = await detRes.json();
      renderResult(det.result);
    } catch (err) {
      setError(err.message || "An internal error occurred.");
    } finally {
      setScanning(false);
    }
  }
```

#### Function Ka Kaam (Explanation):
Yeh main asynchronous engine function hai jo detection request chalata hai. Pehle file ko `/api/upload` endpoint par bhej kar unique `session_id` hasil karta hai, phir `/api/detect` bhej kar server-side MTCNN Face Crop, 4-Model Ensemble Soft-voting, aur Grad-CAM Proof generation execute karwata hai. Detection complete hone par result `renderResult()` ko pass kar deta hai.


### 4. `renderResult(res)` & `animateConfidence(target)` ([Scanner.jsx: L170-L191](file:///d:/Final%20Fyp/deepfake-detector/frontend/src/components/Scanner.jsx#L170-L191))

#### Complete Function Code:
```javascript
  function renderResult(res) {
    setVerdict(res);
    setUrls({
      source: res.media_url,
      face: res.face_url,
      heatmap: res.heatmap_url
    });
    setActiveTab("verdict");
    setScanState("Scan complete");
    animateConfidence(res.confidence);
  }

  function animateConfidence(target) {
    const start = performance.now();
    const dur = 1100;
    (function tick(ts) {
      const p = Math.min((ts - start) / dur, 1);
      const eased = 1 - Math.pow(1 - p, 3);
      setConfidence((target * eased).toFixed(1));
      if (p < 1) requestAnimationFrame(tick);
    })(start);
  }
```

#### Function Ka Kaam (Explanation):
Yeh functions detection results ko UI box par display karte hain. `renderResult` backend se mile image URLs (Source Media, MTCNN Cropped Face, Grad-CAM Heatmap Proof) ko set karta hai aur **REAL** / **FAKE** verdict tab active karta hai. `animateConfidence` confidence percentage score ko 1.1 seconds mein zero se animate karke smooth counter dikhata hai.

---

### 5. `switchDevice(target)` — Hardware Device Switcher ([Scanner.jsx: L109-L128](file:///d:/Final%20Fyp/deepfake-detector/frontend/src/components/Scanner.jsx#L109-L128))

#### Complete Function Code:
```javascript
  async function switchDevice(target) {
    if (switching) return;
    setSwitching(true); setError("");
    try {
      const res = await fetch("/api/device", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ device: target })
      });
      const d = await res.json();
      if (!res.ok) throw d;
      setDevice(target === "auto" ? (d.device) : target);
    } catch (err) {
      setError("Device switch failed.");
    } finally {
      setSwitching(false);
    }
  }
```

#### Function Ka Kaam (Explanation):
Yeh function user ko live hardware execution device switch karne ki permission deta hai. Jab user top bar buttons par `CPU` ya `GPU` select karta hai, yeh Flask API (`/api/device`) bhej kar backend execution device ko live switch karwa deta hai.

---

## 🗂️ History Page Component Breakdown: `History.jsx`

`History.jsx` (`frontend/src/pages/History.jsx`) application ka **Past Scans Log & History Page** (`/history`) hai.

---

### 1. `load()` — Fetch History Records ([History.jsx: L23-L39](file:///d:/Final%20Fyp/deepfake-detector/frontend/src/pages/History.jsx#L23-L39))

#### Complete Function Code:
```javascript
  function load() {
    setLoading(true);
    setError("");
    const q = new URLSearchParams();
    if (filterKind) q.set("kind", filterKind);
    if (filterVerdict) q.set("verdict", filterVerdict);
    q.set("limit", "200");
    fetch("/api/history?" + q.toString())
      .then(res => res.json().then(d => { if (!res.ok) throw d; return d; }))
      .then(d => {
        setItems(d.items || []);
        setTotal(d.total || 0);
        if (selected && !d.items.some(x => x.id === selected)) setSelected(null);
      })
      .catch(err => setError((err && err.error && err.error.message) || "Failed to load history."))
      .finally(() => setLoading(false));
  }
```

#### Function Ka Kaam (Explanation):
Yeh function Flask API `/api/history` se pehle kiye gaye tamam scan records (Images/Videos, REAL/FAKE verdicts, confidence %, timestamps) ko SQLite database se fetch karke state mein set karta hai. User jab search filters (Media type: Image/Video, Verdict type: REAL/FAKE) select karta hai, toh `load()` dynamically Query Parameters (`kind`, `verdict`, `limit=200`) ke mutabiq filtered history list load kar ke dikhata hai.

---

### 2. `remove(id)` & `clearAll()` — Record Deletion ([History.jsx: L43-L56](file:///d:/Final%20Fyp/deepfake-detector/frontend/src/pages/History.jsx#L43-L56))

#### Complete Function Code:
```javascript
  function remove(id) {
    fetch("/api/history/" + id, { method: "DELETE" })
      .then(res => res.json().then(d => { if (!res.ok) throw d; return d; }))
      .then(() => { if (selected === id) setSelected(null); load(); })
      .catch(err => setError((err && err.error && err.error.message) || "Delete failed."));
  }

  function clearAll() {
    if (!confirm("Delete all scan history?")) return;
    fetch("/api/history", { method: "DELETE" })
      .then(res => res.json().then(d => { if (!res.ok) throw d; return d; }))
      .then(() => { setSelected(null); load(); })
      .catch(err => setError((err && err.error && err.error.message) || "Clear failed."));
  }
```

#### Function Ka Kaam (Explanation):
* **`remove(id)`**: Jab user history list mein kisi specific item ke cross (`✕`) button par click karta hai, yeh function `DELETE /api/history/:id` API request bhej kar us particular record ko SQLite database se aur UI se permanent delete kar deta hai.
* **`clearAll()`**: Jab user "Clear all" button click karta hai, confirm dialog ke baad `DELETE /api/history` call bhej kar poori detection history flush/delete kar deta hai.

---

### 3. `History()` — Page Component Layout ([History.jsx: L14-L154](file:///d:/Final%20Fyp/deepfake-detector/frontend/src/pages/History.jsx#L14-L154))

#### Complete Function Code:
```javascript
export default function History() {
  const [items, setItems] = useState([]);
  const [total, setTotal] = useState(0);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [filterKind, setFilterKind] = useState("");
  const [filterVerdict, setFilterVerdict] = useState("");
  const [selected, setSelected] = useState(null);

  // load(), remove(), clearAll() functions...

  return (
    <div>
      {/* Topbar, Toolbar Dropdowns, History Items List, Selected Item Detail Panel, Footer */}
    </div>
  );
}
```

#### Function Ka Kaam (Explanation):
Yeh History Page ka primary UI Component hai. Yeh topbar navigation (Back to Detector button), media type dropdown filter, verdict dropdown filter, total scan count badge, scan logs list, aur jab kisi row par click kiya jaye toh side detail panel (jisme original media image, Grad-CAM heatmap, model-wise individual scores list shamil hote hain) render karta hai.

---

## 📊 Developer Metrics Dashboard Component: `Metrics.jsx`

`Metrics.jsx` (`frontend/src/pages/Metrics.jsx`) system ka **Developer Evaluation Dashboard Page** (`/metrics`) hai.

---

### `Metrics()` — Metrics Evaluation & Target Benchmarks ([Metrics.jsx: L11-L91](file:///d:/Final%20Fyp/deepfake-detector/frontend/src/pages/Metrics.jsx#L11-L91))

#### Complete Function Code:
```javascript
export default function Metrics() {
  const [doc, setDoc] = useState(null);
  const [error, setError] = useState("");
  const [active, setActive] = useState(null);

  useEffect(() => {
    fetch("/api/metrics")
      .then(res => res.json().then(d => { if (!res.ok) throw d; return d; }))
      .then(d => {
        setDoc(d);
        const keys = Object.keys(d.models || {});
        if (keys.length) setActive(keys[0]);
      })
      .catch(err => setError("Failed to load metrics."));
  }, []);

  return (
    <div>
      {/* Topbar, Model Selector Tabs, Stats Cards (Accuracy, Precision, Recall, F1, ROC-AUC), Confusion Matrix & ROC Curves */}
    </div>
  );
}
```

#### Function Ka Kaam (Explanation):
`Metrics()` component Python backend API `/api/metrics` se AI models ke test evaluations fetch karta hai. Yeh 4-model ensemble (XceptionNet, EfficientNet-B3, ViT, ViT-L/14) ke darmiyan tabs switch karta hai aur har model ki Test Accuracy, Precision, Recall, F1-Score, aur ROC-AUC target values dikhata hai. Yeh automatically SRS Project Targets (Accuracy ≥ 75%, ROC-AUC ≥ 0.85) ke saath value compare kar ke **`✓ ≥ SRS target`** (Pass) ya Fail tag display karta hai, aur Confusion Matrix Heatmap & ROC Curve Charts display karta hai.

---

## 🏷️ Model Labels Config: `modelLabels.js`

`modelLabels.js` (`frontend/src/modelLabels.js`) AI Deep Learning models ke backend keys ko readable UI names aur format mein convert karta hai.

---

### `formatModelScores(scores)` ([modelLabels.js: L4-L15](file:///d:/Final%20Fyp/deepfake-detector/frontend/src/modelLabels.js#L4-L15))

#### Complete Function Code:
```javascript
export const MODEL_LABELS = {
  cnn: "XceptionNet",
  efficientnet: "EfficientNet-B3",
  vit: "ViT model",
  vit_l14: "ViT-L/14",
};

export function formatModelScores(scores = {}) {
  return Object.entries(scores)
    .map(([k, v]) => (MODEL_LABELS[k] || k) + " " + (+v * 100).toFixed(0) + "%")
    .join(" · ");
}
```

#### Function Ka Kaam (Explanation):
Yeh utility function backend se milne wale raw model score keys (jaise `cnn: 0.99`, `efficientnet: 0.98`) ko clean human-readable text string mein format karta hai: **`XceptionNet 99% · EfficientNet-B3 98% · ViT model 97% · ViT-L/14 99%`**.

---

## 🛡️ Backend Core Module: `core/validator.py` (File Input Validator)

`validator.py` (`core/validator.py`) Flask backend ka **Media Input Validation Engine** (P1.0) hai. Yeh Functional Requirements **FR-01 se FR-04** ko enforce karta hai aur malicious file uploads (e.g. renamed `.exe` ya virus files) se application ko protect karta hai.

---

### 1. `sniff_extension(header: bytes)` — Magic Bytes Sniffer ([validator.py: L47-L60](file:///d:/Final%20Fyp/deepfake-detector/core/validator.py#L47-L60))

#### Complete Code Block:
```python
_JPEG = b"\xff\xd8\xff"
_PNG = b"\x89PNG\r\n\x1a\n"
_AVI = b"RIFF"
_MP4_BRANDS = (b"isom", b"mp41", b"mp42", b"av01", b"iso6", b"MSNV", b"cm")

def sniff_extension(header: bytes) -> str | None:
    """Detect the real container/image format from magic bytes (FR-03)."""
    if not header:
        return None
    if header.startswith(_JPEG):
        return "jpeg"
    if header.startswith(_PNG):
        return "png"
    if header[4:8] in _MP4_BRANDS or header[8:12] in _MP4_BRANDS:
        return "mp4"
    if header.startswith(_AVI) and header[8:12] == b"AVI ":
        return "avi"
    return None
```

#### Function Ka Kaam (Explanation):
Yeh function Security Check (FR-03) ke liye file ke shuruati 64 binary bytes (**Magic Bytes**) ko inspect karta hai. Yeh filename extension par bharosa karne ke bajaye file ke asal binary header signature ko scan karta hai taake asli format (`JPEG`, `PNG`, `MP4`, `AVI`) dhoond sake. Agar koi user kisi dangerous script ya `.exe` file ko rename karke `.jpg` bana kar upload karne ki koshish kare, toh yeh function use Pakad kar reject kar deta hai.

---

### 2. `FileValidator` Class & `validate(filename, data)` Method ([validator.py: L75-L123](file:///d:/Final%20Fyp/deepfake-detector/core/validator.py#L75-L123))

#### Complete Code Block:
```python
class FileValidator:
    """Validates an uploaded file against the SRS upload palette."""

    def __init__(self, cfg: Config):
        up = cfg.uploads
        self.max_image_bytes = int(up.max_image_size_mb) * 1024 * 1024
        self.max_video_bytes = int(up.max_video_size_mb) * 1024 * 1024
        self.allowed_image = frozenset(up.allowed_image_types)
        self.allowed_video = frozenset(up.allowed_video_types)

    def validate(self, filename: str, data: bytes) -> dict:
        """Validate bytes + name; return SRS-compliant metadata or raise."""
        if not data:
            raise ValidationError("Empty file. Please try again.")

        ext = sniff_extension(data[: _HEADER_BYTES])
        if ext is None:
            raise ValidationError("Unsupported format. Accepted: JPEG, PNG, MP4, AVI.")

        mime = EXT_MIME[ext]
        kind = MIME_KIND[mime]

        declared_mime, _ = _mimetypes.guess_type(filename)
        if declared_mime is not None and declared_mime in EXT_MIME.values():
            if declared_mime != mime:
                raise ValidationError("File contents do not match its extension.")

        allowed = self.allowed_image if kind == "image" else self.allowed_video
        if mime not in allowed:
            raise ValidationError("Unsupported format. Accepted: JPEG, PNG, MP4, AVI.")

        if kind == "image":
            if len(data) > self.max_image_bytes:
                raise ValidationError("File too large. Maximum: 10 MB for images.")
        elif len(data) > self.max_video_bytes:
            raise ValidationError("File too large. Maximum: 100 MB for videos.")

        return {
            "kind": kind,
            "extension": ext,
            "mime": mime,
            "size": len(data),
            "original_name": filename,
        }
```

#### Class & Method Ka Kaam (Explanation):
`FileValidator` class system configuration (`config.yaml`) se size limits aur allowed file types load karti hai. Iska main `validate()` method uploaded file stream par 5 sequential security checks chalata hai:
1. **Empty File Check**: Khali (0-byte) files ko foran reject karta hai (`Empty file. Please try again.`).
2. **Magic Bytes Header Sniffing**: Asal file binary structure determine karta hai.
3. **Extension Mismatch Guard (FR-03)**: File name ki extension aur asli binary content agar match na kar rahe hon, toh error throw karta hai (`File contents do not match its extension.`).
4. **MIME Type Validation**: Verify karta hai ke format allowed palette (JPEG, PNG, MP4, AVI) mein se ho.
5. **Strict Size Limits (FR-04)**: Verify karta hai ke image size **10 MB** aur video size **100 MB** se kam ho.

Saare checks pass hone par validated metadata dictionary (`kind`, `extension`, `mime`, `size`, `original_name`) return kar deta hai jo agay MTCNN face extraction pipeline ko di jati hai.

---

## 🧠 Master Detection Module: `core/detector.py` (Inference & Grad-CAM Pipeline)

`detector.py` (`core/detector.py`) poore Detection Engine ka **Master Orchestrator / Chief Director (`Detector` Class)** hai. Yeh MTCNN face detection, multi-model AI inference (XceptionNet, EfficientNet-B3, ViT, ViT-L/14), Grad-CAM heatmap proof generation, aur Ensemble Soft-Voting ko coordinate karta hai.

---

### 💡 High-Level Overview (Pehle Aasan Alfaz Mein):

Jab user kisi image ya video ko scan karta hai, `detector.py` yeh 4 steps sequentially execute karta hai:

```
[ 1. MTCNN Face Extraction ] ➔ [ 2. Multi-Model Inference ] ➔ [ 3. Grad-CAM Heatmap ] ➔ [ 4. Ensemble Verdict ]
```

1. **Chehra Dhoondna (MTCNN)**: Media se face locate karke background remove karke crop karta hai.
2. **AI Models Feed**: Cropped face ko 4 AI Models (**XceptionNet**, **EfficientNet-B3**, **ViT**, **ViT-L/14**) ko bhej kar prediction probabilities hasil karta hai.
3. **Grad-CAM Proof**: Chehre par fake/manipulated regions ko highlight karne ke liye Red/Yellow Heatmap Image (`heatmap.png`) banata hai.
4. **Final Decision (Ensemble)**: Tamam models ke scores ka weighted average nikaal kar **REAL** ya **FAKE** verdict aur confidence score generate karta hai.

---

### 💻 Complete Functions & Methods Breakdown in `detector.py`:

---

### 1. `_load_bundle(cfg, device_override)` — Parallel Model Loader ([detector.py: L43-L71](file:///d:/Final%20Fyp/deepfake-detector/core/detector.py#L43-L71))

#### Complete Code Block:
```python
    @staticmethod
    def _load_bundle(cfg: Config, device_override: str | None = None) -> ModelBundle:
        from concurrent.futures import ThreadPoolExecutor
        from .models import (cnn_efficientnet, cnn_xception, lstm_temporal,
                             vit_community, vit_lnclip, vit_vision)

        device = get_device(device_override or cfg.models.device)
        active = {
            k for k in set(cfg.ensemble.image_weights) | set(cfg.ensemble.video_weights)
            if cfg.ensemble.image_weights.get(k, 0.0) > 0
            or cfg.ensemble.video_weights.get(k, 0.0) > 0
        }
        loaders = {
            "cnn": lambda: cnn_xception.load_cnn(cfg, device=device),
            "efficientnet": lambda: cnn_efficientnet.load_effnet(cfg, device=device),
            "vit": lambda: vit_community.load_community_vit(cfg, device=device),
            "vit_l14": lambda: vit_lnclip.load_lnclip(cfg, device=device),
            "lstm": lambda: lstm_temporal.load_temporal(cfg, device=device),
            "vit_b16": lambda: vit_vision.load_vit(cfg, device=device),
        }
        jobs = {name: fn for name, fn in loaders.items() if name in active}
        with ThreadPoolExecutor(max_workers=min(4, len(jobs))) as pool:
            results = {name: pool.submit(fn) for name, fn in jobs.items()}
            loaded = {name: fut.result() for name, fut in results.items()}
        return ModelBundle(cnn=loaded.get("cnn"), effnet=loaded.get("efficientnet"),
                           vit=loaded.get("vit"), vit_l14=loaded.get("vit_l14"),
                           device=device)
```

#### Function Ka Kaam (Explanation):
Yeh static method startup time speed-up karne ke liye `ThreadPoolExecutor` (multithreading) se active AI Models ko GPU/CPU memory par parallel load karta hai. Yeh config se active models read karke un ki weight checkpoints load karta hai aur unhe single `ModelBundle` object mein organize karta hai.

---

### 2. `detect_image(image_path, artifacts_dir, original_name)` — Image Pipeline ([detector.py: L92-L121](file:///d:/Final%20Fyp/deepfake-detector/core/detector.py#L92-L121))

#### Complete Code Block:
```python
    def detect_image(self, image_path, artifacts_dir,
                     original_name: str | None = None) -> dict:
        """UC-03: XceptionNet + EfficientNet-B3 + ViT, Grad-CAM."""
        with self.lock:
            artifacts = self._artifacts_dir(artifacts_dir)
            bgr = cv2.imread(str(image_path))
            if bgr is None:
                raise RuntimeError("Could not read the uploaded image.")

            face, box, conf = self.pre.detect_face(bgr)          # FR-06..FR-08
            clip = self._to_clip_tensor([face])             # [1,3,224,224]

            with torch.no_grad(), torch.inference_mode():
                p_cnn = torch.sigmoid(self.bundle.cnn(clip)).item()
                p_effnet = torch.sigmoid(self.bundle.effnet(clip)).item()
                p_vit = float(self.bundle.vit.predict_proba([face]).item())

            saliency = compute_gradcam_heatmap(self.bundle.cnn, clip, self.device)
            heat_p = save_heatmap_overlay(saliency, face, artifacts / "heatmap.png")
            crop_p = self._persist_png(face, artifacts / "face_crop.png")

            result = self.ensemble.combine(
                {"cnn": p_cnn, "efficientnet": p_effnet,
                 "vit": p_vit},
                kind="image")
            return self._finalize(result, kind="image", artifacts=artifacts,
                                  original_name=original_name,
                                  heatmap_path=heat_p.name,
                                  face_crop_path=crop_p.name,
                                  frames_analyzed=1)
```

#### Function Ka Kaam (Explanation):
Yeh image detection (UC-03) handler function hai. Yeh image read karke MTCNN se chehra crop karta hai, cropped face tensor ko XceptionNet, EfficientNet-B3, aur ViT models ko pass karta hai, XceptionNet par Grad-CAM heatmap proof generate karke `heatmap.png` save karta hai, aur Ensemble algorithm se **REAL** / **FAKE** verdict dictionary return karta hai.

---

### 3. `detect_video(video_path, artifacts_dir, original_name)` — Video Pipeline ([detector.py: L124-L188](file:///d:/Final%20Fyp/deepfake-detector/core/detector.py#L124-L188))

#### Complete Code Block:
```python
    def detect_video(self, video_path, artifacts_dir,
                     original_name: str | None = None) -> dict:
        """UC-04: frames -> EfficientNet-B3 frame scores + ViT + ViT-L/14 votes."""
        with self.lock:
            artifacts = self._artifacts_dir(artifacts_dir)
            frames = extract_video_frames(
                video_path, fps=self.fps, max_seconds=self.max_duration,
                max_frames=self.max_frames, tmp_dir=artifacts / "frames",
            )
            if not frames:
                raise RuntimeError("Video processing failed. Please try a different file format.")

            faces: list[np.ndarray] = []
            for fp in frames:
                bgr = cv2.imread(str(fp))
                if bgr is None:
                    continue
                try:
                    face, _, _ = self.pre.detect_face(bgr)
                    faces.append(face)
                except NoFaceError:
                    continue

            if not faces:
                raise NoFaceError("No face detected. Please upload media containing a clearly visible human face.")

            clip = self._to_clip_tensor(faces)                  # [T,3,224,224]
            with torch.no_grad(), torch.inference_mode():
                eff_logits = self.bundle.effnet(clip)           # [T,1]
                p_eff_frames = torch.sigmoid(eff_logits).cpu().numpy().reshape(-1)
                p_effnet = float(p_eff_frames.mean())
                p_vit = float(self.bundle.vit.predict_proba(faces).mean())
                p_vit_l14 = float(self.bundle.vit_l14.predict_proba(faces).mean())

            worst = int(np.argmax(p_eff_frames))
            t_worst = self._to_clip_tensor([faces[worst]])  # [1,3,224,224]
            saliency = compute_gradcam_heatmap(self.bundle.cnn, t_worst, self.device)
            heat_p = save_heatmap_overlay(saliency, faces[worst].astype(np.uint8),
                                          artifacts / "heatmap.png")
            crop_p = self._persist_png(faces[0], artifacts / "face_crop.png")

            result = self.ensemble.combine(
                {"efficientnet": p_effnet,
                 "vit": p_vit, "vit_l14": p_vit_l14},
                kind="video"
            )
            return self._finalize(
                result, kind="video", artifacts=artifacts,
                original_name=original_name, heatmap_path=heat_p.name,
                face_crop_path=crop_p.name, frames_analyzed=len(faces),
                most_manipulated_frame=worst + 1,
                analyzed_seconds=round(len(frames) / self.fps, 1),
            )
```

#### Function Ka Kaam (Explanation):
Yeh video detection (UC-04) handler function hai. Yeh video se 5 FPS sampling rate par frames extract karke har frame se chehra crop karta hai. Entire video clip tensor par EfficientNet-B3, ViT, aur ViT-L/14 models se frame-by-frame scores calculate karta hai, sab se ziada manipulated/fake frame (`worst`) par Grad-CAM heatmap proof draw karta hai, aur combined Ensemble decision return karta hai.

---

### 4. `_finalize(result, ...)` — Result Document Assembler ([detector.py: L191-L215](file:///d:/Final%20Fyp/deepfake-detector/core/detector.py#L191-L215))

#### Complete Code Block:
```python
    def _finalize(self, result: dict, *, kind: str, artifacts: Path,
                  original_name: str | None, heatmap_path: str,
                  face_crop_path: str, frames_analyzed: int, **extra) -> dict:
        doc = {
            "kind": kind,
            "verdict": result["verdict"],
            "p_fake": result["p_fake"],
            "confidence": result["confidence"],
            "threshold": result["threshold"],
            "disagreement": bool(result.get("disagreement", False)),
            "scores": result["scores"],
            "cnn_score": result["scores"].get("cnn"),
            "effnet_score": result["scores"].get("efficientnet"),
            "vit_score": result["scores"].get("vit"),
            "vit_l14_score": result["scores"].get("vit_l14"),
            "heatmap_path": heatmap_path,
            "face_crop_path": face_crop_path,
            "faces_analyzed": frames_analyzed,
            "video_fps": self.fps if kind == "video" else None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        doc.update(extra)
        if original_name is not None:
            doc["original_name"] = original_name
        return doc
```

#### Function Ka Kaam (Explanation):
Yeh output dict utility function hai. Yeh final detection dictionary format prepare karta hai jisme verdict (`REAL`/`FAKE`), confidence percentage, threshold, individual model scores, generated heatmap file paths, aur UTC ISO timestamp assemble hota hai.

---

## ✂️ Preprocessing Pipeline Module: core/preprocessing.py

`preprocessing.py` (`core/preprocessing.py`) system ka **Media Preprocessing & Face Extraction Engine** (Level-2 Sub-processes P2.1..P2.4) hai. Yeh raw images aur videos ko downsample, face detect, eye alignment, crop, scale aur ImageNet normalise karke AI models ke ready PyTorch Tensors mein convert karta hai (Functional Requirements **FR-05 se FR-10**).

---

### 💡 High-Level Overview (Pehle Aasan Alfaz Mein):

Jab backend ko user ki upload ki hui media (Image ya Video) milti hai, raw pixels ko seedha AI Models ko nahi bheja ja sakta. `preprocessing.py` yeh 4 steps sequentially execute karta hai:

```
[ 1. Video Frame Extraction ] ➔ [ 2. MTCNN Face Detection ] ➔ [ 3. Eye Alignment & Crop ] ➔ [ 4. Tensor Normalisation ]
```

1. **Video Frame Extraction (FR-05)**: Agar uploaded media video ho, toh yeh FFmpeg (ya OpenCV) ke zariye 5 FPS sampling rate par uniform frames extract karta hai (pehle 60 seconds mein se max 24 frames).
2. **Chehra Dhoondna (MTCNN - FR-06, FR-07)**: Extract hone wale har frame/image par MTCNN (Multi-task Cascaded Convolutional Networks) chala kar human face dhoondta hai. Agar face confidence score 0.95 (95%) se kam ho ya chehra na mile, toh `NoFaceError` exception raise kar deta hai.
3. **Face Alignment & Margin Crop (FR-08)**: Chehre ki aankhon ke landmarks (left & right eye coordinates) dhoond kar Trigonometric Rotation (`atan2`) se face ko rotate karta hai taake dono aankhein perfectly horizontal level par aa jayen, aur margins (padding) add karke face crop kar leta hai.
4. **Resizing & Normalization (FR-09, FR-10)**: Cropped face ko $224 \times 224$ pixels par resize karta hai, pixels ko $[0, 1]$ range par divide karta hai, aur ImageNet Mean ($\mu = [0.485, 0.456, 0.406]$) aur Standard Deviation ($\sigma = [0.229, 0.224, 0.225]$) se normalise karke PyTorch Tensor format (Channel $\times$ Height $\times$ Width) banata hai.

---

### 💻 Complete Functions & Methods Breakdown in `preprocessing.py`:

---

### 1. `find_ffmpeg()` — FFmpeg Binary Auto-Detector ([preprocessing.py: L35-L51](file:///d:/Final%20Fyp/deepfake-detector/core/preprocessing.py#L35-L51))

#### Complete Code Block:
```python
def find_ffmpeg() -> str | None:
    """Return a usable ffmpeg executable path, or None."""
    for probe in ("ffmpeg", "ffmpeg.exe"):
        try:
            r = subprocess.run([probe, "-version"], capture_output=True, timeout=15)
            if r.returncode == 0:
                return probe
        except Exception:
            pass
    if _HAS_IIO_FFMPEG:
        try:
            exe = _iio_ffmpeg.get_ffmpeg_exe()
            if Path(exe).is_file():
                return exe
        except Exception:
            pass
    return None

FFMPEG = find_ffmpeg()
```

#### Function Ka Kaam (Explanation):
Yeh helper function server system par FFmpeg binary executable location dhoondta hai. Yeh sab se pehle System PATH mein `ffmpeg` ya `ffmpeg.exe` ko probe/check karta hai. Agar system PATH par FFmpeg na mile, toh Python package `imageio-ffmpeg` ke bundled ffmpeg binary executable path ko detect karke return karta hai. Agar dono na milein toh `None` return karta hai taake system OpenCV fallback mode par switch kar sake.

---

### 2. `NoFaceError` — Custom Exception Class ([preprocessing.py: L57-L59](file:///d:/Final%20Fyp/deepfake-detector/core/preprocessing.py#L57-L59))

#### Complete Code Block:
```python
class NoFaceError(Exception):
    """FR-07: no face detected in the uploaded media."""
```

#### Function Ka Kaam (Explanation):
Yeh Custom Python Exception Class hai jo Functional Requirement **FR-07** ko implement karti hai. Jab kisi uploaded image ya video frames mein MTCNN ko koi human face na mile ya detection confidence 95% threshold se kam ho, toh system yeh error raise karta hai jisse API user ko clear response message return karti hai: `"No face detected. Please upload media containing a clearly visible human face."`

---

### 3. `extract_video_frames(...)` — Video Frames Sampler ([preprocessing.py: L61-L85](file:///d:/Final%20Fyp/deepfake-detector/core/preprocessing.py#L61-L85))

#### Complete Code Block:
```python
def extract_video_frames(video_path: str | Path, fps: float = 5.0,
                         max_seconds: float | None = 60.0,
                         max_frames: int = 24,
                         tmp_dir: str | Path | None = None) -> list[Path]:
    """Extract at most `max_frames` uniformly-spaced frames from a video.

    FR-05: extraction rate is configurable via `fps`.
    FR-02: only the first `max_seconds` are considered (v1.0 limit).
    Returns a list of PNG frame paths (already downsampled to `max_frames`).
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(video_path)

    tmp_root = Path(tmp_dir) if tmp_dir else Path(tempfile.mkdtemp(prefix="df_frames_"))
    tmp_root.mkdir(parents=True, exist_ok=True)

    frames = _extract_all_frames(video_path, fps, max_seconds, tmp_root)

    # uniform subsample down to max_frames (keeps temporal spread)
    if len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
        frames = [frames[i] for i in idx]
    return frames
```

#### Function Ka Kaam (Explanation):
Yeh main video frame extraction function (FR-05) hai. Yeh video file path receiving ke baad temporary directory banata hai, `_extract_all_frames()` ke zariye target FPS (default 5 FPS) par frames dump karta hai, aur agar extracted frames ki taadad `max_frames` (default 24) se ziada ho, toh NumPy linear spacing (`np.linspace`) se pooray video duration se uniformly spaced 24 frames select kar leta hai taake temporal video flow maintain rahe.

---

### 4. `_extract_all_frames`, `_extract_ffmpeg`, & `_extract_opencv` — Frame Extraction Engines ([preprocessing.py: L87-L137](file:///d:/Final%20Fyp/deepfake-detector/core/preprocessing.py#L87-L137))

#### Complete Code Block:
```python
def _extract_all_frames(video_path: Path, fps: float, max_seconds: float | None,
                        out_dir: Path) -> list[Path]:
    if FFMPEG:
        return _extract_ffmpeg(video_path, fps, max_seconds, out_dir)
    return _extract_opencv(video_path, fps, max_seconds, out_dir)


def _extract_ffmpeg(video_path: Path, fps: float, max_seconds: float | None,
                    out_dir: Path) -> list[Path]:
    """Frame extraction via ffmpeg -vf fps=N (FR-05 default 5 fps)."""
    import re

    pattern = str(out_dir / "frame_%06d.png")
    args = [FFMPEG, "-y", "-i", str(video_path), "-vf", f"fps={fps}"]
    if max_seconds is not None:
        args += ["-t", str(max_seconds)]
    args += ["-q:v", "2", pattern]
    subprocess.run(args, capture_output=True, timeout=600)

    files = sorted(out_dir.glob("frame_*.png"),
                   key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))
    return files


def _extract_opencv(video_path: Path, fps: float, max_seconds: float | None,
                    out_dir: Path) -> list[Path]:
    """OpenCV VideoCapture fallback when no ffmpeg binary is available."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Could not open video file.")
    try:
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        interval = max(1.0, round(src_fps / max(fps, 0.1)))
        step = int(interval)
        frames: list[Path] = []
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_seconds is not None and n / max(src_fps, 1.0) > max_seconds:
                break
            if n % step == 0:
                out = out_dir / f"frame_{len(frames) + 1:06d}.png"
                cv2.imwrite(str(out), frame)
                frames.append(out)
            n += 1
        return frames
    finally:
        cap.release()
```

#### Function Ka Kaam (Explanation):
Yeh functions internal frame extraction execution engines hain:
* **`_extract_all_frames()`**: Smart dispatcher function jo system mein `FFMPEG` available hone par FFmpeg CLI method call karta hai, otherwise OpenCV VideoCapture method par switch ho jata hai.
* **`_extract_ffmpeg()`**: Fast C-based FFmpeg sub-process command execution (`-vf fps=5 -t 60 -q:v 2`) se sub-second speed mein HQ PNG frames decode aur extract karta hai.
* **`_extract_opencv()`**: Pure Python/OpenCV fallback loop. OpenCV `cv2.VideoCapture` se frame-by-frame read karta hai aur frame rate step size math (`src_fps / fps`) compute karke sampling frames disk par PNG format mein write karta hai.

---

### 5. `FacePreprocessor` Class & `__init__(cfg)` ([preprocessing.py: L139-L151](file:///d:/Final%20Fyp/deepfake-detector/core/preprocessing.py#L139-L151))

#### Complete Code Block:
```python
class FacePreprocessor:
    """MTCNN face detection, alignment and tensor normalisation (P2.2-P2.4)."""

    def __init__(self, cfg: Config):
        from facenet_pytorch import MTCNN  # lazy import keeps startup light

        prep = cfg.preprocessing
        self.target_size = int(prep.target_size)
        self.margin = float(prep.face_margin)
        self.conf_threshold = float(prep.mtcnn_confidence)
        self.mean = np.array(prep.imagenet_mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(prep.imagenet_std, dtype=np.float32).reshape(3, 1, 1)
        self._mtcnn = MTCNN(keep_all=False, post_process=False, device="cpu")
```

#### Class & Constructor Ka Kaam (Explanation):
`FacePreprocessor` main class candidate face processing pipeline hai. Constructor mein:
* Lazy import technique se `facenet_pytorch.MTCNN` import karta hai taake startup time load instant rahe.
* Configuration file (`config.yaml`) se parameters parse karta hai: Target Image Size (`224x224`), Bounding Box Margin (`0.2` or 20%), MTCNN Confidence Gate Threshold (`0.95` or 95%), aur ImageNet normalization parameters (Mean & Std arrays reshaped to `[3,1,1]` format).
* PyTorch CPU instance MTCNN neural network initialize karta hai.

---

### 6. `detect_face(image_bgr)` — MTCNN Face Localization ([preprocessing.py: L154-L180](file:///d:/Final%20Fyp/deepfake-detector/core/preprocessing.py#L154-L180))

#### Complete Code Block:
```python
    def detect_face(self, image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Return (aligned_face_rgb, box_xyxy, confidence) for the largest face.

        Raises NoFaceError when no face passes the confidence gate (FR-07).
        """
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        boxes, probs, landmarks = self._mtcnn.detect(rgb, landmarks=True)

        if boxes is None or len(boxes) == 0:
            raise NoFaceError(
                "No face detected. Please upload media containing a clearly visible human face."
            )

        best = 0
        best_conf = float(probs[best])
        for i in range(1, len(boxes)):
            if float(probs[i]) > best_conf:
                best, best_conf = i, float(probs[i])
        if best_conf < self.conf_threshold:
            raise NoFaceError(
                "No face detected. Please upload media containing a clearly visible human face."
            )

        box = boxes[best].astype(int)
        lm = landmarks[best] if landmarks is not None else None
        face = self._align_and_crop(rgb, box, lm)
        return face, box, best_conf
```

#### Function Ka Kaam (Explanation):
Yeh primary MTCNN face detection and confidence verification method (FR-06, FR-07) hai:
1. Input BGR image array ko RGB color space mein convert karta hai.
2. MTCNN neural network execute karke image mein tamam candidate faces ke Bounding Boxes, Probability Scores, aur Facial Landmark Points (Eyes, Nose, Mouth corners) detect karta hai.
3. Agar koi face detect na ho ya MTCNN response empty ho, toh `NoFaceError` raise kar deta hai.
4. Detect hone wale sab se highest confidence/best probability score wale face index ko find karta hai.
5. Verification filter check karta hai: Agar highest confidence score configured threshold (e.g. `0.95`) se kam ho, toh request reject karke `NoFaceError` raise karta hai.
6. Selected face landmark points aur bounding box array ko `_align_and_crop()` method ko pass karke processed face crop return karta hai.

---

### 7. `_align_and_crop` & `_rotate_eyes_horizontal` — Eye Alignment & Margin Cropping ([preprocessing.py: L182-L223](file:///d:/Final%20Fyp/deepfake-detector/core/preprocessing.py#L182-L223))

#### Complete Code Block:
```python
    def _align_and_crop(self, rgb: np.ndarray, box: np.ndarray,
                        landmarks) -> np.ndarray:
        """FR-08: rotate so eyes are horizontal, then crop with margin."""
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        margin_x = int(w * self.margin)
        margin_y = int(h * self.margin)
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(rgb.shape[1] - 1, x2 + margin_x)
        y2 = min(rgb.shape[0] - 1, y2 + margin_y)

        crop = rgb[y1:y2 + 1, x1:x2 + 1]
        if crop.size == 0:
            raise NoFaceError("Face crop out of bounds.")

        if landmarks is not None:
            crop = self._rotate_eyes_horizontal(rgb, landmarks, x1, y1, crop)

        return cv2.resize(crop, (self.target_size, self.target_size),
                          interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _rotate_eyes_horizontal(rgb, landmarks, x1, y1, crop) -> np.ndarray:
        """Rotate around the face center so the eye landmarks are level."""
        try:
            left = landmarks[0] - np.array([x1, y1])
            right = landmarks[1] - np.array([x1, y1])
            dy = right[1] - left[1]
            dx = right[0] - left[0]
            angle = math.degrees(math.atan2(dy, dx))
            if abs(angle) < 0.5:
                return crop
            h, w = crop.shape[:2]
            center = (w / 2.0, h / 2.0)
            m = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(crop, m, (w, h),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE)
            return rotated
        except Exception:
            return crop
```

#### Function Ka Kaam (Explanation):
Yeh methods Facial Alignment aur Bounding Box Margins (FR-08, FR-09) enforce karte hain:
* **Margin Expansion**: Raw Bounding Box coordinates $[x_1, y_1, x_2, y_2]$ ke around 20% margin ($\text{width} \times 0.20$) add karta hai taake forehead, jawline aur hair artifacts crop se bahar na rah jaayen.
* **Eye Alignment (`_rotate_eyes_horizontal`)**: Left eye (Landmark 0) aur Right eye (Landmark 1) ke coordinates ka slope delta ($\Delta y / \Delta x$) nikaal kar `math.atan2` se rotation angle nikaalta hai. Phir OpenCV `cv2.getRotationMatrix2D` aur `cv2.warpAffine` se image tilt correct karta hai taake dono aankhein 0-degree level horizontal line par align ho jayein.
* **Bilinear Scaling (FR-09)**: Final aligned face crop ko OpenCV `cv2.INTER_LINEAR` interpolation filter se exactly $224 \times 224$ pixels grid par scale return karta hai.

---

### 8. `face_to_tensor` & `tensor_for_vis` — Tensor Conversion & Normalization ([preprocessing.py: L226-L235](file:///d:/Final%20Fyp/deepfake-detector/core/preprocessing.py#L226-L235))

#### Complete Code Block:
```python
    def face_to_tensor(self, face_rgb: np.ndarray) -> np.ndarray:
        """FR-09/FR-10: uint8 RGB -> float32 CHW, ImageNet-normalised."""
        img = face_rgb.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = (img - self.mean) / self.std
        return np.ascontiguousarray(img)

    def tensor_for_vis(self, face_rgb: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(face_rgb)  # original uint8 crop for overlay
```

#### Function Ka Kaam (Explanation):
* **`face_to_tensor()` (FR-10)**: Cropped uint8 RGB face image $[224, 224, 3]$ ko $[0.0, 1.0]$ float32 range par scaling karta hai, NumPy `np.transpose` se Height-Width-Channel format ko PyTorch standard **Channel-Height-Width (CHW)** $[3, 224, 224]$ format mein transpose karta hai, aur ImageNet Mean aur Standard Deviation Formula apply karta hai:
  $$\text{Normalized Input} = \frac{\text{Pixel Value} - \mu}{\sigma}$$
* **`tensor_for_vis()`**: Original un-normalized uint8 face array return karta hai jo UI visualization aur Grad-CAM heatmap overlay blend ke kaam aata hai.

---

### 9. `tensor_to_vis(t, mean, std)` — Inverse Normalization Helper ([preprocessing.py: L237-L243](file:///d:/Final%20Fyp/deepfake-detector/core/preprocessing.py#L237-L243))

#### Complete Code Block:
```python
def tensor_to_vis(t: np.ndarray, mean=(0.485, 0.456, 0.406),
                  std=(0.229, 0.224, 0.225)) -> np.ndarray:
    """Inverse ImageNet normalisation (used for Grad-CAM overlays)."""
    img = np.transpose(t, (1, 2, 0)) * np.array(std) + np.array(mean)
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img
```

#### Function Ka Kaam (Explanation):
Yeh standalone utility function **Inverse ImageNet Normalization** perform karta hai. Jab Grad-CAM heatmap engine PyTorch normalized float tensor $[3, 224, 224]$ par visual heatmaps compute kar leta hai, `tensor_to_vis()` mathematical inverse formula execute karta hai:
$$\text{Pixel} = (\text{Tensor} \times \sigma + \mu) \times 255.0$$
Yeh image float tensor ko wapas uint8 RGB array $[224, 224, 3]$ (0..255) image mein convert kar deta hai taake red/yellow Grad-CAM heatmap mask ko original face ke upar blend overlay kiya ja sake.

---

---

## 🌐 Main Backend Web Gateway: `app.py` (Flask Server & API Controller)

`app.py` poore system ka **Main Backend Web Server (Flask API Gateway Controller)** hai. Yeh React Frontend (`frontend/dist`) aur PyTorch AI Engine (`core/detector.py`, `core/validator.py`, `core/db.py`) ke darmiyan **Central Bridge** ka kaam karta hai.

---

### 💡 High-Level Overview (Pehle Aasan Alfaz Mein):

`app.py` ke 4 main responsibilities hain:
1. **Background Model Loading**: Server start hote hi PyTorch AI models (**XceptionNet**, **EfficientNet-B3**, **ViT**) ko non-blocking background thread mein load karta hai.
2. **REST API Endpoints**: Frontend se aane wali Upload (`/api/upload`), Detection (`/api/detect`), Status (`/api/status`), History (`/api/history`), aur Hardware Device Switch (`/api/device`) requests ko handle karta hai.
3. **Session Expiry Sweeper (FR-20)**: Har 5 minute baad 1 ghante se purani temporary upload files ko auto-delete karke disk space clean rakhta hai.
4. **Frontend Serving**: Compiled React UI Single Page Application (`index.html`) aur Grad-CAM Heatmap images serve karta hai.

---

### 💻 Complete Functions & Routes Breakdown in `app.py`:

---

### 1. `start_loading()` & `_load()` — Non-Blocking Model Warmup ([app.py: L70-L90](file:///d:/Final%20Fyp/deepfake-detector/app.py#L70-L90))

#### Complete Code Block:
```python
def start_loading():
    """Load all model weights asynchronously at startup (kept non-blocking)."""
    global _loading, _detector, _detector_error

    if _detector is not None or _detector_error or _loading:
        return
    _loading = True

    def _load():
        global _detector, _detector_error, _loading, _load_stage
        try:
            _load_stage = "Loading model weights (first load can take a few minutes)…"
            _detector = Detector(CFG, device_override=_custom_device)
            _load_stage = "Ready"
        except Exception as exc:  # noqa: BLE001 - surfaced via /api/status
            _detector_error = f"{type(exc).__name__}: {exc}"
        finally:
            _loading = False

    threading.Thread(target=_load, daemon=True).start()
```

#### Function Ka Kaam (Explanation):
Yeh function Flask web server start hone par background daemon thread initiate karta hai jo PyTorch AI Models bundle (`Detector(CFG)`) ko memory (GPU/RAM) mein load karta hai. Thread asynchronous hone ki wajah se web server foran start ho jata hai aur UI block nahi hota.

---

### 2. `_cleanup()`, `_sweep()` & `register_sweeper()` — Session Expiry Sweeper (FR-20) ([app.py: L92-L123](file:///d:/Final%20Fyp/deepfake-detector/app.py#L92-L123))

#### Complete Code Block:
```python
def _cleanup():
    """FR-20: expire session files older than the session TTL (1 hour)."""
    now = time.time()
    with sessions_lock:
        expired = [sid for sid, rec in sessions.items()
                   if rec["expires_at"] <= now]
        for sid in expired:
            sessions.pop(sid, None)
            shutil.rmtree(sessions.get(sid, {}).get("dir") or UPLOAD_ROOT / sid,
                          ignore_errors=True)

    if not UPLOAD_ROOT.is_dir():
        return
    for d in list(UPLOAD_ROOT.iterdir()):
        if d.is_dir():
            try:
                stale = time.time() - d.stat().st_mtime > APP_SESSION_EXPIRY
            except OSError:
                stale = False
            if stale:
                shutil.rmtree(d, ignore_errors=True)

def _sweep():
    while True:
        time.sleep(300)
        _cleanup()

def register_sweeper():
    threading.Thread(target=_sweep, daemon=True).start()
```

#### Function Ka Kaam (Explanation):
Yeh background thread Functional Requirement **FR-20** (Session File Cleanup) ko enforce karta hai. Har 300 seconds (5 minutes) baad `_cleanup()` run hota hai jo 1 ghante (`APP_SESSION_EXPIRY = 3600s`) se purani temporary upload session directories ko `uploads/` folder se hard-delete kar deta hai.

---

### 3. `api_config()` — Upload Limits Endpoint (`GET /api/config`) ([app.py: L199-L206](file:///d:/Final%20Fyp/deepfake-detector/app.py#L199-L206))

#### Complete Code Block:
```python
@app.route("/api/config")
def api_config():
    """Client-side upload limits used by the React UI."""
    return jsonify({
        "maxImageMb": int(CFG.uploads.max_image_size_mb),
        "maxVideoMb": int(CFG.uploads.max_video_size_mb),
        "expiryHours": APP_SESSION_EXPIRY // 3600,
    })
```

#### Function Ka Kaam (Explanation):
Yeh HTTP GET API endpoint React frontend ko `config.yaml` se load shuda file upload limits return karta hai (`maxImageMb: 10`, `maxVideoMb: 100`, `expiryHours: 1`). Frontend `Scanner.jsx` is API call se dynamic limits read karta hai.

---

### 4. `api_status()` — Readiness Probe Endpoint (`GET /api/status`) ([app.py: L209-L227](file:///d:/Final%20Fyp/deepfake-detector/app.py#L209-L227))

#### Complete Code Block:
```python
@app.route("/api/status")
def api_status():
    models = []
    if _detector:
        b = _detector.bundle
        models = [n for n, m in (("cnn", b.cnn),
                             ("efficientnet", b.effnet),
                             ("vit", b.vit),
                             ("vit_l14", b.vit_l14))
                  if m is not None]
    return jsonify({
        "ready": _detector is not None,
        "loading": _loading,
        "loading_stage": _load_stage,
        "error": _detector_error,
        "device": str(_detector.device) if _detector else _resolved_device(),
        "requested_device": _resolved_device(),
        "models_loaded": models,
    })
```

#### Function Ka Kaam (Explanation):
Yeh readiness status endpoint hai jo React frontend dwara har 1.2s baad poll kiya jata hai. Yeh JSON mein batata hai ke models ready hain ya load ho rahe hain (`ready: true/false`), currently loading stage kya hai, aur konsa hardware (`cpu` ya `cuda`) active hai.

---

### 5. `api_upload()` — Media Input Validator & Session Store (`POST /api/upload`) ([app.py: L230-L267](file:///d:/Final%20Fyp/deepfake-detector/app.py#L230-L267))

#### Complete Code Block:
```python
@app.route("/api/upload", methods=["POST"])
def api_upload():
    """UC-01/UC-02: validate the upload and store it under a session."""
    if _detector is None:
        if _detector_error:
            return _error("MODEL_ERROR", _detector_error, 503)
        return _error("LOADING", "Models are still warming up. Please retry.", 503)

    file = request.files.get("file")
    if not file or not file.filename:
        return _error("NO_FILE", "No file selected.", 400)

    data = file.read()
    try:
        meta = validator.validate(file.filename, data)
    except ValidationError as exc:
        return _error("INVALID_FILE", str(exc), 422)

    rec = _new_session()
    rec["media"] = meta
    dir_ = rec["dir"]
    dir_.mkdir(parents=True, exist_ok=True)

    stem = "".join(c if c.isalnum() or c in "._-" else "_"
                   for c in Path(file.filename).stem)[:80] or "upload"
    stored = f"{stem}.{meta['extension']}"
    (dir_ / stored).write_bytes(data)

    return jsonify({
        "session_id": rec["id"],
        "media": {
            "type": meta["kind"], "extension": meta["extension"],
            "mime": meta["mime"], "size": meta["size"],
            "original_name": file.filename,
        },
        "media_url": f"/media/{rec['id']}/{stored}",
        "expires_in": APP_SESSION_EXPIRY,
    })
```

#### Function Ka Kaam (Explanation):
Yeh media file upload handler (UC-01, UC-02) hai. User jab file upload karta hai, yeh `validator.validate()` ([core/validator.py](file:///d:/Final%20Fyp/deepfake-detector/core/validator.py)) se Magic Bytes signature check aur size limits (10MB/100MB) verify karta hai. Fail hone par 422 error throw karta hai, aur pass hone par `uploads/<session_id>/` folder mein file write karke unique `session_id` return karta hai.

---

### 6. `api_detect()` — AI Detection Pipeline Orchestrator (`POST /api/detect`) ([app.py: L270-L318](file:///d:/Final%20Fyp/deepfake-detector/app.py#L270-L318))

#### Complete Code Block:
```python
@app.route("/api/detect", methods=["POST"])
def api_detect():
    """UC-03/UC-04: run the ensemble pipeline and return the verdict."""
    if _detector is None:
        return _error("NOT_READY", "Detection service is not ready yet.", 503)

    payload = request.get_json(silent=True) or {}
    sid = payload.get("session_id", "")
    rec = sessions.get(sid)
    if not rec:
        return _error("BAD_SESSION", "Upload session expired. Please re-upload.", 404)
    dir_ = rec.get("dir")
    if not dir_ or not dir_.is_dir():
        return _error("BAD_SESSION", "Upload session expired. Please re-upload.", 404)

    media = rec.get("media")
    if not media:
        return _error("NO_MEDIA", "No media uploaded in this session.", 400)

    stored = _find_media(dir_, media)
    if stored is None:
        return _error("NO_MEDIA", "Media file is missing from the session.", 500)

    try:
        if media["kind"] == "video":
            result = _detector.detect_video(stored, dir_,
                                            original_name=media["original_name"])
        else:
            result = _detector.detect_image(stored, dir_,
                                            original_name=media["original_name"])
    except NoFaceError as exc:
        return _error("NO_FACE", str(exc), 422)
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("inference failed")
        return _error("INTERNAL",
                      "An internal error occurred. Please try again.", 500)

    result["session_id"] = sid
    result["heatmap_url"] = f"/media/{sid}/{result['heatmap_path']}"
    result["face_url"] = f"/media/{sid}/{result['face_crop_path']}"
    result["media_url"] = f"/media/{sid}/{stored.name}"

    with sessions_lock:
        sessions[sid]["result"] = result

    scan_id = history.add_scan(result)
    result["history_id"] = scan_id

    return jsonify({"ok": True, "result": result})
```

#### Function Ka Kaam (Explanation):
Yeh core AI Detection API Endpoint (UC-03 Image, UC-04 Video) hai. Yeh `session_id` se stored file read karke `_detector.detect_image()` ya `_detector.detect_video()` ([core/detector.py](file:///d:/Final%20Fyp/deepfake-detector/core/detector.py)) ko invoke karta hai. Detection complete hone par MTCNN face crop, Grad-CAM heatmap proof URL, model-wise voting scores, aur REAL/FAKE verdict JSON return karta hai, aur record ko SQLite Database (`history.add_scan()`) mein log kar deta hai.

---

### 7. `api_device_switch()` — Live Hardware Device Switcher (`POST /api/device`) ([app.py: L410-L451](file:///d:/Final%20Fyp/deepfake-detector/app.py#L410-L451))

#### Complete Code Block:
```python
@app.route("/api/device", methods=["POST"])
def api_device_switch():
    """Switch inference device (cpu | cuda | auto) and reload the models."""
    global _custom_device, _detector, _detector_error, _loading, _load_stage

    payload = request.get_json(silent=True) or {}
    device = str(payload.get("device", "")).lower()
    if device not in ("cpu", "cuda", "auto"):
        return _error("BAD_DEVICE", "device must be 'cpu', 'cuda' or 'auto'.", 400)

    if _detector is not None and str(_detector.device) == _resolved_device() and \
            device == (_custom_device or "auto"):
        return jsonify({"ok": True, "device": str(_detector.device),
                        "reload": False})

    if device == "auto":
        _custom_device = None
    else:
        _custom_device = device

    if _detector is None and _loading:
        return jsonify({"ok": True, "device": _resolved_device(),
                        "reload": True, "loading": True})

    _detector, _detector_error, _loading = None, None, True
    _load_stage = f"Switching to {_resolved_device()}…"

    def _load():
        global _detector, _detector_error, _loading, _load_stage
        try:
            _load_stage = f"Loading model weights on {_resolved_device()}…"
            _detector = Detector(CFG, device_override=_custom_device)
            _load_stage = "Ready"
        except Exception as exc:  # noqa: BLE001
            _detector_error = f"{type(exc).__name__}: {exc}"
        finally:
            _loading = False

    threading.Thread(target=_load, daemon=True).start()
    return jsonify({"ok": True, "device": _resolved_device(),
                    "reload": True, "loading": True})
```

#### Function Ka Kaam (Explanation):
Yeh API endpoint user ko React UI (`Scanner.jsx`) par CPU aur GPU (CUDA) buttons click kar ke runtime par hardware execution device live switch karne ki ijazat deta hai. Switch command milne par purane detector object ko unload karke naye device target par PyTorch AI models reload karta hai.

---

### 8. `api_history_*` — SQLite Scan History Logs CRUD (`GET/DELETE /api/history`) ([app.py: L372-L408](file:///d:/Final%20Fyp/deepfake-detector/app.py#L372-L408))

#### Complete Code Block:
```python
@app.route("/api/history")
def api_history_list():
    """List saved detections (newest first), optionally filtered."""
    limit = max(1, min(int(request.args.get("limit", 50)), 200))
    offset = max(0, int(request.args.get("offset", 0)))
    kind = request.args.get("kind") or None
    verdict = request.args.get("verdict") or None
    rows = history.list_scans(limit=limit, offset=offset,
                               kind=kind, verdict=verdict)
    return jsonify({
        "items": rows,
        "total": history.count_scans(),
        "limit": limit,
        "offset": offset,
    })

@app.route("/api/history/<int:scan_id>", methods=["DELETE"])
def api_history_delete(scan_id):
    if not history.delete_scan(scan_id):
        return _error("NOT_FOUND", "Scan not found.", 404)
    return jsonify({"ok": True, "deleted": scan_id})

@app.route("/api/history", methods=["DELETE"])
def api_history_clear():
    deleted = history.clear_all()
    return jsonify({"ok": True, "deleted": deleted})
```

#### Function Ka Kaam (Explanation):
Yeh routes History page ([History.jsx](file:///d:/Final%20Fyp/deepfake-detector/frontend/src/pages/History.jsx)) ke liye CRUD operations provide karti hain:
* `GET /api/history`: SQLite database (`data/history.db`) se past scan logs (newest first) filter/pagination ke sath read karta hai.
* `DELETE /api/history/<id>`: Specific single scan record delete karta hai.
* `DELETE /api/history`: Tamam saved history clear kar deta hai.

---

### 9. `api_metrics()` — Developer Dashboard Metrics (`GET /api/metrics`) ([app.py: L356-L370](file:///d:/Final%20Fyp/deepfake-detector/app.py#L356-L370))

#### Complete Code Block:
```python
@app.route("/api/metrics")
def api_metrics():
    import json

    fp = METRICS_DIR / "metrics.json"
    if not fp.is_file():
        return _error("NO_METRICS",
                      "No evaluation results available. Please run model evaluation first.", 404)
    doc = json.loads(fp.read_text(encoding="utf-8"))
    for key, row in doc.get("models", {}).items():
        row["confusion_url"] = f"/metrics-chart/{row['chart_confusion']}"
        row["roc_url"] = f"/metrics-chart/{row['chart_roc']}"
    return jsonify(doc)
```

#### Function Ka Kaam (Explanation):
Yeh API Developer Metrics Page ([Metrics.jsx](file:///d:/Final%20Fyp/deepfake-detector/frontend/src/pages/Metrics.jsx)) ke liye `metrics/metrics.json` file se evaluation benchmark numbers (Ensemble Accuracy 98.0%, ROC-AUC 0.999) aur Confusion Matrix / ROC Curve PNG chart URLs JSON format mein return karti hai.

---

## 🔴 `core/gradcam.py` Module — Visual Explainability & Proof Heatmap Engine

### Purpose of Module:
Is module ka main maqsad deepfake detection results par **Visual Explainability (Grad-CAM - FR-17)** provide karna hai. Jab AI model kisi photo ko **FAKE** declare karta hai, toh yeh module XceptionNet model ki final conv layer (`conv4`) se activation gradients calculate karke chehre par tampered regions ko **Red/Yellow Color Heatmap (`heatmap.png`)** ke taur par highlight karta hai.

---

### 1. `compute_gradcam_heatmap()` — Saliency Heatmap Generator ([core/gradcam.py: L24-L34](file:///d:/Final%20Fyp/deepfake-detector/core/gradcam.py#L24-L34))

#### Complete Code Block:
```python
def compute_gradcam_heatmap(cnn_model, face_tensor: torch.Tensor,
                            device: torch.device) -> np.ndarray:
    """0..1 saliency map for a [1, 3, 224, 224] ImageNet-normalised tensor."""
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    target_layer = cnn_model.grad_cam_target_layer
    cam = GradCAM(model=cnn_model, target_layers=[target_layer])
    grayscale = cam(input_tensor=face_tensor.to(device),
                    targets=[ClassifierOutputTarget(0)])
    return grayscale[0]
```

#### Function Ka Kaam (Explanation):
Yeh function XceptionNet model ki target conv layer (`conv4`) ko Grad-CAM engine ke sath hook karta hai aur face tensor `[1, 3, 224, 224]` ko pass karke 2D Grayscale Saliency Map ($0.0 \dots 1.0$) calculate karta hai jo AI model ke focus region (dhyan) ko represent karta hai.

---

### 2. `_overlay()` — JET Color Map Blending ([core/gradcam.py: L15-L21](file:///d:/Final%20Fyp/deepfake-detector/core/gradcam.py#L15-L21))

#### Complete Code Block:
```python
def _overlay(heatmap: np.ndarray, canvas_rgb: np.ndarray,
             alpha: float = 0.5) -> np.ndarray:
    """Blend a 0..1 heatmap onto a 0..255 RGB image (jet colormap)."""
    heat = np.uint8(255 * np.clip(heatmap, 0.0, 1.0))
    jet = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    jet = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(canvas_rgb, 1.0 - alpha, jet, alpha, 0)
```

#### Function Ka Kaam (Explanation):
Yeh function Grayscale Saliency Map ko OpenCV `COLORMAP_JET` (Red/Yellow/Blue) color palette mein convert karta hai aur `cv2.addWeighted()` ke zariye original cropped face image aur color heatmap ko 50%-50% blend kar deta hai taake chehre ke upar fake spots laal rang mein dikhayi dein.

---

### 3. `save_heatmap_overlay()` — Heatmap Disk Persistence ([core/gradcam.py: L37-L44](file:///d:/Final%20Fyp/deepfake-detector/core/gradcam.py#L37-L44))

#### Complete Code Block:
```python
def save_heatmap_overlay(saliency: np.ndarray, face_rgb: np.ndarray,
                         out_png: str | Path, alpha: float = 0.5) -> Path:
    """Persist the Grad-CAM overlay PNG (FR-17)."""
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    overlay = _overlay(saliency, np.ascontiguousarray(face_rgb), alpha=alpha)
    cv2.imwrite(str(out_png), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return out_png
```

#### Function Ka Kaam (Explanation):
Yeh function blended heatmap array ko hard disk par `uploads/<session_id>/heatmap.png` file ke taur par save karta hai taake React UI ([Scanner.jsx](file:///d:/Final%20Fyp/deepfake-detector/frontend/src/components/Scanner.jsx)) par **Grad-CAM Proof Tab** mein red/yellow heatmap overlay image render ho sake.

---

## ⚖️ `core/ensemble.py` Module — Weighted Soft-Voting Decision Engine

### Purpose of Module:
Is module ka main maqsad multimodality prediction outputs ko combine karke 1 final **`REAL`** ya **`FAKE`** verdict aur confidence percentage score ($0.0\% \dots 100.0\%$) generate karna hai (**FR-14..FR-16**). Yeh module XceptionNet, EfficientNet-B3, ViT, aur ViT-L/14 models ke individual scores ko `config.yaml` ke weights ke mutabiq weighted soft-voting se combine karta hai.

---

### 1. `weighted_average()` — Weighted Mean Calculator ([core/ensemble.py: L17-L27](file:///d:/Final%20Fyp/deepfake-detector/core/ensemble.py#L17-L27))

#### Complete Code Block:
```python
def weighted_average(scores: dict[str, float], weights: dict[str, float]) -> float:
    """Weighted mean of P(Fake) over the models actually present in `scores`."""
    total_w = 0.0
    acc = 0.0
    for key, p in scores.items():
        w = weights.get(key, 0.0)
        acc += w * p
        total_w += w
    if total_w <= 0:
        raise ValueError("ensemble: no weighted model scores provided")
    return acc / total_w
```

#### Function Ka Kaam (Explanation):
Yeh function tamam active AI models ke individual fake scores (e.g. XceptionNet: 0.95, EfficientNet: 0.98, ViT: 0.92) ko unke assigned weights ke sath multiply kar ke ek single **Average Ensemble Fake Score** ($0.0 \dots 1.0$) calculate karta hai.

---

### 2. `decision_from_p()` — Verdict & Confidence Record Builder ([core/ensemble.py: L30-L39](file:///d:/Final%20Fyp/deepfake-detector/core/ensemble.py#L30-L39))

#### Complete Code Block:
```python
def decision_from_p(p_fake: float, threshold: float = 0.5) -> dict:
    """Build the final verdict + confidence record (FR-15/FR-16)."""
    verdict = "FAKE" if p_fake >= threshold else "REAL"
    confidence = p_fake * 100.0 if verdict == "FAKE" else (1.0 - p_fake) * 100.0
    return {
        "verdict": verdict,
        "p_fake": round(float(p_fake), 4),
        "confidence": round(float(confidence), 2),
        "threshold": threshold,
    }
```

#### Function Ka Kaam (Explanation):
Yeh function ensemble fake probability score $P_{\text{fake}}$ ko $0.5$ threshold se compare karta hai:
* Agar $P_{\text{fake}} \ge 0.5$ ➔ Verdict **`FAKE`**, Confidence $= P_{\text{fake}} \times 100$.
* Agar $P_{\text{fake}} < 0.5$ ➔ Verdict **`REAL`**, Confidence $= (1 - P_{\text{fake}}) \times 100$.

---

### 3. `Ensemble` Class — Soft-Voting Orchestrator ([core/ensemble.py: L42-L67](file:///d:/Final%20Fyp/deepfake-detector/core/ensemble.py#L42-L67))

#### Complete Code Block:
```python
class Ensemble:
    """Configured weighted-soft-voting helper."""

    def __init__(self, cfg: Config):
        self.threshold = float(cfg.ensemble.threshold)
        self.image_weights = {k: float(v) for k, v in cfg.ensemble.image_weights.items()}
        self.video_weights = {k: float(v) for k, v in cfg.ensemble.video_weights.items()}

    def combine(self, scores: dict[str, float], kind: str) -> dict:
        weights = self.video_weights if kind == "video" else self.image_weights
        p_fake = weighted_average(scores, weights)
        result = self.to_verdict(p_fake)
        result["scores"] = {k: round(v, 4) for k, v in scores.items()}
        result["disagreement"] = False
        return result

    def to_verdict(self, p_fake: float) -> dict:
        return {
            "verdict": "FAKE" if p_fake >= self.threshold else "REAL",
            "p_fake": round(float(p_fake), 4),
            "confidence": round(
                p_fake * 100.0 if p_fake >= self.threshold else (1.0 - p_fake) * 100.0,
                2,
            ),
            "threshold": self.threshold,
        }
```

#### Class Ka Kaam (Explanation):
Yeh class `config.yaml` se Image aur Video weights load karti hai. `combine()` method media type ke mutabiq correct weights select karke `weighted_average()` aur `to_verdict()` dwara final verdict, confidence %, aur individual model scores ka clean dictionary object assemble karta hai.

---

## 💾 `core/history.py` Module — SQLite Persistent Scan History Manager

### Purpose of Module:
Is module ka main maqsad tamam completed detection scans (images & videos) ki history ko **SQLite Database (`data/history.db`)** mein permanently save, retrieve, filter, aur delete karna hai. Yeh multi-threading safety ke liye RLock aur Write-Ahead Logging (`WAL`) mode use karta hai.

---

### 1. `_connect()` — Safe SQLite Connection Helper ([core/history.py: L43-L48](file:///d:/Final%20Fyp/deepfake-detector/core/history.py#L43-L48))

#### Complete Code Block:
```python
def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn
```

#### Function Ka Kaam (Explanation):
Yeh function target directory `data/` ko verify karta hai, SQLite database `data/history.db` se connection establish karta hai, column name access (`sqlite3.Row`) active karta hai, aur high-performance WAL mode set karta hai taake simultaneous read/write operations safe aur fast hon.

---

### 2. `init_history()` — Database Schema Initializer ([core/history.py: L51-L56](file:///d:/Final%20Fyp/deepfake-detector/core/history.py#L51-L56))

#### Complete Code Block:
```python
def init_history(db_path=None) -> Path:
    """Create the schema if missing and return the resolved DB path."""
    path = Path(db_path) if db_path else resolve("data") / "history.db"
    with _lock, _connect(path) as conn:
        conn.executescript(_SCHEMA)
    return path
```

#### Function Ka Kaam (Explanation):
Yeh function server startup par `_SCHEMA` execute karke `scans` table aur `created_at` index create/verify karta hai taake scan results save karne ke liye database schema tayyar rahe.

---

### 3. `add_scan()` — Scan Log Inserter ([core/history.py: L59-L88](file:///d:/Final%20Fyp/deepfake-detector/core/history.py#L59-L88))

#### Complete Code Block:
```python
def add_scan(record: dict, db_path=None) -> int:
    """Insert one detection result. `record` is the API result dict."""
    path = Path(db_path) if db_path else resolve("data") / "history.db"
    with _lock, _connect(path) as conn:
        cur = conn.execute(
            """
            INSERT INTO scans
                (session_id, kind, original_name, verdict, confidence,
                 p_fake, threshold, scores, frames_analyzed,
                 media_url, face_url, heatmap_url, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.get("session_id", ""),
                record.get("kind", "image"),
                record.get("original_name", ""),
                record.get("verdict", "REAL"),
                float(record.get("confidence", 0.0)),
                float(record.get("p_fake", 0.0)),
                float(record.get("threshold", 0.5)),
                json.dumps(record.get("scores") or {}),
                record.get("faces_analyzed"),
                record.get("media_url", ""),
                record.get("face_url"),
                record.get("heatmap_url"),
                record.get("created_at") or datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)
```

#### Function Ka Kaam (Explanation):
Yeh function detection scan complete hone par final API result dictionary ko SQLite database ki `scans` table mein 1 new row ke taur par insert (save) karta hai aur unique row ID return karta hai.

---

### 4. `list_scans()` — Scan Logs Query & Pagination ([core/history.py: L91-L109](file:///d:/Final%20Fyp/deepfake-detector/core/history.py#L91-L109))

#### Complete Code Block:
```python
def list_scans(limit: int = 100, offset: int = 0,
               kind: str | None = None, verdict: str | None = None,
               db_path=None) -> list[dict]:
    """Return scan history rows newest-first, with the SQLite integer id."""
    path = Path(db_path) if db_path else resolve("data") / "history.db"
    sql = "SELECT * FROM scans WHERE 1=1"
    params: list = []
    if kind in ("image", "video"):
        sql += " AND kind = ?"
        params.append(kind)
    if verdict in ("REAL", "FAKE"):
        sql += " AND verdict = ?"
        params.append(verdict)
    sql += " ORDER BY id DESC LIMIT ? OFFSET ?"
    params += [int(limit), int(offset)]

    with _lock, _connect(path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [_row_to_dict(r) for r in rows]
```

#### Function Ka Kaam (Explanation):
Yeh function History page ([History.jsx](file:///d:/Final%20Fyp/deepfake-detector/frontend/src/pages/History.jsx)) ke liye database se saved scan logs newest-first order mein fetch karta hai. Is mein `kind` (`image`/`video`), `verdict` (`REAL`/`FAKE`), aur pagination (`limit`/`offset`) filters shamil hain.

---

### 5. `get_scan()`, `count_scans()`, `delete_scan()`, `clear_all()`, & `_row_to_dict()` ([core/history.py: L112-L148](file:///d:/Final%20Fyp/deepfake-detector/core/history.py#L112-L148))

#### Complete Code Block:
```python
def get_scan(scan_id: int, db_path=None) -> dict | None:
    path = Path(db_path) if db_path else resolve("data") / "history.db"
    with _lock, _connect(path) as conn:
        row = conn.execute("SELECT * FROM scans WHERE id = ?",
                           (int(scan_id),)).fetchone()
    return _row_to_dict(row) if row else None

def count_scans(db_path=None) -> int:
    path = Path(db_path) if db_path else resolve("data") / "history.db"
    with _lock, _connect(path) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM scans").fetchone()[0])

def delete_scan(scan_id: int, db_path=None) -> bool:
    path = Path(db_path) if db_path else resolve("data") / "history.db"
    with _lock, _connect(path) as conn:
        cur = conn.execute("DELETE FROM scans WHERE id = ?", (int(scan_id),))
        conn.commit()
    return cur.rowcount > 0

def clear_all(db_path=None) -> int:
    path = Path(db_path) if db_path else resolve("data") / "history.db"
    with _lock, _connect(path) as conn:
        cur = conn.execute("DELETE FROM scans")
        conn.commit()
    return int(cur.rowcount)

def _row_to_dict(row: sqlite3.Row) -> dict:
    d = {k: row[k] for k in row.keys()}
    try:
        d["scores"] = json.loads(d["scores"]) if d.get("scores") else {}
    except (ValueError, TypeError):
        d["scores"] = {}
    return d
```

#### Function Ka Kaam (Explanation):
* `get_scan()`: Specific ID par Single scan log fetch karta hai.
* `count_scans()`: Total saved scans count return karta hai.
* `delete_scan()`: Specific single scan record delete karta hai.
* `clear_all()`: Tamam saved scan history clear kar deta hai.
* `_row_to_dict()`: Raw SQLite row ko Python/JSON dictionary mein format karta hai.

---

## 📊 Complete Summary Table of All Project Files & Functions

| File Path | Function / Class Name | Overall Responsibility / Kaam |
| :--- | :--- | :--- |
| `history.py` | `init_history()` | Creates SQLite database `data/history.db` and `scans` table on server startup. |
| `history.py` | `add_scan()` | Inserts completed detection scan record into SQLite database table. |
| `history.py` | `list_scans()` | Queries saved scan history rows newest-first with kind/verdict filters & pagination. |
| `history.py` | `delete_scan()` / `clear_all()` | Deletes a single scan record or clears all history records from SQLite database. |
| `ensemble.py` | `weighted_average()` | Computes weighted mean P(Fake) over active model scores. |
| `ensemble.py` | `decision_from_p()` | Determines REAL/FAKE verdict and confidence % based on 0.5 threshold. |
| `ensemble.py` | `Ensemble.combine()` | Combines multi-model outputs for images/videos into structured verdict result. |
| `gradcam.py` | `compute_gradcam_heatmap()` | Generates 2D grayscale saliency map from XceptionNet conv4 layer gradients. |
| `gradcam.py` | `_overlay()` | Blends 0..1 saliency map onto face RGB image using OpenCV JET colormap (Red/Yellow/Blue). |
| `gradcam.py` | `save_heatmap_overlay()` | Saves blended Grad-CAM heatmap PNG (`heatmap.png`) to upload session directory on disk. |
| `App.jsx` | `App()` | Root component & client-side router between `/`, `/history`, and `/metrics`. |
| `App.jsx` | `Landing()` | Renders complete landing page structure, topbar glass effect, & section cards. |
| `App.jsx` | `useReveal()` | Custom Hook: Triggers scroll reveal animations when elements enter viewport. |
| `App.jsx` | `CountUp()` | Animates stat numbers from 0 to target value using cubic easing 60fps loop. |
| `App.jsx` | `scrollToAnalyze()` | Smooth scrolls page to `#analyze` scanner section and opens file dialog. |
| `Scanner.jsx` | `useEffect()` | Polls `/api/status` for backend model readiness and CPU/GPU hardware status. |
| `Scanner.jsx` | `pickFile()` | Validates file format (JPEG/PNG/MP4/AVI), size limits, & generates preview. |
| `Scanner.jsx` | `detect()` | Uploads file & executes MTCNN face crop, 4-model ensemble, & Grad-CAM proof. |
| `Scanner.jsx` | `renderResult()` | Displays REAL/FAKE verdict, face crop, heatmap proof, & tab navigation. |
| `Scanner.jsx` | `animateConfidence()` | Animates confidence score % from 0 to target value in 1.1 seconds. |
| `Scanner.jsx` | `switchDevice()` | Live switches hardware execution device between CPU and GPU via `/api/device`. |
| `History.jsx` | `load()` | Fetches past scan logs from SQLite database with type and verdict filters. |
| `History.jsx` | `remove()` | Deletes a specific scan record from SQLite database and UI list view. |
| `History.jsx` | `clearAll()` | Deletes all recorded scan history from database after user confirmation. |
| `History.jsx` | `History()` | Renders scan history log page, filter dropdowns, & detailed heatmap panel. |
| `Metrics.jsx` | `Metrics()` | Displays developer metrics, SRS benchmark targets, confusion matrices & ROC curves. |
| `modelLabels.js` | `formatModelScores()` | Formats raw model keys into clean human-readable score string. |
| `style.css` | `:root` & Design System | Dark mode palette, glassmorphism backdrop blur, gradients, & scroll animations. |
| `validator.py` | `sniff_extension()` | Inspects file Magic Bytes header to detect real file container signature. |
| `validator.py` | `FileValidator.validate()` | Validates uploaded file against FR-01..FR-04 size (10MB/100MB) & extension spoofing guards. |
| `detector.py` | `_load_bundle()` | Parallel multithreaded AI models weight checkpoints loader into memory. |
| `detector.py` | `detect_image()` | UC-03 Image detection pipeline (MTCNN crop, 3-model inference, Grad-CAM proof). |
| `detector.py` | `detect_video()` | UC-04 Video detection pipeline (5fps frame extraction, multi-model vote, worst-frame heatmap). |
| `detector.py` | `_finalize()` | Assembles final detection dict with verdict, confidence %, and heatmap file paths. |
| `preprocessing.py` | `find_ffmpeg()` | Detects system FFmpeg executable or bundled imageio-ffmpeg binary path. |
| `preprocessing.py` | `extract_video_frames()` | Extracts & downsamples frames uniformly up to 24 frames at configurable FPS. |
| `preprocessing.py` | `FacePreprocessor.detect_face()` | Runs MTCNN face localization, confidence filtering (>= 0.95), & face alignment. |
| `preprocessing.py` | `FacePreprocessor._align_and_crop()` | Trigonometric eye horizontal rotation, 20% margin crop, & 224x224 resize. |
| `preprocessing.py` | `FacePreprocessor.face_to_tensor()` | Converts uint8 HWC RGB array to float32 CHW PyTorch tensor with ImageNet normalization. |
| `preprocessing.py` | `tensor_to_vis()` | Inverse ImageNet normalization helper for Grad-CAM overlay visualization. |
| `app.py` | `start_loading()` | Asynchronous non-blocking background thread model warmup loader on startup. |
| `app.py` | `_cleanup()` / `_sweep()` | FR-20 session TTL sweeper thread auto-deleting upload session files older than 1 hour. |
| `app.py` | `api_config()` | `GET /api/config` returning 10MB/100MB file upload limits to React UI. |
| `app.py` | `api_status()` | `GET /api/status` readiness probe returning model load state and CPU/GPU device. |
| `app.py` | `api_upload()` | `POST /api/upload` validating raw file bytes and storing temporary session file. |
| `app.py` | `api_detect()` | `POST /api/detect` orchestrating MTCNN, 4-model ensemble, Grad-CAM proof & SQLite logging. |
| `app.py` | `api_device_switch()` | `POST /api/device` live switching hardware execution device (`cpu` / `cuda` / `auto`). |
| `app.py` | `api_history_*()` | `GET/DELETE /api/history` CRUD endpoints for SQLite scan logs. |
| `app.py` | `api_metrics()` | `GET /api/metrics` serving evaluation benchmark numbers (98% accuracy) & PNG charts. |
| `main.jsx` | `ReactDOM.render` | Mounts root `<App />` component into HTML `#root` container. |

---

*Generated for Deepfake Detection System Final Year Project (IAC, Lahore).*





