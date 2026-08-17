# 🛠️ Project Technologies & Language Choices

Is document mein yeh bataya gaya hai ke hamare Deepfake Detection System mein **Frontend** aur **Backend** par kaunsi technologies use hui hain, kyun use ki gayi hain, aur alternatives se behtar kyun hain.

---

## 🖥️ FRONTEND Technologies

Frontend woh hissa hai jo **user screen par dekhta hai** — website ka design, buttons, upload box, results page sab frontend hai.

---

### 1. React (JSX) — UI Framework

**Kya hai?**
React ek JavaScript library hai jo Facebook/Meta ne banai. Isse Component-Based UI banate hain — matlab har cheez (Scanner card, History list, Metrics page) ek alag reusable piece hoti hai.

**Hamare Project Mein Kahan Use Hua?**

| File | Kaam |
|:---|:---|
| `App.jsx` | Root component — routing aur landing page |
| `Scanner.jsx` | File upload, AI scan, result display |
| `History.jsx` | Past scan history page |
| `Metrics.jsx` | Developer performance dashboard |

**Kyun React? Alternatives Se Behtar Kyun?**

| Alternative | Kyun Nahi Chuna? |
|:---|:---|
| Plain HTML/JS | Complex dynamic UI mushkil hoti — scan progress, real-time tabs, animated confidence counter sirf HTML mein manage karna impossible tha |
| Vue.js | React se kam popular, kam community support |
| Angular | Bahut heavy framework — hamare simple SPA ke liye overkill |
| Next.js | Server-side rendering ki zarurat nahi thi — hamare paas Flask backend hai |

**React Best Tha Kyunke:**
- `useState` se scan progress, tabs, verdict real-time update hote hain
- Components reusable hain — Scanner ek baar bana, kahin bhi use karo
- Duniya ki sabse popular frontend library
- Vite ke sath fast development server milta hai

---

### 2. JavaScript (JS) — Programming Logic

**Kya hai?**
JavaScript browser mein chalne wali language hai. React bhi JavaScript par hi bana hai.

**Hamare Project Mein Kahan Use Hua?**
- `fetch()` API calls — backend ko HTTP request bhejna
- File validation logic — size check, format check
- `URL.createObjectURL()` — file ka local preview banana
- `setInterval()` — scan progress animation

**Kyun JavaScript?**
Browser sirf JavaScript samajhta hai — koi alternative nahi! TypeScript better hota lekin FYP ke liye extra complexity thi.

---

### 3. CSS — Styling & Design

**Kya hai?**
CSS se website ka visual design hota hai — colors, fonts, layout, animations.

**Hamare Project Mein Kahan Use Hua?**
- `style.css` — Dark theme, glassmorphism cards, gradient buttons
- Responsive layout — mobile aur desktop dono par sahi dikhna
- Hover animations, scan progress loader

**Kyun Vanilla CSS?**

| Alternative | Kyun Nahi? |
|:---|:---|
| TailwindCSS | Utility classes se custom design time-consuming tha |
| Bootstrap | Generic look — hamare premium dark design ke liye sahi nahi |
| SCSS/SASS | Extra build step — simple project ke liye overkill |

**Vanilla CSS Best Tha Kyunke:** Full control tha design par — custom dark theme, glassmorphism, aur animations bilkul apni marzi se banaye.

---

### 4. Vite — Frontend Build Tool

**Kya hai?**
Vite ek fast development server aur build tool hai jo React project ko run aur build karta hai.

**Kyun Vite?**

| Alternative | Kyun Nahi? |
|:---|:---|
| Create React App (CRA) | Slow, outdated, officially deprecated |
| Webpack | Complex configuration, slow hot reload |

**Vite Best Tha Kyunke:** Ultra-fast hot module replacement (HMR) — code change hote hi instantly browser update hota hai.

---

## ⚙️ BACKEND Technologies

Backend woh hissa hai jo **server par chalta hai** — AI models, database, file processing, API — sab backend mein hai.

---

### 1. Python — Main Language

**Kya hai?**
Python ek high-level programming language hai jo AI/ML ke liye duniya mein #1 hai.

**Hamare Project Mein Kahan Use Hua?**

| File | Kaam |
|:---|:---|
| `app.py` | Flask web server — API routes |
| `core/detector.py` | AI detection pipeline |
| `core/preprocessing.py` | Face extraction |
| `core/ensemble.py` | Soft-voting algorithm |
| `core/gradcam.py` | Heatmap generation |
| `core/history.py` | Database operations |
| `core/metrics.py` | Performance evaluation |

**Kyun Python?**

| Alternative | Kyun Nahi? |
|:---|:---|
| Node.js | PyTorch Node mein nahi chalta — AI ecosystem Python mein hai |
| Java | PyTorch Java support limited hai, complex setup |
| C#/.NET | ML libraries Python mein hain |
| Go/Rust | AI/ML libraries available nahi hain |
| PHP | Scientific computing support nahi |

**Python ZARURI Tha Kyunke:**
- PyTorch — AI models sirf Python mein properly chalte hain
- MTCNN — Face detection Python library
- OpenCV — Image processing
- NumPy — Array/matrix operations
- scikit-learn — Metrics calculation
- Duniya ka sabse bada AI/ML ecosystem Python mein hai

---

### 2. Flask — Web Framework

**Kya hai?**
Flask ek lightweight Python web framework hai jo HTTP API routes banata hai.

**Hamare Project Mein Kahan Use Hua?**
- `/api/upload` — File upload endpoint
- `/api/detect` — AI detection endpoint
- `/api/history` — History CRUD endpoints
- `/api/metrics` — Metrics dashboard endpoint
- Static React files serve karna

**Kyun Flask?**

| Alternative | Kyun Nahi? |
|:---|:---|
| Django | Bahut heavy — full ORM, admin panel — hamare paas sirf REST API chahiye thi |
| FastAPI | Flask se slightly better performance, lekin Flask zyada familiar tha |
| Tornado | Complex, async management mushkil |

**Flask Best Tha Kyunke:** Minimal code mein REST API banti hai — `@app.route` decorator se kaam ho jata hai. Lightweight aur simple.

---

### 3. PyTorch — AI/Deep Learning Framework

**Kya hai?**
PyTorch Facebook ka Deep Learning framework hai jis par hamare charon AI models chalte hain.

**Hamare Project Mein Kahan Use Hua?**
- `torch.load()` — Model weights `.pt` file se load karna
- `model.forward(tensor)` — Image tensor par AI inference
- `torch.no_grad()` — Memory-efficient prediction mode
- `torch.device()` — CPU/GPU management

**Kyun PyTorch?**

| Alternative | Kyun Nahi? |
|:---|:---|
| TensorFlow/Keras | Hamare pre-trained models `.pt` format mein hain — TF mein load nahi hote |
| ONNX Runtime | Conversion ka extra step lagta |
| JAX | Research tool — production ke liye mature nahi |

**PyTorch Best Tha Kyunke:** Hamare sab downloaded checkpoints PyTorch format mein hain — sirf PyTorch se load hote hain.

---

### 4. SQLite — Database

**Kya hai?**
SQLite ek file-based lightweight database hai jo `data/history.db` mein scan history save karta hai.

**Kyun SQLite?**

| Alternative | Kyun Nahi? |
|:---|:---|
| PostgreSQL | Server setup chahiye, FYP ke liye overkill |
| MySQL | Same — separate server process |
| MongoDB | SQL queries history ke liye zyada powerful hain |

**SQLite Best Tha Kyunke:** Zero setup — sirf 1 `.db` file, koi server nahi, Python mein built-in support.

---

### 5. OpenCV — Image Processing

**Hamare Project Mein Kahan Use Hua?**
- `cv2.resize()` — Face image 224x224 resize karna
- `cv2.applyColorMap()` — Grad-CAM JET colormap apply karna
- `cv2.imwrite()` — Heatmap PNG disk par save karna
- `cv2.cvtColor()` — RGB to BGR color conversion

**OpenCV Best Tha Kyunke:** Image processing ka industry standard — fast, reliable, extensive documentation.

---

### 6. NumPy — Array & Matrix Operations

**Hamare Project Mein Kahan Use Hua?**
- Image pixels ko `np.ndarray` mein convert karna
- Grad-CAM saliency map calculations
- Model prediction probabilities process karna

**NumPy Best Tha Kyunke:** PyTorch tensors NumPy ke sath natively compatible hain — `tensor.numpy()` se direct convert hota hai.

---

### 7. scikit-learn — Performance Metrics

**Hamare Project Mein Kahan Use Hua?** (`core/metrics.py`)
- `accuracy_score()`, `precision_score()`, `recall_score()`, `f1_score()`
- `roc_auc_score()`, `roc_curve()`
- `confusion_matrix()`, `classification_report()`

**scikit-learn Best Tha Kyunke:** 2 lines mein koi bhi metric calculate ho jati hai — industry standard library.

---

### 8. Matplotlib + Seaborn — Charts

**Hamare Project Mein Kahan Use Hua?** (`core/metrics.py`)
- `seaborn.heatmap()` — Confusion matrix blue grid chart
- `matplotlib.pyplot` — ROC Curve graph

---

## 📋 Complete Technology Stack Summary Table

| Layer | Technology | Kaam |
|:---|:---|:---|
| Frontend UI | React 18 | Component-based UI |
| Frontend Logic | JavaScript ES6+ | API calls, file handling |
| Frontend Design | CSS3 | Dark theme, animations |
| Frontend Build | Vite 5 | Fast dev server & bundler |
| Backend Language | Python 3.10+ | Sab kuch |
| Backend Framework | Flask 3 | REST API server |
| AI Framework | PyTorch 2 | Model inference |
| Face Detection | MTCNN (facenet-pytorch) | Face localization |
| Image Processing | OpenCV 4 | Resize, colormap, save |
| Array Operations | NumPy | Matrix calculations |
| ML Metrics | scikit-learn | Accuracy, AUC, F1 |
| Charts | Matplotlib + Seaborn | ROC, Confusion Matrix |
| Database | SQLite3 (built-in) | History storage |
| Config | PyYAML | config.yaml loading |
| Model Hub | HuggingFace Hub | ViT model download |
| Model Architecture | timm | EfficientNet, Xception |
