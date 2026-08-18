# 🎓 FYP Defence — Questions & Answers

**Project**: Deepfake Detection System
**Students**: Tehzeeb Ul Hassan & Muhammad Khubaib Khalid
**Supervisor**: Mr. Adeel Mustafa
**Institute**: Institute for Art and Culture (IAC), Lahore

---

## 🤖 AI / Models Related Questions

---

**Q1. Aap ne yeh 4 models hi kyun choose kiye?**

Hamara maqsad accuracy aur generalization dono achieve karna tha. Isliye 2 alag types ke models choose kiye:

- **XceptionNet & EfficientNet-B3** — CNN (Convolutional Neural Network) based models hain. Yeh pixel-level texture artifacts aur frequency patterns dhoondnte hain. FF++ C23 dataset par XceptionNet ka AUC 0.9976 hai.
- **ViT-Small (CommunityForensics)** — Vision Transformer hai jo 2.7 Million samples aur 4803 different deepfake generators par train hua — zyada generalize karta hai.
- **ViT-L/14 (LNCLIP)** — CLIP-based large model hai jo unseen deepfake generators par bhi kaam karta hai (DFDC: 87.2%, Celeb-DF: 96.6%).

Charon models ka ensemble isliye kiya taake koi ek model ki weakness doosra cover kar sake.

---

**Q2. XceptionNet aur EfficientNet mein kya farq hai?**

| | XceptionNet | EfficientNet-B3 |
|:---|:---|:---|
| Architecture | Depthwise Separable Convolutions | Compound Scaling (depth + width + resolution) |
| Special Role | Grad-CAM heatmap generate karta hai (conv4 layer) | Sabse high single-model accuracy (98.29%) |
| Parameters | ~22 Million | ~12 Million |
| Training Data | FF++ C23 | FF++ C23 |

---

**Q3. ViT aur CNN mein kya farq hai?**

- **CNN**: Image ko chote patches mein todta hai aur local patterns (edges, textures) dhoondta hai. Ek jagah ek waqt mein dekhta hai.
- **ViT (Vision Transformer)**: Poori image ko patches mein todta hai aur **Self-Attention** mechanism se patches ke darmiyan global relationships dhoondta hai. Ek waqt mein poori image ka context dekhta hai.

CNN local features mein better hai, ViT global inconsistencies pakad leta hai — isliye dono sath use kiye.

---

**Q4. Model accuracy kitni hai?**

| Model | Accuracy | AUC |
|:---|:---|:---|
| XceptionNet | ~97% | 0.9976 |
| EfficientNet-B3 | 98.29% | 0.9976 |
| ViT-Small | ~95%+ | High |
| ViT-L/14 | DFDC 87.2% / Celeb-DF 96.6% | — |
| **Ensemble** | **~98%+** | **~0.99** |

---

**Q5. Agar model galat result de toh?**

Yeh possible hai — koi bhi AI model 100% perfect nahi hota. Is liye hamne:
1. **4 models ka ensemble** rakha — ek ki galti baki cover karte hain
2. **Confidence % dikhate hain** — agar 51% FAKE hai toh user samjhe ke borderline case hai
3. **Grad-CAM heatmap** provide kiya — user khud dekh sakta hai model ne kya dekha
4. **Threshold 0.5** — user `config.yaml` mein customize kar sakta hai

---

**Q6. Training kahan ki thi? Aap ne khud train kiya?**

Nahi — hamne pre-trained models use kiye jo already publicly available hain:
- **XceptionNet & EfficientNet**: Hugging Face `Khubaib7/deepfake-models` par available hain — FF++ C23 dataset par train kiye gaye
- **ViT-Small (CommunityForensics)**: `buildborderless/CommunityForensics-DeepfakeDet-ViT` — CVPR 2025 research paper
- **ViT-L/14 (LNCLIP)**: `yermandy/deepfake-detection` — WACV 2026 paper

Hamara contribution tha in models ko ek unified detection pipeline mein integrate karna aur ensemble system banana.

---

**Q7. FF++ dataset kya hai?**

FF++ (FaceForensics++) ek standard deepfake detection benchmark dataset hai:
- **100 original videos** se 1000 manipulated videos banaye gaye
- **4 manipulation methods**: DeepFakes, Face2Face, FaceSwap, NeuralTextures
- **C23**: Light compression version — detection ke liye zyada challenging
- Research mein deepfake detection models is par train aur test kiye jate hain

---

**Q8. Ensemble kyun use kiya, single model kyun nahi?**

Single model ki limitations hoti hain:
- Ek model sirf specific type ke deepfakes pakad sakta hai
- New unseen generators par fail ho sakta hai

Ensemble ke faide:
- **Ek model ki weakness = doosra cover karta hai**
- **Weighted voting** — zyada reliable model ko zyada weight
- **Overall accuracy badh jati hai**
- Real world mein zyada robust

---

## 🖼️ Grad-CAM Related Questions

---

**Q9. Grad-CAM kya hota hai?**

Grad-CAM (Gradient-weighted Class Activation Mapping) ek explainability technique hai jo batati hai ke AI model ne image ke **kaunse hisse** dekh kar FAKE/REAL decision kiya.

Process:
1. XceptionNet ke **conv4 layer** se gradients calculate hote hain
2. Yeh gradients ek **2D saliency map** banate hain (224x224)
3. Is map par **JET colormap** lagaya jata hai
4. Face image par blend ho kar **Red/Yellow/Blue heatmap** banti hai

Red = Model ne is jagah zyada suspicion detect kiya
Blue = Is jagah kam concern tha

---

**Q10. Red color kyun dikhta hai heatmap mein?**

Red color = **Highest activation area** — jahan model ne sabse zyada "fake evidence" dhundha.

JET colormap mein:
- 🔴 **Red** = Highest value (1.0) — most suspicious area
- 🟡 **Yellow** = Medium-high value
- 🟢 **Green** = Medium value
- 🔵 **Blue** = Lowest value (0.0) — least suspicious area

Typically deepfake images mein **cheek boundaries, ear area, aur eye corners** red hote hain.

---

**Q11. Yeh XceptionNet par hi kyun apply kiya?**

- XceptionNet mein `backbone.conv4` layer convolution-based hai — Grad-CAM CNN layers par best kaam karta hai
- ViT (Transformer) par Grad-CAM directly apply nahi hota — different explainability techniques chahiye hoti hain (Attention Rollout)
- XceptionNet ka `conv4` layer spatial features retain karta hai jo visual heatmap ke liye zaruri hain

---

## 💻 Technical Questions

---

**Q12. Flask kyun use kiya Django kyun nahi?**

Django bahut heavy framework hai — ORM, admin panel, templates sab include hote hain. Hamein sirf ek **lightweight REST API** chahiye thi. Flask mein `@app.route` decorator se seedha API bana li. Django use karna overkill hota.

---

**Q13. React kyun use kiya?**

- **Component-based architecture** — Scanner, History, Metrics alag reusable pieces hain
- **State management** (`useState`, `useEffect`) — real-time scan progress, tabs, confidence animation ke liye zaruri tha
- Plain HTML/JS mein itna complex dynamic UI banana mushkil hota
- Duniya ki sabse popular frontend library — documentation easily milti hai

---

**Q14. SQLite kyun use kiya?**

- **Zero setup** — koi server nahi, sirf ek `data/history.db` file
- Python mein **built-in support** — koi extra installation nahi
- FYP scale ke liye sufficient — thousands of records handle kar sakta hai
- **WAL mode** use kiya — simultaneous read/write safe hoti hai

---

**Q15. Magic bytes kya hote hain?**

Har file type ke shuruati bytes ek fixed pattern hote hain jo file ki actual type identify karte hain:

- JPEG: `FF D8 FF`
- PNG: `89 50 4E 47`
- MP4: `66 74 79 70` (`ftyp`)
- AVI: `52 49 46 46` (`RIFF`)

Agar koi `.jpg` extension wali file andar se actually `.exe` (virus) ho, toh magic bytes check se pakad li jati hai. Extension badal kar attack nahi ho sakta.

---

**Q16. Session ID kyun use kiya?**

Har scan request ko ek **unique 32-character ID** milti hai (UUID). Is se:
- Har user ki files alag folder mein save hoti hain (`uploads/<session_id>/`)
- Multiple simultaneous scans mein files mix nahi hoti
- History record mein specific scan track karna easy hota hai

---

**Q17. WAL mode kya hota hai?**

WAL (Write-Ahead Logging) SQLite ka high-performance mode hai:
- **Without WAL**: Database write hone ke dauran read blocked hoti hai
- **With WAL**: Read aur Write ek sath ho sakti hain — no blocking

Hamare case mein: ek user scan save kar raha hai, doosra history page dekh raha hai — WAL mode se dono simultaneously kaam karte hain bina crash ke.

---

## 📊 Performance Metrics Questions

---

**Q18. Accuracy, Precision, Recall, F1 mein kya farq hai?**

| Metric | Formula | Matlab |
|:---|:---|:---|
| **Accuracy** | (TP+TN) / Total | Overall kitne correct? |
| **Precision** | TP / (TP+FP) | Jitne FAKE bole unme se kitne sach mein FAKE? |
| **Recall** | TP / (TP+FN) | Saare real FAKE mein se kitne pakde? |
| **F1** | 2×(P×R)/(P+R) | Precision aur Recall ka balanced average |

TP = True Positive (FAKE sahi pakda)
TN = True Negative (REAL sahi pakda)
FP = False Positive (REAL ko FAKE bola — galti)
FN = False Negative (FAKE ko REAL bola — dangerous galti)

---

**Q19. ROC-AUC kya hota hai?**

ROC (Receiver Operating Characteristic) curve model ki overall discrimination power dikhata hai:
- X-axis: False Positive Rate (Real ko FAKE bolne ki rate)
- Y-axis: True Positive Rate (FAKE pakadne ki rate)
- **AUC = Area Under Curve** — 0.5 se 1.0 tak
  - AUC 0.5 = Random guess (bekar)
  - AUC 1.0 = Perfect model
  - Hamare models: ~0.9976 (bahut acha!)

---

**Q20. Confusion Matrix explain karein?**

```
                 Predicted
                REAL    FAKE
Actual  REAL  [ TN  |  FP  ]
        FAKE  [ FN  |  TP  ]
```

- **TN (True Negative)**: Real image ko Real bola ✅
- **TP (True Positive)**: Fake image ko Fake bola ✅
- **FP (False Positive)**: Real image ko Fake bola ❌ (false alarm)
- **FN (False Negative)**: Fake image ko Real bola ❌ (miss — dangerous!)

Hamare model mein FN (miss karna) minimize karna sabse important tha.

---

## 🔒 Security Questions

---

**Q21. Koi security measure hai?**

Haan, 3 layers:
1. **Magic Bytes Validation** — File ka actual type check (extension spoof attack prevent)
2. **Size Limits** — Image max 10MB, Video max 100MB (DoS attack prevent)
3. **Path Traversal Protection** — `media/<sid>/<file>` endpoint mein `..` navigation block hai

---

**Q22. User data kahan store hota hai?**

- **Files**: `uploads/<session_id>/` folder mein locally disk par
- **History**: `data/history.db` SQLite file mein locally
- **Koi data cloud par nahi jata** — 100% local system par hai
- Website par likha hai: "100% local — nothing is uploaded anywhere"

---

**Q23. Internet connection chahiye?**

**Nahi** — ek baar models download ho jayen toh:
- Sab kuch local machine par chalta hai
- No cloud API calls
- No internet required for detection
- Privacy protected — koi data bahar nahi jata

---

## 🎯 Project Related Questions

---

**Q24. Yeh project banane ki zarurat kyun pari?**

Generative AI (Stable Diffusion, MidJourney, DeepFaceLab) itni advanced ho gayi hai ke fake videos/images se:
- **Political manipulation** ho sakti hai (leaders ke fake speeches)
- **Financial fraud** ho sakta hai (identity theft)
- **Misinformation** phail sakti hai social media par

Ek aam user ke paas koi tool nahi tha verify karne ka. Hamare project ne **accessible, free, local deepfake detector** banaya.

---

**Q25. Real world mein kahan use ho sakta hai?**

- **Journalism**: News images/videos verify karna
- **Social Media Platforms**: Fake content automatically flag karna
- **Banks/Finance**: Video KYC verification
- **Legal**: Court evidence authenticity check
- **Government**: Election campaigns mein fake videos detect karna

---

**Q26. Future improvements kya ho sakte hain?**

1. **Audio deepfake detection** add karna (voice cloning)
2. **Real-time video stream** detection (live webcam)
3. **Batch processing** — multiple files ek sath scan karna
4. **Mobile app** banana (Android/iOS)
5. **More models** integrate karna (newer architectures)
6. **Explainability improve** karna (ViT attention maps)

---

**Q27. Sabse bada challenge kya tha?**

1. **4 alag AI models ko ek pipeline mein integrate karna** — har model ki alag preprocessing requirements thi
2. **LNCLIP model TorchScript format** mein tha — loading alag tarike se hoti thi
3. **Grad-CAM aur ViT compatibility** — Grad-CAM sirf CNN models par direct apply hota hai
4. **Server warmup time** — 4 models load hone mein 2-3 minutes lagte hain, is ke liye async polling system banana pada

---

## ⚡ Quick Reference Answers

| Question | Short Answer |
|:---|:---|
| Kitne models? | 4 (XceptionNet, EfficientNet-B3, ViT-Small, ViT-L/14) |
| Dataset? | FF++ C23, CommunityForensics (2.7M images) |
| Accuracy? | ~98% (Ensemble) |
| Backend? | Python + Flask |
| Frontend? | React + JavaScript + CSS |
| Database? | SQLite (`data/history.db`) |
| Local ya Cloud? | 100% Local |
| Grad-CAM? | XceptionNet conv4 layer par visual proof |
| Threshold? | 0.5 (configurable in config.yaml) |
| Face Detection? | MTCNN (95% confidence required) |
