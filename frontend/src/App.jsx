import { useEffect, useRef, useState } from "react";
import Scanner from "./components/Scanner";
import Metrics from "./pages/Metrics";
import History from "./pages/History";
import "./style.css";

/* ------------------------------------------------------------------ */
/* Landing page: port of templates/index.html + static/js/app.js       */
/* ------------------------------------------------------------------ */

const FAQ = [
  {
    q: "What is a deepfake?",
    a: "A deepfake is media in which a person's face, voice or expressions are swapped or manipulated using deep-learning models such as GANs and diffusion networks. Today's deepfakes are often indistinguishable from real footage visually."
  },
  {
    q: "Why do you analyze only the face?",
    a: "Because deepfake manipulation almost always lives in the face. Cropping to the detected face removes irrelevant background detail and reduces the noise the models have to reason over, which measurably improves accuracy (FR-08 / FR-09 preprocessing)."
  },
  {
    q: "Which files can I scan?",
    a: "JPEG / PNG images up to 10 MB, and MP4 / AVI videos up to 100 MB. Videos are scanned at 5 fps, limited to the first 60 seconds and a maximum of 24 key frames."
  },
  {
    q: "Is my upload kept private?",
    a: "Yes — all processing happens locally (NFR-07). Your file never leaves your computer and temporary files are removed after one hour."
  },
  {
    q: "How accurate is it?",
    a: "The ensemble reaches ≈98% accuracy and 0.999 ROC-AUC on the FaceForensics++ (C23) benchmark — above all SRS targets (75% accuracy, 80% F1, 0.85 AUC)."
  },
  {
    q: "What should I do if I find a deepfake?",
    a: "Don't share it further. Report it (platform moderation, cybercrime reporting) and warn anyone it could affect. Treat unverified content skeptically."
  }
];

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
      <header className={"topbar" + (scrolled ? " scrolled" : "")}>
        <div className="brand">
          <span className="logo">🕵️</span>
          <div>
            <h1>FAKE <span>FILTER</span></h1>
            <p className="tagline">Deepfake scanning, on your device</p>
          </div>
        </div>
        <nav>
          <a href="#analyze" className="nav-link active">Detect</a>
          <a href="#how" className="nav-link">How it works</a>
          <a href="#features" className="nav-link">Features</a>
          <a href="#faq" className="nav-link">FAQs</a>
          <a href="/history" className="nav-link">History</a>
          <a href="#about" className="nav-link">About</a>
        </nav>
        <button className="btn primary nav-cta" onClick={scrollToAnalyze}>Analyze now</button>
      </header>

      <main className="container" ref={reveal}>
        {/* HERO + SCANNER */}
        <section id="analyze" className="section hero">
          <div className="hero-grid">
            <div className="hero-left">
              <span className="eyebrow"><i className="dot"></i> AI-powered deepfake detection</span>
              <h1 className="hero-title">
                Is this media <span className="grad-accent">real</span> or{" "}
                <span className="grad-danger">fake?</span>
              </h1>
              <p className="hero-sub">
                Generative AI can make fakes that fool the eye — but not our
                <b> four-model ensemble</b>. Upload an image or short video and get a
                confident verdict plus visual proof in seconds.
              </p>
              <div className="hero-cta">
                <button className="btn primary btn-lg" onClick={scrollToAnalyze}>Scan a file free</button>
                <span className="hero-trust">🔒 100% local — nothing is uploaded anywhere</span>
              </div>
              <div className="hero-stats">
                <div className="hs"><CountUp to={98} /><span>% test accuracy</span></div>
                <div className="hs"><CountUp to={99} /><span>ROC-AUC</span></div>
                <div className="hs"><CountUp to={40} /><span>frames per video</span></div>
                <div className="hs"><CountUp to={100} /><span>% local &amp; private</span></div>
              </div>
            </div>

            <Scanner />
          </div>

          <div className="engines">
            <div className="card engine reveal">
              <span className="engine-ico">🖼️</span>
              <div>
                <h4>Image scan</h4>
                <p>Only the <b>detected face</b> is analyzed — background noise removed for cleaner, more accurate verdicts.</p>
              </div>
            </div>
            <div className="card engine reveal">
              <span className="engine-ico">🎞️</span>
              <div>
                <h4>Video scan</h4>
                <p>Faces are extracted frame by frame, then temporal consistency is checked across up to 24 key frames.</p>
              </div>
            </div>
            <div className="card engine reveal">
              <span className="engine-ico">🧠</span>
              <div>
                <h4>Ensemble verifier</h4>
                <p>XceptionNet + EfficientNet-B3 + ViT + ViT-L/14 weighted soft-voting turns every signal into one defensible verdict.</p>
              </div>
            </div>
          </div>
        </section>

        {/* HOW IT WORKS */}
        <section id="how" className="section">
          <div className="sec-head reveal">
            <span className="eyebrow plain">One flow, three steps</span>
            <h2 className="section-title">How it works</h2>
            <p className="section-sub">From file to verdict and visual proof — in seconds</p>
          </div>
          <div className="how-grid">
            <div className="card step reveal">
              <span className="step-num">01</span>
              <div className="step-ico">📤</div>
              <h4>Upload media</h4>
              <p>Drop an image or short video. It's validated instantly and stored temporarily on your machine.</p>
            </div>
            <div className="card step reveal">
              <span className="step-num">02</span>
              <div className="step-ico">🎯</div>
              <h4>Extract the face</h4>
              <p>MTCNN locates and aligns the face — the only region analyzed, so background noise can't hurt accuracy.</p>
            </div>
            <div className="card step reveal">
              <span className="step-num">03</span>
              <div className="step-ico">⚖️</div>
              <h4>Ensemble verdict</h4>
              <p>Four models vote, Grad-CAM highlights the proof, and you get one REAL / FAKE answer with confidence.</p>
            </div>
          </div>
        </section>

        {/* FEATURES */}
        <section id="features" className="section">
          <div className="sec-head reveal">
            <span className="eyebrow plain">Why you'll love it</span>
            <h2 className="section-title">Built for clarity and trust</h2>
            <p className="section-sub">Simple results, rigorous science, zero cloud</p>
          </div>
          <div className="feature-grid">
            <div className="card feature reveal"><span className="f-ico">🎯</span><h4>One clear verdict</h4><p>REAL or FAKE with confidence — no confusing model percentages.</p></div>
            <div className="card feature reveal"><span className="f-ico">🧬</span><h4>Four-model ensemble</h4><p>XceptionNet + EfficientNet-B3 + ViT + ViT-L/14 catch spatial, global and temporal cues.</p></div>
            <div className="card feature reveal"><span className="f-ico">🔍</span><h4>Evidence it explains itself</h4><p>Grad-CAM heatmaps show exactly which regions look manipulated.</p></div>
            <div className="card feature reveal"><span className="f-ico">🔒</span><h4>100% private</h4><p>Runs on your device. Files auto-delete after one hour.</p></div>
            <div className="card feature reveal"><span className="f-ico">🎬</span><h4>Images &amp; video</h4><p>Face-first analysis on photos and clips, with temporal checks.</p></div>
            <div className="card feature reveal"><span className="f-ico">📱</span><h4>Effortless to use</h4><p>A focused, friendly app that works in any modern browser.</p></div>
          </div>
        </section>

        {/* FAQ */}
        <section id="faq" className="section">
          <div className="sec-head reveal">
            <span className="eyebrow plain">Know your enemy</span>
            <h2 className="section-title">FAQs</h2>
            <p className="section-sub">Everything about deepfakes &amp; how this tool works</p>
          </div>
          <div className="faq-list reveal">
            {FAQ.map((f, i) => (
              <details className="faq" key={f.q} open={i === 0}>
                <summary>{f.q}</summary>
                <p>{f.a}</p>
              </details>
            ))}
          </div>
        </section>

        {/* ABOUT */}
        <section id="about" className="section">
          <div className="sec-head reveal">
            <span className="eyebrow plain">The team behind it</span>
            <h2 className="section-title">About the project</h2>
            <p className="section-sub">Final-year project at the Institute for Art and Culture, Lahore</p>
          </div>
          <div className="panel about-panel reveal">
            <p className="about-text">
              The <b>Deepfake Detection System</b> is a final-year project that ships a complete
              web application — file upload &amp; validation, face detection and alignment
              (MTCNN), a multi-model inference core (XceptionNet + EfficientNet-B3 +
              ViT + ViT-L/14 ensemble)
              and an explainable Grad-CAM verdict. All of it runs entirely on localhost in a clean
              three-tier architecture (Flask API · React UI · preprocessing pipeline · inference engine).
            </p>
          </div>
          <h3 className="sub-title reveal">Team</h3>
          <div className="team-grid">
            <div className="card team-card reveal">
              <div className="avatar av-1">TH</div>
              <h4>Tehzeeb Ul Hassan</h4>
              <p className="roll">IAC-FA22-BCS-009</p>
              <p className="role">Developer &amp; Documentation</p>
            </div>
            <div className="card team-card reveal">
              <div className="avatar av-2">MK</div>
              <h4>Muhammad Khubaib Khalid</h4>
              <p className="roll">IAC-FA22-BCS-007</p>
              <p className="role">Developer &amp; Documentation</p>
            </div>
          </div>
          <h3 className="sub-title reveal">Supervision</h3>
          <div className="team-grid">
            <div className="card team-card supervisor reveal">
              <div className="avatar av-3">AM</div>
              <h4>Mr. Adeel Mustafa</h4>
              <p className="roll">Project Supervisor · Dept. of Computer Science</p>
              <p className="role">Institute for Art and Culture (IAC), Lahore</p>
            </div>
          </div>
          <h3 className="sub-title reveal">Contact</h3>
          <div className="panel contact-panel reveal">
            <p>📧 <a href="mailto:tehzeebh360@gmail.com">tehzeebh360@gmail.com</a> — Tehzeeb Ul Hassan</p>
            <p>📧 <a href="mailto:khubaibkh67@gmail.com">mkhubaibkhalid9@gmail.com</a> — Muhammad Khubaib Khalid</p>
            <p>🏫 Institute for Art and Culture (IAC) · 7.5 km Main Raiwind Road, Lahore, Pakistan · Session 2022–2026</p>
          </div>
        </section>
      </main>

<footer className="footer">
        <small>FYP — Tehzeeb Ul Hassan (IAC-FA22-BCS-009) &amp; Muhammad Khubaib Khalid (IAC-FA22-BCS-007)</small>
        <span className="footer-sep">·</span>
        <a href="/history" className="dev-link">History →</a>
        <span className="footer-sep">·</span>
        <a href="/metrics" className="dev-link">Developer Metrics Dashboard →</a>
      </footer>
    </div>
  );
}

/* ------------------------------------------------------------------ */

export default function App() {
  const [path, setPath] = useState(window.location.pathname);
  useEffect(() => {
    const onPop = () => setPath(window.location.pathname);
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);
  return path === "/metrics" ? <Metrics /> : path === "/history" ? <History /> : <Landing />;
}