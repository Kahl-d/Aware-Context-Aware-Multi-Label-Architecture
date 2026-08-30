import { useState } from 'react';
import { PageContainer } from '../components/layout/PageContainer';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from 'recharts';

// ─── Data ──────────────────────────────────────────────────────

const PER_THEME = [
  { theme: 'Navigational', abbr: 'NAV', f1: 0.707, p: 0.669, r: 0.749, support: 430 },
  { theme: 'Familial_Capital', abbr: 'FAM', f1: 0.600, p: 0.533, r: 0.686, support: 70 },
  { theme: 'Aspirational', abbr: 'ASP', f1: 0.585, p: 0.549, r: 0.627, support: 276 },
  { theme: 'Social', abbr: 'SOC', f1: 0.506, p: 0.538, r: 0.478, support: 90 },
  { theme: 'Perseverance', abbr: 'PER', f1: 0.480, p: 0.570, r: 0.415, support: 176 },
  { theme: 'Resistance', abbr: 'RES', f1: 0.395, p: 0.303, r: 0.570, support: 86 },
  { theme: 'Spiritual', abbr: 'SPI', f1: 0.375, p: 0.291, r: 0.526, support: 95 },
  { theme: 'Attainment', abbr: 'ATT', f1: 0.303, p: 0.224, r: 0.469, support: 32 },
];

const MODELS_CMP = [
  { model: 'Random', f1: 0.084, prauc: 0.085 },
  { model: 'Oracle', f1: 0.151, prauc: 0.085 },
  { model: 'TF-IDF+LR', f1: 0.378, prauc: 0.350 },
  { model: 'AWARE Base', f1: 0.474, prauc: 0.473 },
  { model: 'AWARE v3', f1: 0.461, prauc: 0.462 },
  { model: 'AWARE v4', f1: 0.494, prauc: 0.484 },
];

const SINGLE_MULTI = [
  { theme: 'NAV', single: 0.896, overall: 0.707 },
  { theme: 'ASP', single: 0.866, overall: 0.585 },
  { theme: 'ATT', single: 0.727, overall: 0.303 },
  { theme: 'FAM', single: 0.697, overall: 0.600 },
  { theme: 'PER', single: 0.673, overall: 0.480 },
  { theme: 'SOC', single: 0.626, overall: 0.506 },
  { theme: 'RES', single: 0.585, overall: 0.395 },
  { theme: 'SPI', single: 0.568, overall: 0.375 },
];

const FAILURES = [
  { id: 1, title: 'Encoder memorized too fast', v3: 'LR=1.5e-5', v4: 'LR=5.0e-6', impact: 'Slowed memorization 3×' },
  { id: 2, title: 'SWA never activated', v3: 'Start at 50%', v4: 'Start at 25%', impact: '0→22 checkpoints' },
  { id: 3, title: 'LLRD froze bottom layers', v3: 'decay=0.85', v4: 'decay=0.92', impact: 'Bottom LR: 3e-7→8.5e-7' },
  { id: 4, title: 'R-Drop dominated loss', v3: 'α=2.0 (55-72%)', v4: 'α=1.0 (~11%)', impact: 'Task loss restored' },
  { id: 5, title: 'Phase 1 too short', v3: '4ep (AUC=0.14)', v4: '8ep (AUC=0.40)', impact: 'Stable head init' },
  { id: 6, title: 'Early stopping too eager', v3: 'patience=5', v4: 'patience=8', impact: 'Late-peaking themes' },
  { id: 7, title: 'Phase 3 ineffective', v3: '20.5K params', v4: 'Same', impact: 'Acknowledged limitation' },
];

// ─── Reusable pieces ───────────────────────────────────────────

function SectionHeading({ id, title }: { id: string; title: string }) {
  return (
    <h2
      id={id}
      className="text-[17px] font-semibold text-gray-900 pt-10 pb-3 border-b border-gray-200 sticky top-14 bg-white z-10 m-0"
    >
      {title}
    </h2>
  );
}

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="text-center">
      <p className="text-2xl font-semibold font-mono text-gray-900 m-0">{value}</p>
      <p className="text-[12px] text-gray-500 mt-0.5 m-0">{label}</p>
      {sub && <p className="text-[11px] text-gray-400 m-0">{sub}</p>}
    </div>
  );
}

function Callout({ color, title, children }: { color: 'blue' | 'amber' | 'green'; title: string; children: React.ReactNode }) {
  const cls = {
    blue: 'bg-blue-50 border-blue-100 text-blue-800',
    amber: 'bg-amber-50 border-amber-100 text-amber-800',
    green: 'bg-green-50 border-green-100 text-green-800',
  }[color];
  return (
    <div className={`border rounded-lg p-4 ${cls}`}>
      <p className="text-[13px] font-semibold m-0">{title}</p>
      <div className="text-[13px] mt-1 leading-relaxed opacity-90">{children}</div>
    </div>
  );
}

// ─── Main Paper page ───────────────────────────────────────────

export function PaperPage() {
  const [resultView, setResultView] = useState<'comparison' | 'pertheme' | 'bottleneck'>('comparison');
  const [expandedFM, setExpandedFM] = useState<number | null>(null);

  return (
    <PageContainer className="max-w-3xl">
      {/* ── Hero ── */}
      <div className="text-center pt-8 pb-6">
        <p className="text-[11px] uppercase tracking-[0.2em] text-gray-400 mb-2 m-0">M.S. Thesis — San Francisco State University — Spring 2026</p>
        <h1 className="text-[22px] font-bold text-gray-900 m-0 tracking-tight leading-tight">
          AWARE: Adaptive Weighted Architecture for Reflective Essays
        </h1>
        <p className="text-[13px] text-gray-500 mt-2 max-w-xl mx-auto leading-relaxed">
          Automated Detection of Community Cultural Wealth Themes in Student Reflective Essays Using Domain-Adapted DeBERTa
        </p>
        <p className="text-[13px] text-gray-500 mt-2">Khalid Khan &middot; Department of Computer Science &middot; M.S. Artificial Intelligence</p>
      </div>

      {/* ── Key numbers ── */}
      <div className="grid grid-cols-4 gap-4 py-6 border-y border-gray-100">
        <Stat label="Sentences" value="17,622" sub="from 2,636 essays" />
        <Stat label="CCW Themes" value="8" sub="multi-label" />
        <Stat label="Macro-F1" value="0.494" sub="[0.464, 0.520]" />
        <Stat label="vs Baseline" value="+30.8%" sub="over TF-IDF" />
      </div>

      {/* ━━━━ ABSTRACT ━━━━ */}
      <SectionHeading id="abstract" title="Abstract" />
      <div className="py-4 space-y-4">
        <p className="text-[14px] text-gray-700 leading-relaxed">
          Identifying Community Cultural Wealth (CCW) themes in student reflective essays is a sentence-level, multi-label classification problem. Human annotation does not scale: six trained coders produced 2,636 annotated essays over multiple years of the ALMA Project at San Francisco State University. Automating this annotation requires a model that can detect eight overlapping themes in short, informal sentences where the dominant signal in the embedding space is not thematic but structural — nearest-neighbor analysis misclassifies 38.7% of sentences, and five of eight themes have separability indices at or below 1.0.
        </p>
        <p className="text-[14px] text-gray-700 leading-relaxed">
          This thesis presents <strong>AWARE</strong> (Adaptive Weighted Architecture for Reflective Essays), a framework that addresses the specific failure modes of this task. AWARE combines domain-adaptive pre-training on ALMA essays, essay-level context encoding that supplies each sentence with its surrounding narrative, asymmetric loss to handle severe class imbalance (up to 21.8:1), and per-theme threshold optimization to recover rare-class recall. The backbone is DeBERTa-v3, evaluated at base (86M parameters) and large (438M parameters) scales. The data pipeline contributes an embedding-based semantic cleaning method that removed 1,705 ambiguous sentences (8.6%) from the 17,622-sentence dataset without manual inspection.
        </p>
        <p className="text-[14px] text-gray-700 leading-relaxed">
          The best model achieves <strong>Macro-F1 = 0.494</strong> [0.464, 0.520], a 30.8% improvement over the TF-IDF baseline. Per-theme F1 ranges from 0.303 (Attainment) to 0.707 (Navigational). However, a group-level evaluation reveals that the model is substantially more capable than aggregate metrics suggest: on single-theme sentences, F1 ranges from 0.568 to 0.896 with perfect precision across all eight themes. The primary bottleneck is not theme detection but multi-label disentanglement — correctly identifying all co-occurring themes within the same sentence.
        </p>
        <p className="text-[14px] text-gray-700 leading-relaxed">
          The thesis also presents the <strong>ALMA Research Dashboard</strong>, a web application that unifies the data pipeline, model deployment, and annotation workflow, enabling researchers to apply the trained model to the 1,443 unannotated essays in the corpus.
        </p>
        <Callout color="blue" title="Key Finding">
          <p className="m-0">On single-theme sentences (one theme present, no co-occurrence ambiguity), AWARE achieves <strong>F1 = 0.568–0.896 with perfect precision</strong> across all 8 themes. The 0.494 macro-F1 reflects multi-label disentanglement difficulty, not detection failure.</p>
        </Callout>
      </div>

      {/* ━━━━ THE PROBLEM ━━━━ */}
      <SectionHeading id="problem" title="The Problem" />
      <div className="py-4 space-y-4">
        <p className="text-[14px] text-gray-700 leading-relaxed">
          CCW classification is structurally hard. Embedding analysis reveals that 8 CCW themes occupy deeply overlapping semantic space — centroid cosine similarities exceed 0.93 across most theme pairs. Natural clustering finds only 2 groups (themed vs. unthemed), not 8 distinct themes. A nearest-neighbor baseline misclassifies <strong>38.7%</strong> of sentences — establishing a lower bound on error independent of model architecture.
        </p>
        <div className="grid grid-cols-3 gap-3">
          <div className="border border-gray-200 rounded-lg p-3 text-center">
            <p className="text-lg font-semibold font-mono text-gray-900 m-0">21.8:1</p>
            <p className="text-[11px] text-gray-500 mt-0.5 m-0">Max class imbalance</p>
          </div>
          <div className="border border-gray-200 rounded-lg p-3 text-center">
            <p className="text-lg font-semibold font-mono text-gray-900 m-0">77.3%</p>
            <p className="text-[11px] text-gray-500 mt-0.5 m-0">Attainment multi-label rate</p>
          </div>
          <div className="border border-gray-200 rounded-lg p-3 text-center">
            <p className="text-lg font-semibold font-mono text-gray-900 m-0">38.7%</p>
            <p className="text-[11px] text-gray-500 mt-0.5 m-0">KNN error (lower bound)</p>
          </div>
        </div>
        <p className="text-[14px] text-gray-700 leading-relaxed">
          Five of eight themes have separability indices (d-prime) at or below 1.0. Context determines meaning: "They helped me figure it out" is Familial Capital in a family narrative but Social Capital in a peer study group essay. The rarest theme (Attainment, 2.3% prevalence) has only 92 single-label examples across the entire corpus.
        </p>
      </div>

      {/* ━━━━ DATA PIPELINE ━━━━ */}
      <SectionHeading id="data" title="Data Processing Pipeline" />
      <div className="py-4 space-y-4">
        <p className="text-[14px] text-gray-700 leading-relaxed">
          The raw ALMA corpus consists of 13,563 individual Excel workbooks from 6+ annotators across 8 academic periods, 21 courses, and 5 reflective prompts. A four-stage processing pipeline transforms this into a clean multi-label training dataset.
        </p>
        <div className="flex items-center gap-2 overflow-x-auto py-3">
          {[
            { label: 'Raw Data', count: '13,563 files', desc: 'Excel + CSV across semesters' },
            { label: 'Assembly', count: '2,710 essays', desc: 'Normalized, deduplicated' },
            { label: 'Segmentation', count: '19,724 sents', desc: 'Label propagation' },
            { label: 'Cleaning', count: '−1,705', desc: 'Embedding-based removal' },
            { label: 'Final V4', count: '17,622', desc: '8 themes, 50/50 balance' },
          ].map((step, i) => (
            <div key={i} className="flex items-center gap-2 shrink-0">
              {i > 0 && <span className="text-gray-300 text-lg">→</span>}
              <div className="border border-gray-200 rounded-lg px-3 py-2.5 bg-white min-w-[130px]">
                <p className="text-[12px] font-semibold text-gray-800 m-0">{step.label}</p>
                <p className="text-[14px] font-mono font-semibold text-gray-900 mt-0.5 m-0">{step.count}</p>
                <p className="text-[10px] text-gray-500 mt-0.5 m-0">{step.desc}</p>
              </div>
            </div>
          ))}
        </div>
        <p className="text-[14px] text-gray-700 leading-relaxed">
          The novel contribution is <strong>embedding-based semantic cleaning</strong>: using all-MiniLM-L6-v2 embeddings to identify and remove 1,705 ambiguous boundary sentences (8.6%) without manual inspection. Theme consolidation merged Familial + Filial Piety (cosine similarity 0.987) and dropped First Generation (33 examples, 88% LOO-KNN error) and Community Consciousness (119 examples, 84% multi-label rate).
        </p>
      </div>

      {/* ━━━━ METHODOLOGY ━━━━ */}
      <SectionHeading id="methodology" title="Methodology" />
      <div className="py-4 space-y-6">
        <h3 className="text-[14px] font-semibold text-gray-800 m-0">AWARE Architecture</h3>
        <p className="text-[14px] text-gray-700 leading-relaxed">
          Each essay is processed as a single input sequence. The DeBERTa-v3 encoder (DAPT-adapted via masked language modeling on the essay corpus, reducing MLM loss from 11.98 → 2.71) produces token embeddings. Sentence mean pooling extracts per-sentence representations using tracked token boundaries. Learned position embeddings capture narrative structure. A 2-layer BiLSTM models inter-sentence context with residual connections. The classification head uses multi-sample dropout (K=3) for regularization.
        </p>
        <div className="flex items-center gap-2 overflow-x-auto py-2">
          {[
            { label: 'Essay Text', desc: 'Joined with ". "' },
            { label: 'DeBERTa-v3', desc: 'DAPT encoder (438M)' },
            { label: 'Mean Pool', desc: 'Token → sentence' },
            { label: 'Position', desc: 'Narrative order' },
            { label: 'BiLSTM', desc: '2-layer context' },
            { label: 'Head', desc: '8 theme logits' },
          ].map((step, i) => (
            <div key={i} className="flex items-center gap-2 shrink-0">
              {i > 0 && <span className="text-gray-300">→</span>}
              <div className="border border-gray-200 rounded-lg px-3 py-2 bg-white min-w-[100px]">
                <p className="text-[11px] font-semibold text-gray-800 m-0">{step.label}</p>
                <p className="text-[10px] text-gray-500 mt-0.5 m-0">{step.desc}</p>
              </div>
            </div>
          ))}
        </div>

        <h3 className="text-[14px] font-semibold text-gray-800 m-0">3-Phase Progressive Training</h3>
        <p className="text-[14px] text-gray-700 leading-relaxed">
          With a parameter-to-example ratio of ~31,000:1 (438M parameters / 14,023 training sentences), aggressive regularization is essential. Training proceeds in three phases:
        </p>
        <div className="grid grid-cols-3 gap-3">
          {[
            { phase: 'Phase 1', ep: '8 epochs', desc: 'Frozen encoder. Train BiLSTM + classification head. Establishes stable, non-random head before encoder unfreezing.', metric: 'PR-AUC: 0.398' },
            { phase: 'Phase 2', ep: '40 epochs', desc: 'Progressive unfreeze (top-12 first, then all). LLRD (0.92), SWA (22 checkpoints), R-Drop (α=1.0), AEDA augmentation.', metric: 'PR-AUC: 0.521' },
            { phase: 'Phase 3', ep: '5 epochs', desc: 'Head retrain on frozen encoder. Minimal benefit — Phase 2 SWA ensemble is the final model.', metric: 'PR-AUC: 0.522' },
          ].map((p) => (
            <div key={p.phase} className="border border-gray-200 rounded-lg p-3 bg-white">
              <div className="flex items-center justify-between">
                <p className="text-[12px] font-semibold text-gray-800 m-0">{p.phase}</p>
                <span className="text-[10px] font-mono text-gray-400">{p.ep}</span>
              </div>
              <p className="text-[11px] text-gray-600 mt-1.5 m-0 leading-relaxed">{p.desc}</p>
              <p className="text-[11px] font-mono text-gray-500 mt-1.5 m-0">{p.metric}</p>
            </div>
          ))}
        </div>

        <h3 className="text-[14px] font-semibold text-gray-800 m-0">Loss Function (at convergence)</h3>
        <div className="grid grid-cols-3 gap-3">
          {[
            { name: 'Asymmetric Loss', pct: '77%', desc: 'γ+=0 (preserve all positive gradient), γ-=2.5 (suppress confident negatives)' },
            { name: 'R-Drop KL', pct: '11%', desc: 'Consistency regularization between two dropout-masked forward passes (α=1.0)' },
            { name: 'Essay Auxiliary', pct: '12%', desc: 'Essay-level OR-union supervision — Attainment has 3-4× more essay-level than sentence-level signal' },
          ].map((l) => (
            <div key={l.name} className="border border-gray-200 rounded-lg p-3 bg-white">
              <div className="flex items-center justify-between">
                <p className="text-[12px] font-semibold text-gray-800 m-0">{l.name}</p>
                <span className="text-[12px] font-mono font-semibold text-gray-600">{l.pct}</span>
              </div>
              <p className="text-[11px] text-gray-500 mt-1.5 m-0 leading-relaxed">{l.desc}</p>
            </div>
          ))}
        </div>

        <h3 className="text-[14px] font-semibold text-gray-800 m-0">Regularization Stack (11 techniques)</h3>
        <p className="text-[14px] text-gray-700 leading-relaxed">
          Dropout (0.30), BiLSTM dropout (0.25), multi-sample dropout (K=3), weight decay (0.08), R-Drop (α=1.0), AEDA augmentation (p=0.40), LLRD (d=0.92), SWA (22 checkpoints), label smoothing (0.02), gradient clipping (1.0), progressive unfreezing. Combined, these reduce the train-test F1 gap to 0.265 — substantial but manageable given the 31,000:1 parameter ratio.
        </p>
      </div>

      {/* ━━━━ ENGINEERING ━━━━ */}
      <SectionHeading id="engineering" title="Scaling Failures & Fixes" />
      <div className="py-4 space-y-4">
        <p className="text-[14px] text-gray-700 leading-relaxed">
          The initial large model (v3, 438M parameters) achieved F1=0.461 — <strong>worse</strong> than the 86M-parameter base model (F1=0.474). Systematic diagnosis revealed 7 failure modes. Targeted configuration changes (zero architectural modifications) produced v4 (F1=0.494).
        </p>
        <div className="space-y-2">
          {FAILURES.map((fm) => (
            <div key={fm.id} className="border border-gray-200 rounded-lg bg-white overflow-hidden">
              <button
                onClick={() => setExpandedFM(expandedFM === fm.id ? null : fm.id)}
                className="w-full flex items-center gap-3 px-4 py-2.5 text-left hover:bg-gray-50 transition-colors"
              >
                <span className="w-5 h-5 rounded-full bg-gray-100 flex items-center justify-center text-[11px] font-mono text-gray-600 shrink-0">{fm.id}</span>
                <span className="text-[13px] font-medium text-gray-800 flex-1">{fm.title}</span>
                <svg className={`w-3.5 h-3.5 text-gray-400 transition-transform ${expandedFM === fm.id ? 'rotate-90' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </button>
              {expandedFM === fm.id && (
                <div className="px-4 pb-3 grid grid-cols-3 gap-3 text-[12px]">
                  <div><p className="text-gray-400 m-0">v3 (failed)</p><p className="font-mono text-red-600 m-0 mt-0.5">{fm.v3}</p></div>
                  <div><p className="text-gray-400 m-0">v4 (fixed)</p><p className="font-mono text-green-600 m-0 mt-0.5">{fm.v4}</p></div>
                  <div><p className="text-gray-400 m-0">Impact</p><p className="text-gray-700 m-0 mt-0.5">{fm.impact}</p></div>
                </div>
              )}
            </div>
          ))}
        </div>
        <Callout color="amber" title="Lesson">
          <p className="m-0">Scaling a transformer on small data requires re-tuning every optimization parameter. Same hyperparameters that work for 86M parameters can catastrophically fail at 438M.</p>
        </Callout>
      </div>

      {/* ━━━━ RESULTS ━━━━ */}
      <SectionHeading id="results" title="Results" />
      <div className="py-4 space-y-6">
        {/* Embedded tabs for different result views */}
        <div className="flex gap-1 border-b border-gray-200">
          {[
            { id: 'comparison', label: 'Model Comparison' },
            { id: 'pertheme', label: 'Per-Theme F1' },
            { id: 'bottleneck', label: 'Multi-Label Bottleneck' },
          ].map((t) => (
            <button
              key={t.id}
              onClick={() => setResultView(t.id as typeof resultView)}
              className={`px-3 py-2 text-[12px] font-medium border-b-2 transition-colors ${
                resultView === t.id ? 'border-gray-800 text-gray-800' : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>

        {resultView === 'comparison' && (
          <div>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={MODELS_CMP} margin={{ top: 10, right: 10, bottom: 10, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="model" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} domain={[0, 0.6]} />
                <Tooltip contentStyle={{ fontSize: 12, borderRadius: 6, border: '1px solid #e5e7eb' }} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Bar dataKey="f1" name="Macro-F1" fill="#111827" radius={[3, 3, 0, 0]} />
                <Bar dataKey="prauc" name="PR-AUC" fill="#9ca3af" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {resultView === 'pertheme' && (
          <div>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={PER_THEME.map((r) => ({ theme: r.abbr, F1: r.f1, Precision: r.p, Recall: r.r }))} margin={{ top: 10, right: 10, bottom: 10, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="theme" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} domain={[0, 1]} />
                <Tooltip contentStyle={{ fontSize: 12, borderRadius: 6, border: '1px solid #e5e7eb' }} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Bar dataKey="F1" fill="#111827" radius={[3, 3, 0, 0]} />
                <Bar dataKey="Precision" fill="#6b7280" radius={[3, 3, 0, 0]} />
                <Bar dataKey="Recall" fill="#d1d5db" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {resultView === 'bottleneck' && (
          <div className="space-y-4">
            <Callout color="blue" title="The Multi-Label Bottleneck">
              <p className="m-0">When themes appear alone, AWARE achieves F1 = 0.568–0.896 with <strong>perfect precision</strong>. The gap between single-theme and overall F1 reveals the bottleneck is multi-label disentanglement, not theme detection.</p>
            </Callout>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={SINGLE_MULTI} margin={{ top: 10, right: 10, bottom: 10, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="theme" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} domain={[0, 1]} />
                <Tooltip contentStyle={{ fontSize: 12, borderRadius: 6, border: '1px solid #e5e7eb' }} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Bar dataKey="single" name="Single-Theme F1" fill="#2563eb" radius={[3, 3, 0, 0]} />
                <Bar dataKey="overall" name="Overall F1" fill="#d1d5db" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Threshold optimization (always visible below tabs) */}
        <Callout color="green" title="Threshold Optimization: +0.084 Macro-F1">
          <p className="m-0">The single highest-impact post-training decision. Bigger than scaling from 86M → 438M parameters (+0.020). Default 0.5 threshold is catastrophically misaligned with actual positive rates (2-24%). Rarest themes need thresholds as low as 0.05.</p>
        </Callout>
      </div>

      {/* ━━━━ CONCLUSION ━━━━ */}
      <SectionHeading id="conclusion" title="Conclusion" />
      <div className="py-4 space-y-6">
        <h3 className="text-[14px] font-semibold text-gray-800 m-0">Research Questions</h3>
        {[
          { rq: 'RQ1', q: 'Can AWARE classify multiple CCW themes simultaneously?', a: 'Yes. Macro-F1 = 0.494, ROC-AUC > 0.80 for all 8 themes. On single-theme sentences: F1 = 0.568–0.896 with perfect precision.' },
          { rq: 'RQ2', q: 'How does model scale affect performance?', a: 'Modest: +0.020 F1 (not statistically significant). Naive scaling fails — 7 failure modes required systematic re-tuning. Threshold optimization (+0.084) contributed 4× more.' },
          { rq: 'RQ3', q: 'What are the fundamental challenges?', a: 'Multi-label co-occurrence is the primary bottleneck. Data scarcity for rare themes drives overfitting (Attainment: 0.519 train-test gap). Future improvements come from data, not architecture.' },
        ].map((item) => (
          <div key={item.rq} className="border border-gray-200 rounded-lg p-4 bg-white">
            <p className="text-[11px] font-mono text-gray-400 m-0">{item.rq}</p>
            <p className="text-[13px] font-semibold text-gray-700 mt-0.5 m-0">{item.q}</p>
            <p className="text-[13px] text-gray-600 mt-2 m-0 leading-relaxed">{item.a}</p>
          </div>
        ))}

        <h3 className="text-[14px] font-semibold text-gray-800 m-0">Contributions</h3>
        <ol className="space-y-2 list-decimal list-inside text-[13px] text-gray-600 leading-relaxed pl-1">
          <li><strong>Data processing pipeline</strong> — noisy heterogeneous data → clean multi-label dataset with novel embedding-based semantic cleaning (1,705 sentences removed, 8.6%)</li>
          <li><strong>Empirical characterization</strong> — embedding analysis proving 38.7% LOO-KNN error as architecture-independent lower bound on classification error</li>
          <li><strong>AWARE framework</strong> — DAPT + essay-level context + asymmetric loss + per-theme threshold optimization achieving Macro-F1 = 0.494</li>
          <li><strong>Systematic failure analysis</strong> — 7 documented failure modes when scaling from 86M → 438M parameters, with targeted configuration fixes</li>
          <li><strong>ALMA Research Dashboard</strong> — web platform unifying dataset, model deployment, and annotation workflow for the research team</li>
        </ol>
      </div>

      {/* Footer spacer */}
      <div className="h-20" />
    </PageContainer>
  );
}
