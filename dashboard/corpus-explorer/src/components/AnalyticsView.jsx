import { useState, useMemo } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
  PieChart, Pie, Legend as RLegend
} from 'recharts'
import { THEME_COLORS, THEMES_ORDER, THEME_SHORT } from '../constants/themes'
import { fmt, pct } from '../utils/helpers.jsx'

// ─── Shared small components ────────────────────────────────────
function ThemeDot({ theme, size = 10, showLabel = false }) {
  return (
    <span className="theme-dot-wrap" title={theme}>
      <span className="theme-dot" style={{ backgroundColor: THEME_COLORS[theme] || '#94a3b8', width: size, height: size }} />
      {showLabel && <span className="theme-dot-label">{theme}</span>}
    </span>
  )
}

function Stat({ value, label, sub, accent }) {
  return (
    <div className="stat-card" style={accent ? { borderTop: `3px solid ${accent}` } : {}}>
      <div className="stat-val">{fmt(value)}</div>
      <div className="stat-lbl">{label}</div>
      {sub && <div className="stat-sub">{sub}</div>}
    </div>
  )
}

function MiniBar({ data, colorFn, maxVal }) {
  const max = maxVal || Math.max(...data.map(d => d.value), 1)
  return (
    <div className="mini-bar-chart">
      {data.map(({ label, value, extra }) => (
        <div key={label} className="mini-bar-row">
          <span className="mini-bar-label" title={label}>{label}</span>
          <div className="mini-bar-track">
            <div className="mini-bar-fill" style={{
              width: `${Math.max((value / max) * 100, 0.5)}%`,
              backgroundColor: colorFn ? colorFn(label) : 'var(--primary)'
            }} />
          </div>
          <span className="mini-bar-val">{fmt(value)}{extra ? ` ${extra}` : ''}</span>
        </div>
      ))}
    </div>
  )
}

// Custom tooltip for recharts
function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div className="chart-tooltip">
      <div className="chart-tooltip-label">{label || payload[0]?.name}</div>
      {payload.map((p, i) => (
        <div key={i} className="chart-tooltip-val">
          <span className="chart-tooltip-dot" style={{ background: p.fill || p.color }} />
          {fmt(p.value)}
        </div>
      ))}
    </div>
  )
}

// ─── ANALYTICS VIEW ─────────────────────────────────────────────
export default function AnalyticsView({ stats }) {
  const [selectedTheme, setSelectedTheme] = useState(null)
  const [activeSection, setActiveSection] = useState(null)

  const bc = stats.basic_counts || {}
  const qm = stats.quality_metrics || {}
  const tr = stats.training_readiness || {}
  const rs = tr.recommended_split || {}
  const sentCounts = stats.theme_sentence_counts || {}
  const essayCounts = stats.theme_essay_counts || {}
  const totalSents = bc.total_sentences || 1
  const totalEssays = bc.total_essays || 1
  const density = stats.annotation_density || {}
  const details = stats.per_theme_details || {}
  const provenance = stats.sentence_provenance || {}
  const semYear = stats.label_dist_by_semester_year || {}
  const sentHist = stats.sentence_length_histogram || []
  const essayHist = stats.essay_length_histogram || []

  const cooc = stats.cooccurrence || {}
  const coocLabels = cooc.labels || THEMES_ORDER
  const coocMatrix = useMemo(() => cooc.matrix || [], [cooc.matrix])
  const matrixMax = useMemo(() => {
    let max = 0
    for (let i = 0; i < coocMatrix.length; i++)
      for (let j = 0; j < (coocMatrix[i] || []).length; j++)
        if (i !== j && coocMatrix[i][j] > max) max = coocMatrix[i][j]
    return max || 1
  }, [coocMatrix])

  // Recharts data for theme distribution bar
  const themeBarData = THEMES_ORDER.map(t => ({
    name: t,
    sentences: sentCounts[t] || 0,
    essays: essayCounts[t] || 0,
    fill: THEME_COLORS[t]
  }))

  // Annotation coverage donut
  const coverageData = [
    { name: 'Annotated', value: bc.annotated_sentences || 0, fill: 'var(--primary)' },
    { name: 'Unannotated', value: bc.class0_only_sentences || 0, fill: '#e2e8f0' },
  ]

  // Histogram data (FIX: arrays of {range, count} → recharts format)
  const sentHistData = Array.isArray(sentHist) ? sentHist.map(d => ({
    name: d.range, count: d.count
  })) : []
  const essayHistData = Array.isArray(essayHist) ? essayHist.map(d => ({
    name: d.range, count: d.count
  })) : []

  const themeDetail = selectedTheme ? details[selectedTheme] : null

  // Section anchors for navigation
  const sections = [
    { id: 'overview', label: 'Overview' },
    { id: 'pipeline', label: 'Data Pipeline' },
    { id: 'themes', label: 'Themes' },
    { id: 'cooccurrence', label: 'Co-occurrence' },
    { id: 'temporal', label: 'Temporal' },
    { id: 'sources', label: 'Sources' },
    { id: 'distributions', label: 'Distributions' },
    { id: 'training', label: 'Training' },
  ]

  return (
    <div className="analytics">
      {/* ── Section Nav ── */}
      <nav className="section-nav">
        {sections.map(s => (
          <a key={s.id} href={`#${s.id}`}
            className={activeSection === s.id ? 'active' : ''}
            onClick={e => { e.preventDefault(); setActiveSection(s.id); document.getElementById(s.id)?.scrollIntoView({ behavior: 'smooth' }) }}>
            {s.label}
          </a>
        ))}
      </nav>

      {/* ── Theme Legend (always visible) ── */}
      <div className="theme-legend-bar">
        {THEMES_ORDER.map(t => (
          <span key={t} className="legend-chip" onClick={() => setSelectedTheme(selectedTheme === t ? null : t)}
            style={selectedTheme === t ? { background: THEME_COLORS[t], color: '#fff' } : {}}>
            <span className="theme-dot" style={{ backgroundColor: THEME_COLORS[t], width: 8, height: 8 }} />
            {t}
          </span>
        ))}
        <span className="legend-chip muted">
          <span className="theme-dot" style={{ backgroundColor: '#94a3b8', width: 8, height: 8 }} />
          class_0
        </span>
      </div>

      {/* ════════════════════════════════════════════════════════════ */}
      {/* SECTION 1: DATASET OVERVIEW                                 */}
      {/* ════════════════════════════════════════════════════════════ */}
      <section id="overview">
        <h2 className="section-title">Dataset Overview</h2>
        <div className="stats-row-4">
          <Stat value={bc.total_essays} label="Total Essays" sub="Unified from 16 sources" />
          <Stat value={bc.total_sentences} label="Total Sentences" sub={`${qm.avg_sentences_per_essay || '--'} avg per essay`} />
          <Stat value={bc.annotated_sentences} label="Annotated Sentences" sub={`${qm.annotation_coverage_pct || '--'}% coverage`} accent="var(--primary)" />
          <Stat value={bc.unique_themes} label="CCW Themes" sub={`${qm.multi_label_rate_pct || '--'}% multi-label`} />
        </div>

        {/* Annotation Coverage Split */}
        <div className="grid-2-auto">
          <div className="card">
            <h3>Annotation Coverage</h3>
            <p className="card-desc">Proportion of sentences with identified CCW themes vs. annotator-reviewed with no theme found (class_0)</p>
            <div className="donut-container">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={coverageData} cx="50%" cy="50%" innerRadius={60} outerRadius={90}
                    paddingAngle={2} dataKey="value" nameKey="name" stroke="none">
                    {coverageData.map((d, i) => <Cell key={i} fill={i === 0 ? '#2563eb' : '#e2e8f0'} />)}
                  </Pie>
                  <Tooltip content={<ChartTooltip />} />
                </PieChart>
              </ResponsiveContainer>
              <div className="donut-center">
                <span className="donut-pct">{qm.annotation_coverage_pct}%</span>
                <span className="donut-label">Annotated</span>
              </div>
            </div>
          </div>
          <div className="card">
            <h3>Key Metrics</h3>
            <p className="card-desc">Dataset characteristics for multi-label theme classification</p>
            <div className="metrics-list">
              <div className="metric"><span>Essays with any theme</span><span className="metric-v">{fmt(qm.essays_with_any_theme)}</span></div>
              <div className="metric"><span>Essays with no theme</span><span className="metric-v">{fmt(qm.essays_no_theme)}</span></div>
              <div className="metric"><span>Multi-label essays</span><span className="metric-v">{fmt(qm.multi_label_essays)} ({qm.multi_label_rate_pct}%)</span></div>
              <div className="metric"><span>Avg themes per essay</span><span className="metric-v">{qm.avg_themes_per_essay}</span></div>
              <div className="metric"><span>Avg sentences per essay</span><span className="metric-v">{qm.avg_sentences_per_essay}</span></div>
              <div className="metric"><span>Avg words per sentence</span><span className="metric-v">{qm.avg_words_per_sentence}</span></div>
              <div className="metric"><span>Annotation density avg</span><span className="metric-v">{qm.avg_annotated_sentences_per_essay} sents/essay</span></div>
            </div>
          </div>
        </div>
      </section>

      {/* ════════════════════════════════════════════════════════════ */}
      {/* SECTION 2: DATA PIPELINE                                    */}
      {/* ════════════════════════════════════════════════════════════ */}
      <section id="pipeline">
        <h2 className="section-title">Data Pipeline</h2>
        <p className="section-subtitle">From raw annotator files to a unified, deduplicated research dataset</p>

        <div className="pipeline">
          <div className="pipeline-stage">
            <div className="pipeline-icon">1</div>
            <h4>Raw Collection</h4>
            <div className="pipeline-stat">{fmt(bc.unique_data_sources)} sources</div>
            <p>Multiple annotator batches collected across semesters from the ALMA project (2017-2025). Excel files with sentence-level theme annotations.</p>
            <div className="pipeline-detail">
              <span>{fmt(bc.unique_classes)} courses</span>
              <span>{Object.keys(stats.semester_year_distribution || stats.semester_distribution || {}).length} time periods</span>
            </div>
          </div>
          <div className="pipeline-arrow">&#8594;</div>
          <div className="pipeline-stage">
            <div className="pipeline-icon">2</div>
            <h4>Processing &amp; Unification</h4>
            <div className="pipeline-stat">{fmt(bc.total_essays)} essays</div>
            <p>Deduplicated, normalized text, merged annotations from overlapping sources, assigned unified essay IDs.</p>
            <div className="pipeline-detail">
              <span>{fmt(bc.total_sentences)} sentences</span>
              <span>{qm.avg_sentences_per_essay} avg sents/essay</span>
            </div>
          </div>
          <div className="pipeline-arrow">&#8594;</div>
          <div className="pipeline-stage accent">
            <div className="pipeline-icon">3</div>
            <h4>Annotated Dataset</h4>
            <div className="pipeline-stat">{fmt(bc.annotated_sentences)} labeled</div>
            <p>{bc.unique_themes} CCW themes identified at sentence level. {pct(bc.annotated_sentences, bc.total_sentences)} of sentences have theme annotations.</p>
            <div className="pipeline-detail">
              <span>{fmt(qm.multi_label_essays)} multi-label</span>
              <span>{tr.label_imbalance_ratio}x imbalance</span>
            </div>
          </div>
        </div>
      </section>

      {/* ════════════════════════════════════════════════════════════ */}
      {/* SECTION 3: THEME ANALYSIS                                   */}
      {/* ════════════════════════════════════════════════════════════ */}
      <section id="themes">
        <h2 className="section-title">Theme Distribution</h2>
        <p className="section-subtitle">Sentence and essay counts per Community Cultural Wealth theme. Click any theme to explore its breakdown.</p>

        <div className="grid-2-wide">
          {/* Recharts bar chart */}
          <div className="card">
            <h3>Sentences per Theme</h3>
            <ResponsiveContainer width="100%" height={340}>
              <BarChart data={themeBarData} layout="vertical" margin={{ left: 140, right: 30, top: 5, bottom: 5 }}>
                <XAxis type="number" tickFormatter={v => v >= 1000 ? `${(v/1000).toFixed(0)}k` : v} />
                <YAxis type="category" dataKey="name" width={130} tick={{ fontSize: 12 }} />
                <Tooltip content={<ChartTooltip />} />
                <Bar dataKey="sentences" radius={[0, 4, 4, 0]} cursor="pointer"
                  onClick={(d) => setSelectedTheme(selectedTheme === d.name ? null : d.name)}>
                  {themeBarData.map((d, i) => (
                    <Cell key={i} fill={d.fill} opacity={selectedTheme && selectedTheme !== d.name ? 0.3 : 1} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Theme table with details */}
          <div className="card">
            <h3>Theme Breakdown</h3>
            <table className="theme-table">
              <thead>
                <tr>
                  <th>Theme</th>
                  <th>Sents</th>
                  <th>% Sents</th>
                  <th>Essays</th>
                  <th>Avg/Essay</th>
                </tr>
              </thead>
              <tbody>
                {THEMES_ORDER.map(theme => {
                  const sc = sentCounts[theme] || 0
                  const ec = essayCounts[theme] || 0
                  const isSelected = selectedTheme === theme
                  return (
                    <tr key={theme} className={isSelected ? 'row-active' : ''} onClick={() => setSelectedTheme(isSelected ? null : theme)}>
                      <td><ThemeDot theme={theme} showLabel /></td>
                      <td className="num">{fmt(sc)}</td>
                      <td className="num">{pct(sc, totalSents)}</td>
                      <td className="num">{fmt(ec)}</td>
                      <td className="num">{density[theme]?.mean?.toFixed(2) || '--'}</td>
                    </tr>
                  )
                })}
                <tr className="row-muted">
                  <td><ThemeDot theme="class_0" showLabel /></td>
                  <td className="num">{fmt(sentCounts['class_0'] || bc.class0_only_sentences)}</td>
                  <td className="num">{pct(bc.class0_only_sentences, totalSents)}</td>
                  <td colSpan={2} className="muted-text">No theme identified</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* Theme Detail Drilldown */}
        {themeDetail && (
          <div className="card theme-detail" style={{ borderLeft: `4px solid ${THEME_COLORS[selectedTheme]}` }}>
            <div className="card-header">
              <h3><ThemeDot theme={selectedTheme} size={12} showLabel /> — Deep Dive</h3>
              <button className="btn-close" onClick={() => setSelectedTheme(null)}>Close</button>
            </div>
            <div className="grid-3">
              <div className="sub-card">
                <h4>By Course</h4>
                <MiniBar data={Object.entries(themeDetail.by_course || {}).sort((a, b) => b[1] - a[1]).slice(0, 10).map(([k, v]) => ({ label: k, value: v }))} colorFn={() => THEME_COLORS[selectedTheme]} />
              </div>
              <div className="sub-card">
                <h4>By Year</h4>
                <MiniBar data={Object.entries(themeDetail.by_year || {}).sort((a, b) => a[0].localeCompare(b[0])).map(([k, v]) => ({ label: k, value: v }))} colorFn={() => THEME_COLORS[selectedTheme]} />
              </div>
              <div className="sub-card">
                <h4>By Data Source</h4>
                <MiniBar data={Object.entries(themeDetail.by_data_source || {}).sort((a, b) => b[1] - a[1]).slice(0, 8).map(([k, v]) => ({ label: k.replace('ALMA_', ''), value: v }))} colorFn={() => THEME_COLORS[selectedTheme]} />
              </div>
            </div>
            <div className="sample-section">
              <h4>Example Annotated Sentences</h4>
              <div className="samples">
                {(themeDetail.samples || []).map((s, i) => (
                  <div key={i} className="sample-card">
                    <p className="sample-text">&ldquo;{s.text}&rdquo;</p>
                    <div className="sample-meta">
                      <span className="tag">{s.essay_id}</span>
                      <span className="tag">{s.class}</span>
                      <span className="tag">{s.semester} {s.year}</span>
                      <span className="tag accent">{s.data_source?.replace('ALMA_', '')}</span>
                      {(s.all_themes || []).filter(t => t !== selectedTheme).map(t => (
                        <ThemeDot key={t} theme={t} size={8} showLabel />
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </section>

      {/* ════════════════════════════════════════════════════════════ */}
      {/* SECTION 4: CO-OCCURRENCE MATRIX                             */}
      {/* ════════════════════════════════════════════════════════════ */}
      <section id="cooccurrence">
        <h2 className="section-title">Theme Co-occurrence</h2>
        <p className="section-subtitle">How often themes appear together in the same essay. Darker cells indicate stronger co-occurrence.</p>

        <div className="card">
          <div className="matrix-wrap">
            <table className="cooc-matrix">
              <thead>
                <tr>
                  <th></th>
                  {coocLabels.map(l => (
                    <th key={l} className="matrix-col-header" title={l}>
                      <span className="theme-dot" style={{ backgroundColor: THEME_COLORS[l], width: 6, height: 6 }} />
                      {THEME_SHORT[l] || l.substring(0, 3)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {coocLabels.map((rl, i) => (
                  <tr key={rl}>
                    <td className="matrix-row-label">
                      <span className="theme-dot" style={{ backgroundColor: THEME_COLORS[rl], width: 6, height: 6 }} />
                      {rl}
                    </td>
                    {coocLabels.map((cl, j) => {
                      const val = (coocMatrix[i] || [])[j] || 0
                      const isDiag = i === j
                      const intensity = isDiag ? 0 : val / matrixMax
                      return (
                        <td key={cl} className={`matrix-cell ${isDiag ? 'diag' : ''}`} style={{
                          backgroundColor: isDiag ? '#f1f5f9' : `rgba(37, 99, 235, ${Math.min(intensity * 0.85, 0.85)})`,
                          color: intensity > 0.35 ? '#fff' : '#374151'
                        }} title={`${rl} + ${cl}: ${val} essays`}>
                          {val > 0 ? fmt(val) : ''}
                        </td>
                      )
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* ════════════════════════════════════════════════════════════ */}
      {/* SECTION 5: TEMPORAL ANALYSIS                                */}
      {/* ════════════════════════════════════════════════════════════ */}
      <section id="temporal">
        <h2 className="section-title">Theme Coverage by Semester</h2>
        <p className="section-subtitle">Essay counts per theme across academic periods. Color intensity reflects proportion within each period.</p>

        <div className="card">
          <div className="temporal-wrap">
            <table className="temporal-table">
              <thead>
                <tr>
                  <th className="col-period">Period</th>
                  <th className="col-total">Total</th>
                  {THEMES_ORDER.map(t => (
                    <th key={t} title={t}>
                      <span className="theme-dot" style={{ backgroundColor: THEME_COLORS[t], width: 6, height: 6 }} />
                      {THEME_SHORT[t]}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.entries(semYear).sort((a, b) => {
                  // Sort chronologically
                  const order = ['Pilot', 'Spring', 'Summer', 'Fall']
                  const parseKey = k => {
                    const parts = k.split(' ')
                    const year = parseInt(parts[parts.length - 1]) || 0
                    const sem = order.indexOf(parts[0]) >= 0 ? order.indexOf(parts[0]) : 99
                    return year * 10 + sem
                  }
                  return parseKey(a[0]) - parseKey(b[0])
                }).map(([period, data]) => {
                  const periodTotal = THEMES_ORDER.reduce((s, t) => s + (data[t] || 0), 0)
                  return (
                    <tr key={period}>
                      <td className="period-cell">{period}</td>
                      <td className="total-cell">{fmt(periodTotal)}</td>
                      {THEMES_ORDER.map(t => {
                        const val = data[t] || 0
                        const p = val / (periodTotal || 1) * 100
                        return (
                          <td key={t} className="heat-cell" title={`${t}: ${val} (${p.toFixed(1)}%)`}>
                            {val > 0 && (
                              <span className="heat-val" style={{
                                opacity: Math.max(0.25, Math.min(p / 40, 1)),
                                backgroundColor: THEME_COLORS[t]
                              }}>{val}</span>
                            )}
                          </td>
                        )
                      })}
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* ════════════════════════════════════════════════════════════ */}
      {/* SECTION 6: DATA SOURCES                                     */}
      {/* ════════════════════════════════════════════════════════════ */}
      <section id="sources">
        <h2 className="section-title">Data Source Provenance</h2>
        <p className="section-subtitle">Annotation coverage and theme contributions from each data collection effort</p>

        <div className="grid-2-auto">
          <div className="card">
            <h3>Source Breakdown</h3>
            <div className="provenance-list">
              {Object.entries(provenance).sort((a, b) => (b[1].total_sentences || 0) - (a[1].total_sentences || 0)).map(([src, info]) => {
                const rate = info.annotation_rate_pct || 0
                return (
                  <div key={src} className="prov-row">
                    <div className="prov-header">
                      <span className="prov-name">{src.replace('ALMA_', '')}</span>
                      <span className={`rate-badge ${rate > 30 ? 'high' : rate > 10 ? 'med' : 'low'}`}>{rate}%</span>
                    </div>
                    <div className="prov-bar">
                      <div className="prov-bar-fill" style={{ width: `${Math.min(rate, 100)}%` }} />
                    </div>
                    <div className="prov-stats">
                      <span>{fmt(info.total_essays)} essays</span>
                      <span>{fmt(info.total_sentences)} sents</span>
                      <span>{fmt(info.annotated_sentences)} annotated</span>
                    </div>
                    <div className="prov-themes">
                      {(info.themes_contributed || []).map(t => <ThemeDot key={t} theme={t} size={7} showLabel />)}
                    </div>
                  </div>
                )
              })}
            </div>
          </div>

          <div className="card">
            <h3>Multi-label Analysis</h3>
            <p className="card-desc">Number of themes per essay (excluding class_0)</p>
            <MiniBar data={Object.entries(stats.multi_label_distribution || {}).sort((a, b) => Number(a[0]) - Number(b[0])).map(([k, v]) => ({
              label: `${k} theme${k === '1' ? '' : 's'}`,
              value: v,
              extra: `(${pct(v, totalEssays)})`
            }))} />
            <h3 style={{ marginTop: 20 }}>Top Theme Combinations</h3>
            <p className="card-desc">Most frequent multi-label pairings across essays</p>
            <div className="combo-list">
              {(stats.top_theme_combinations || []).slice(0, 10).map((combo, i) => (
                <div key={i} className="combo-row">
                  <span className="combo-rank">#{i + 1}</span>
                  <span className="combo-themes">{(combo.themes || []).map(t => <ThemeDot key={t} theme={t} size={8} showLabel />)}</span>
                  <span className="combo-count">{fmt(combo.count)}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* ════════════════════════════════════════════════════════════ */}
      {/* SECTION 7: DISTRIBUTIONS                                    */}
      {/* ════════════════════════════════════════════════════════════ */}
      <section id="distributions">
        <h2 className="section-title">Distribution Analysis</h2>
        <p className="section-subtitle">Structural properties of the essay corpus</p>

        <div className="grid-2">
          <div className="card">
            <h3>Essays by Year</h3>
            <MiniBar data={Object.entries(stats.year_distribution || {}).sort((a, b) => a[0].localeCompare(b[0])).map(([k, v]) => ({ label: k, value: v }))} />
          </div>
          <div className="card">
            <h3>Top Courses</h3>
            <MiniBar data={Object.entries(stats.class_distribution || {}).sort((a, b) => b[1] - a[1]).slice(0, 12).map(([k, v]) => ({ label: k || '(unknown)', value: v }))} />
          </div>
        </div>

        <div className="grid-2">
          <div className="card">
            <h3>Sentence Length Distribution</h3>
            <p className="card-desc">Words per sentence</p>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={sentHistData.slice(0, 20)} margin={{ left: 0, right: 10, top: 5, bottom: 5 }}>
                <XAxis dataKey="name" tick={{ fontSize: 10 }} interval={2} />
                <YAxis tickFormatter={v => v >= 1000 ? `${(v/1000).toFixed(0)}k` : v} tick={{ fontSize: 10 }} />
                <Tooltip content={<ChartTooltip />} />
                <Bar dataKey="count" fill="#2563eb" radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="card">
            <h3>Essay Length Distribution</h3>
            <p className="card-desc">Sentences per essay</p>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={essayHistData.slice(0, 15)} margin={{ left: 0, right: 10, top: 5, bottom: 5 }}>
                <XAxis dataKey="name" tick={{ fontSize: 10 }} interval={1} />
                <YAxis tickFormatter={v => v >= 1000 ? `${(v/1000).toFixed(0)}k` : v} tick={{ fontSize: 10 }} />
                <Tooltip content={<ChartTooltip />} />
                <Bar dataKey="count" fill="#7c3aed" radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </section>

      {/* ════════════════════════════════════════════════════════════ */}
      {/* SECTION 8: TRAINING READINESS                               */}
      {/* ════════════════════════════════════════════════════════════ */}
      <section id="training">
        <h2 className="section-title">Training Readiness</h2>
        <p className="section-subtitle">Dataset statistics relevant for multi-label classification model training</p>

        <div className="grid-2-auto">
          <div className="card">
            <h3>Recommended Split (80/10/10)</h3>
            <div className="metrics-list">
              <div className="metric"><span>Total samples</span><span className="metric-v">{fmt(tr.total_samples)}</span></div>
              <div className="metric"><span>Train set</span><span className="metric-v">{fmt(rs.train)} ({rs.train_pct}%)</span></div>
              <div className="metric"><span>Validation set</span><span className="metric-v">{fmt(rs.val)} ({rs.val_pct}%)</span></div>
              <div className="metric"><span>Test set</span><span className="metric-v">{fmt(rs.test)} ({rs.test_pct?.toFixed(1)}%)</span></div>
              <div className="metric"><span>Number of labels</span><span className="metric-v">{tr.num_labels}</span></div>
              <div className="metric"><span>Label imbalance ratio</span><span className="metric-v">{tr.label_imbalance_ratio}x</span></div>
              <div className="metric"><span>Label sparsity</span><span className="metric-v">{tr.sparsity_pct}%</span></div>
            </div>
          </div>
          <div className="card">
            <h3>Class Imbalance</h3>
            <p className="card-desc">Annotated vs. unannotated sentence distribution</p>
            <div className="imbalance-bar">
              <div className="imbalance-pos" style={{ width: `${qm.annotation_coverage_pct}%` }}>{qm.annotation_coverage_pct}%</div>
              <div className="imbalance-neg">{(100 - (qm.annotation_coverage_pct || 0)).toFixed(1)}%</div>
            </div>
            <div className="imbalance-legend">
              <span><span className="dot-blue" /> Annotated sentences</span>
              <span><span className="dot-gray" /> class_0 (no theme)</span>
            </div>

            <h4 style={{ marginTop: 16, fontSize: 13 }}>Theme Essay Counts</h4>
            <MiniBar
              data={THEMES_ORDER.map(t => ({ label: t, value: essayCounts[t] || 0 }))}
              colorFn={l => THEME_COLORS[l]}
            />

            <div className="info-callout">
              Unannotated essays available for prediction: <strong>{fmt(tr.unannotated_essays || qm.essays_no_theme)}</strong>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}
