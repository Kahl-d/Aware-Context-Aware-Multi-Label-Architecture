import { useState, useEffect, useMemo, useRef } from 'react'
import { THEME_COLORS, THEMES_ORDER } from '../constants/themes'
import { highlightText, fmt, normalizeEssay } from '../utils/helpers.jsx'

function ThemeDot({ theme, size = 8, showLabel = false }) {
  return (
    <span className="theme-dot-wrap" title={theme}>
      <span className="theme-dot" style={{ backgroundColor: THEME_COLORS[theme] || '#94a3b8', width: size, height: size }} />
      {showLabel && <span className="theme-dot-label">{theme}</span>}
    </span>
  )
}

export default function DataExplorer({ essays, stats }) {
  const [searchText, setSearchText] = useState('')
  const [debouncedSearch, setDebouncedSearch] = useState('')
  const [filterTheme, setFilterTheme] = useState('all')
  const [filterYear, setFilterYear] = useState('all')
  const [filterClass, setFilterClass] = useState('all')
  const [filterSource, setFilterSource] = useState('all')
  const [filterSemester, setFilterSemester] = useState('all')
  const [currentPage, setCurrentPage] = useState(1)
  const [expandedEssay, setExpandedEssay] = useState(null)
  const [sortField, setSortField] = useState('id')
  const [sortDir, setSortDir] = useState('asc')
  const itemsPerPage = 50
  const searchTimer = useRef(null)

  // Wrap filter setters to also reset page
  const updateFilter = (setter) => (val) => { setter(val); setCurrentPage(1) }

  // Debounce search
  useEffect(() => {
    clearTimeout(searchTimer.current)
    searchTimer.current = setTimeout(() => { setDebouncedSearch(searchText); setCurrentPage(1) }, 250)
    return () => clearTimeout(searchTimer.current)
  }, [searchText])

  const uniqueYears = useMemo(() => Object.keys(stats.year_distribution || {}).sort(), [stats])
  const uniqueClasses = useMemo(() => Object.keys(stats.class_distribution || {}).sort(), [stats])
  const uniqueSources = useMemo(() => Object.keys(stats.data_source_breakdown || {}).sort(), [stats])
  const uniqueSemesters = useMemo(() => Object.keys(stats.semester_distribution || {}).sort(), [stats])

  const filtered = useMemo(() => {
    if (!essays) return []
    let result = essays.map(normalizeEssay).filter(e => {
      if (debouncedSearch) {
        const s = debouncedSearch.toLowerCase()
        if (!e.essay_preview?.toLowerCase().includes(s) && !e.given_id?.toLowerCase().includes(s) &&
            !e.class_name.toLowerCase().includes(s) &&
            !(e.metadata?.alma_id || '').toLowerCase().includes(s)) return false
      }
      if (filterTheme !== 'all' && !e.theme_labels?.[filterTheme]) return false
      if (filterYear !== 'all' && String(e.year) !== filterYear) return false
      if (filterClass !== 'all' && e.class_name !== filterClass) return false
      if (filterSource !== 'all' && e.data_source !== filterSource) return false
      if (filterSemester !== 'all' && e.semester !== filterSemester) return false
      return true
    })
    result.sort((a, b) => {
      let va, vb
      if (sortField === 'id') { va = a.given_id; vb = b.given_id }
      else if (sortField === 'year') { va = String(a.year); vb = String(b.year) }
      else if (sortField === 'semester') { va = a.semester; vb = b.semester }
      else if (sortField === 'class') { va = a.class_name; vb = b.class_name }
      else if (sortField === 'sents') { va = a.sentence_count || 0; vb = b.sentence_count || 0 }
      else if (sortField === 'annotated') { va = a.annotated_sentence_count || 0; vb = b.annotated_sentence_count || 0 }
      if (va < vb) return sortDir === 'asc' ? -1 : 1
      if (va > vb) return sortDir === 'asc' ? 1 : -1
      return 0
    })
    return result
  }, [essays, debouncedSearch, filterTheme, filterYear, filterClass, filterSource, filterSemester, sortField, sortDir])

  const totalPages = Math.ceil(filtered.length / itemsPerPage)
  const safePage = Math.min(currentPage, Math.max(1, totalPages))
  const page = useMemo(() => {
    const start = (safePage - 1) * itemsPerPage
    return filtered.slice(start, start + itemsPerPage)
  }, [filtered, safePage])

  const handleSort = (f) => {
    if (sortField === f) setSortDir(d => d === 'asc' ? 'desc' : 'asc')
    else { setSortField(f); setSortDir('asc') }
  }
  const sortIcon = (f) => sortField === f ? (sortDir === 'asc' ? ' \u2191' : ' \u2193') : ''

  const clearFilters = () => {
    setSearchText('')
    setDebouncedSearch('')
    setFilterTheme('all')
    setFilterYear('all')
    setFilterClass('all')
    setFilterSource('all')
    setFilterSemester('all')
  }

  const hasFilters = searchText || filterTheme !== 'all' || filterYear !== 'all' || filterClass !== 'all' || filterSource !== 'all' || filterSemester !== 'all'

  return (
    <div className="data-explorer">
      {/* Toolbar */}
      <div className="toolbar">
        <div className="toolbar-row">
          <input type="text" placeholder="Search by ID, text, course, alma_id..." value={searchText}
            onChange={e => setSearchText(e.target.value)} className="search-input" />
          <select value={filterTheme} onChange={e => updateFilter(setFilterTheme)(e.target.value)}>
            <option value="all">All Themes</option>
            {THEMES_ORDER.map(t => <option key={t} value={t}>{t}</option>)}
          </select>
          <select value={filterSemester} onChange={e => updateFilter(setFilterSemester)(e.target.value)}>
            <option value="all">Semester</option>
            {uniqueSemesters.map(s => <option key={s} value={s}>{s}</option>)}
          </select>
          <select value={filterYear} onChange={e => updateFilter(setFilterYear)(e.target.value)}>
            <option value="all">Year</option>
            {uniqueYears.map(y => <option key={y} value={y}>{y}</option>)}
          </select>
          <select value={filterClass} onChange={e => updateFilter(setFilterClass)(e.target.value)}>
            <option value="all">Course</option>
            {uniqueClasses.map(c => <option key={c} value={c}>{c || '(empty)'}</option>)}
          </select>
          <select value={filterSource} onChange={e => updateFilter(setFilterSource)(e.target.value)}>
            <option value="all">Source</option>
            {uniqueSources.map(s => <option key={s} value={s}>{s.replace('ALMA_', '')}</option>)}
          </select>
          {hasFilters && <button className="btn-clear" onClick={clearFilters}>Clear all</button>}
        </div>
      </div>

      {/* Results info */}
      <div className="results-info">
        <span>{fmt(filtered.length)} essays</span>
        {totalPages > 1 && <span className="results-page">Page {currentPage} of {totalPages}</span>}
        {hasFilters && <span className="results-filtered">(filtered)</span>}
      </div>

      {/* Table */}
      <div className="table-container">
        <table className="data-table">
          <thead>
            <tr>
              <th className="col-id sortable" onClick={() => handleSort('id')}>Essay ID{sortIcon('id')}</th>
              <th className="col-sem sortable" onClick={() => handleSort('semester')}>Semester{sortIcon('semester')}</th>
              <th className="col-year sortable" onClick={() => handleSort('year')}>Year{sortIcon('year')}</th>
              <th className="col-class sortable" onClick={() => handleSort('class')}>Course{sortIcon('class')}</th>
              <th className="col-sents sortable" onClick={() => handleSort('sents')}>Sentences{sortIcon('sents')}</th>
              <th className="col-ann sortable" onClick={() => handleSort('annotated')}>Annotated{sortIcon('annotated')}</th>
              <th className="col-themes">Themes</th>
              <th className="col-text">Essay Preview</th>
            </tr>
          </thead>
          <tbody>
            {page.flatMap(e => {
              const themes = Object.entries(e.theme_labels || {}).filter(([, v]) => v).map(([k]) => k)
              const isExpanded = expandedEssay === e.given_id
              const rows = [
                <tr key={e.given_id} className={`data-row ${isExpanded ? 'expanded' : ''}`} onClick={() => setExpandedEssay(isExpanded ? null : e.given_id)}>
                  <td className="col-id mono">{e.given_id?.replace('essay_', '#')}</td>
                  <td className="col-sem">{e.semester}</td>
                  <td className="col-year">{e.year}</td>
                  <td className="col-class">{e.class_name}</td>
                  <td className="col-sents center">{e.sentence_count}</td>
                  <td className="col-ann center">{e.annotated_sentence_count}</td>
                  <td className="col-themes">{themes.length === 0 ? <span className="muted-text">—</span> : themes.map(t => <ThemeDot key={t} theme={t} />)}</td>
                  <td className="col-text"><span className="text-preview">{highlightText(e.essay_preview, debouncedSearch)}</span></td>
                </tr>
              ]
              if (isExpanded) {
                rows.push(
                  <tr key={`${e.given_id}-detail`} className="detail-row">
                    <td colSpan={8}>
                      <div className="essay-detail">
                        <div className="detail-meta">
                          <span><strong>Alma ID:</strong> {e.metadata?.alma_id || '--'}</span>
                          <span><strong>Semester:</strong> {e.semester}</span>
                          <span><strong>Section:</strong> {e.metadata?.section || '--'}</span>
                          <span><strong>Type:</strong> {e.metadata?.type || '--'}</span>
                          <span><strong>Source:</strong> {e.data_source}</span>
                          <span><strong>Essay #:</strong> {e.metadata?.essay_number || '--'}</span>
                        </div>
                        <div className="detail-themes">
                          <strong>Themes: </strong>
                          {themes.length > 0 ? themes.map(t => <ThemeDot key={t} theme={t} showLabel />) : <span className="muted-text">class_0 only (no theme identified)</span>}
                        </div>
                        <div className="detail-essay-text">
                          <h4>Full Essay Text</h4>
                          <p>{e.sentences ? e.sentences.map(s => s.text).join('. ') + '.' : e.essay_preview}</p>
                        </div>
                        <div className="sentence-list">
                          <h4>Sentence-Level Annotations</h4>
                          {(e.sentences || []).map((s, si) => {
                            const hasTheme = s.themes && s.themes.length > 0 && s.themes[0] !== 'class_0'
                            return (
                              <div key={si} className={`sentence-row ${hasTheme ? 'annotated' : ''}`}>
                                <span className="sentence-num">{si + 1}</span>
                                <span className="sentence-text">{highlightText(s.text, debouncedSearch)}</span>
                                <span className="sentence-themes">
                                  {hasTheme
                                    ? s.themes.map(t => <ThemeDot key={t} theme={t} showLabel />)
                                    : <span className="muted-text">class_0</span>
                                  }
                                </span>
                              </div>
                            )
                          })}
                        </div>
                      </div>
                    </td>
                  </tr>
                )
              }
              return rows
            })}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="pagination">
          <button onClick={() => setCurrentPage(1)} disabled={currentPage === 1}>First</button>
          <button onClick={() => setCurrentPage(p => Math.max(1, p - 1))} disabled={currentPage === 1}>Prev</button>
          <span className="page-info">
            <input type="number" value={currentPage}
              onChange={e => { const p = parseInt(e.target.value); if (p >= 1 && p <= totalPages) setCurrentPage(p) }}
              min={1} max={totalPages} />
            <span>of {totalPages}</span>
          </span>
          <button onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))} disabled={currentPage === totalPages}>Next</button>
          <button onClick={() => setCurrentPage(totalPages)} disabled={currentPage === totalPages}>Last</button>
        </div>
      )}
    </div>
  )
}
