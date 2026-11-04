import { useCallback, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { GitBranch, Search, BookOpen, Layers } from 'lucide-react'
import { apiClient } from '../services/api'
import { usePage } from '../hooks/usePage'
import { outcomeLabel } from '../services/aiContracts'
import { RepositoryQuestions } from '../components/RepositoryQuestions'
import { InvestigationComposer } from '../components/InvestigationComposer'
import { InvestigationRun } from '../components/InvestigationRun'

function RevisionExperience({ repo, rev, id, onOpen }: { repo: string; rev: string; id: string; onOpen: (id: string) => void }) {
  const [mode, setMode] = useState<'ask' | 'investigate'>(id ? 'investigate' : 'ask')
  const loadHistory = useCallback((offset: number, signal: AbortSignal) => apiClient.investigationHistory(repo, rev, offset, signal), [repo, rev])
  const history = usePage(loadHistory)
  const [openId, setOpenId] = useState('')
  function open(value: string) { setMode('investigate'); onOpen(value) }
  const investigating = mode === 'investigate' || !!id
  return <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_320px]">
    <section className="min-w-0 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-7">
      <div className="mb-6 flex gap-2 border-b border-slate-200 pb-4" aria-label="AI experience mode"><button className={!investigating ? 'ai-primary' : 'ai-secondary'} aria-pressed={!investigating} onClick={() => { setMode('ask'); onOpen('') }}><Search size={16} aria-hidden="true" /> Ask</button><button className={investigating ? 'ai-primary' : 'ai-secondary'} aria-pressed={investigating} onClick={() => setMode('investigate')}><Layers size={16} aria-hidden="true" /> Investigate</button></div>
      {!investigating ? <RepositoryQuestions key={repo+'/'+rev} repositoryId={repo} revisionId={rev} /> : <div className="space-y-6"><InvestigationComposer key={repo+'/'+rev} repo={repo} rev={rev} onStarted={open} refreshHistory={history.reload} />{id ? <InvestigationRun key={id} repo={repo} rev={rev} id={id} onTerminal={history.reload} /> : null}
        <form onSubmit={event => { event.preventDefault(); if (/^[a-f0-9-]{36}$/i.test(openId.trim())) open(openId.trim()) }} className="border-t border-slate-200 pt-4"><label className="block text-sm font-medium text-slate-700">Investigation ID<input className="ai-input mt-1" value={openId} onChange={event => setOpenId(event.target.value)} placeholder="Open a saved execution by identifier" /></label><button className="ai-secondary mt-2" disabled={!/^[a-f0-9-]{36}$/i.test(openId.trim())}>Open investigation</button></form>
      </div>}
    </section>
    <aside className="min-w-0"><div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm"><div className="flex items-center justify-between gap-2"><h2 className="flex items-center gap-2 font-semibold"><BookOpen size={17} aria-hidden="true" /> Saved investigations</h2><button className="text-sm font-medium text-blue-700 hover:underline focus-visible:outline-blue-600" onClick={history.reload} disabled={history.loading}>Refresh history</button></div><p className="mt-2 text-xs text-slate-500">Only this repository and immutable revision. Results are stored on the server.</p>
      {history.error ? <p role="alert" className="ai-error mt-3">{history.error}</p> : null}
      {history.loading && !history.items.length ? <p role="status" className="py-6 text-sm text-slate-500">Loading saved investigations…</p> : !history.items.length && !history.error ? <p className="py-6 text-sm text-slate-500">No saved investigations for this revision yet.</p> : null}
      <ul className="mt-3 space-y-3">{history.items.map(item => <li key={item.id}><button className="w-full rounded-xl border border-slate-200 p-3 text-left hover:border-blue-400 hover:bg-blue-50 focus-visible:outline-blue-600" onClick={() => open(item.id)}><p className="break-words text-sm font-medium text-slate-900">{item.objective}</p><p className="mt-1 text-xs font-medium text-blue-700">{outcomeLabel(item.outcome || item.status)}</p><p className="mt-2 break-all font-mono text-xs text-slate-500">{item.id}</p><p className="mt-1 text-xs text-slate-500">Created {new Date(item.created_at).toLocaleString()}</p>{item.completed_at ? <p className="text-xs text-slate-500">Finished {new Date(item.completed_at).toLocaleString()}</p> : null}</button></li>)}</ul>
      {history.next !== null ? <button className="ai-secondary mt-4" onClick={history.more} disabled={history.loading}>Load older investigations</button> : null}
    </div></aside>
  </div>
}

export function RepositoryWorkspacePage() {
  const [params, setParams] = useSearchParams()
  const repo = params.get('repository') || '', rev = params.get('revision') || '', id = params.get('investigation') || ''
  const loadRepositories = useCallback((offset: number, signal: AbortSignal) => apiClient.repositories(offset, signal), [])
  const repositories = usePage(loadRepositories)
  const selected = repositories.items.find(item => item.id === repo)
  const loadRevisions = useCallback((offset: number, signal: AbortSignal) => apiClient.revisions(repo, offset, signal), [repo])
  const revisions = usePage(loadRevisions, !!selected)
  const revision = revisions.items.find(item => item.id === rev && item.repository_id === repo)
  function onOpen(value: string) { setParams(old => { const next = new URLSearchParams(old); if (value) next.set('investigation', value); else next.delete('investigation'); return next }) }
  return <main className="min-h-[calc(100vh-4rem)] bg-slate-50"><div className="mx-auto max-w-7xl space-y-6 px-4 py-8 sm:px-6 lg:px-8">
    <header className="flex flex-wrap items-start justify-between gap-4"><div><p className="text-xs font-semibold uppercase tracking-[0.2em] text-blue-700">RepoLens AI / repository intelligence</p><h1 className="mt-2 text-3xl font-bold tracking-tight text-slate-950">Explore the evidence.</h1><p className="mt-2 max-w-2xl text-slate-600">Ask a focused question or follow an investigation across sources, anchored to one immutable commit.</p></div><Link className="ai-secondary" to="/repositories/new">Ingest a repository</Link></header>
    <section aria-label="Repository context" className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm"><div className="grid gap-4 md:grid-cols-2"><label className="block text-sm font-medium text-slate-700">Repository<select className="ai-input mt-1" value={selected ? repo : ''} disabled={repositories.loading && !repositories.items.length} onChange={event => setParams(event.target.value ? { repository: event.target.value } : {})}><option value="">{repositories.loading ? 'Loading repositories…' : 'Select an authorized repository'}</option>{repositories.items.map(item => <option key={item.id} value={item.id}>{item.full_name}</option>)}</select></label><label className="block text-sm font-medium text-slate-700">Immutable revision<select className="ai-input mt-1 font-mono" value={revision ? rev : ''} disabled={!selected || revisions.loading && !revisions.items.length} onChange={event => setParams({ repository: repo, ...(event.target.value ? { revision: event.target.value } : {}) })}><option value="">{revisions.loading ? 'Loading revisions…' : 'Select a commit revision'}</option>{revisions.items.map(item => <option key={item.id} value={item.id}>{item.commit_sha}</option>)}</select></label></div>
      {repositories.error || revisions.error ? <p role="alert" className="ai-error mt-3">{repositories.error || revisions.error}</p> : null}
      <div className="mt-3 flex flex-wrap gap-3">{repositories.next !== null ? <button className="text-sm text-blue-700 underline" disabled={repositories.loading} onClick={repositories.more}>More repositories</button> : null}{revisions.next !== null ? <button className="text-sm text-blue-700 underline" disabled={revisions.loading} onClick={revisions.more}>More revisions</button> : null}<button className="text-sm text-blue-700 underline" disabled={repositories.loading} onClick={repositories.reload}>Refresh catalog</button></div>
      {revision ? <p className="mt-4 flex items-start gap-2 rounded-lg bg-slate-100 p-3 font-mono text-xs text-slate-600"><GitBranch size={16} className="shrink-0" aria-hidden="true" /><span className="break-all">Active commit {revision.commit_sha}</span></p> : null}
    </section>
    {selected && revision ? <RevisionExperience key={repo+'/'+rev} repo={repo} rev={rev} id={id} onOpen={onOpen} /> : <section className="rounded-2xl border border-dashed border-slate-300 bg-white px-6 py-14 text-center"><Search className="mx-auto text-slate-400" size={30} aria-hidden="true" /><h2 className="mt-4 text-lg font-semibold text-slate-900">{!repositories.loading && !repositories.items.length && !repositories.error ? 'Your repository workspace starts here' : selected && !revisions.loading && !revisions.items.length ? 'No available revision yet' : 'Choose the source of truth'}</h2><p className="mx-auto mt-2 max-w-lg text-sm text-slate-500">{selected && !revisions.loading && !revisions.items.length ? 'Complete repository ingestion to make an immutable revision available. Keyword Q&A does not require a semantic index.' : 'Select an authorized repository and revision to ask questions, inspect canonical evidence, or reopen saved investigations.'}</p></section>}
  </div></main>
}
