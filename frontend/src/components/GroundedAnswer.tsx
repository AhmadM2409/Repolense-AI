import type { RepositoryAnswer } from '../types'
import { outcomeLabel } from '../services/aiContracts'

export function GroundedAnswer({ answer }: { answer: RepositoryAnswer }) {
  if (answer.status !== 'answered') return <div role="status" className="rounded-xl border border-amber-200 bg-amber-50 p-5 text-amber-950"><h3 className="font-semibold">{outcomeLabel(answer.status)}</h3><p className="mt-2">{answer.status === 'insufficient_evidence' ? answer.uncertainty || 'The selected revision did not provide sufficient evidence.' : 'No validated answer is available. The candidate answer has been withheld.'}</p></div>
  if (!answer.citations.length) return <p role="alert">The answer has no canonical evidence and cannot be displayed.</p>
  return <section aria-label="Grounded answer" className="space-y-5">
    <div className="rounded-xl border border-blue-100 bg-blue-50/50 p-5"><p className="text-xs font-semibold uppercase tracking-widest text-blue-700">AI explanation · validated source references</p><p className="mt-3 whitespace-pre-wrap leading-7 text-slate-900">{answer.answer}</p><p className="mt-4 text-xs text-slate-500">{answer.model} · Generated explanation, separate from deterministic findings.</p></div>
    <div><h3 className="font-semibold text-slate-900">Canonical evidence <span className="text-slate-500">({answer.citations.length})</span></h3><p className="mb-3 text-sm text-slate-500">Open a reference to inspect the exact source excerpt.</p>
      {answer.citations.map(c => <details key={`${c.chunk_id}/${c.line_start}/${c.line_end}`} className="mb-2 overflow-hidden rounded-xl border border-slate-200 bg-white"><summary className="cursor-pointer break-all px-4 py-3 text-sm font-medium text-blue-800 focus-visible:outline-blue-600">{c.path}:{c.line_start}–{c.line_end}</summary><div className="border-t border-slate-200"><p className="break-all bg-slate-50 px-4 py-2 font-mono text-xs text-slate-600">Commit {c.commit_sha} · inclusive lines {c.line_start}–{c.line_end}</p><pre className="overflow-x-auto whitespace-pre-wrap break-words bg-slate-950 p-4 text-sm leading-6 text-slate-100"><code>{c.quote}</code></pre><p className="break-all px-4 py-2 font-mono text-xs text-slate-500">Source {c.source_id} · Chunk {c.chunk_id}</p></div></details>)}
    </div>
  </section>
}
