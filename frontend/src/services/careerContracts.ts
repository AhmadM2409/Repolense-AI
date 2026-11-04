import type { CareerInput, CareerAssessment, CareerCitation, CareerRoadmap, CareerInputSummary, AssessmentSummary, Manifest } from '../types/career'
export const uuid = (value: unknown): value is string => typeof value === 'string' && /^[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}$/i.test(value)
const hex = (v: unknown, size: number) => typeof v === 'string' && new RegExp(`^[0-9a-f]{${size}}$`).test(v)
const text = (v: unknown): v is string => typeof v === 'string'
const integer = (v: number, min: number, max: number) => Number.isInteger(v) && v >= min && v <= max
const classes = ['demonstrated', 'partially demonstrated', 'documented/claimed', 'not evidenced']
function invalid(): never { throw Object.assign(new Error('Invalid career response'), { status: 502 }) }
export function checked<T>(read: () => T): T { try { return read() } catch { return invalid() } }
export function equal(a: unknown, b: unknown): boolean {
  if (a === b) return true
  if (!a || !b || typeof a !== 'object' || typeof b !== 'object') return false
  if (Array.isArray(a) || Array.isArray(b)) return Array.isArray(a) && Array.isArray(b) && a.length === b.length && a.every((v, i) => equal(v, b[i]))
  const x = a as Record<string, unknown>, y = b as Record<string, unknown>
  return Object.keys(x).length === Object.keys(y).length && Object.keys(x).every(k => equal(x[k], y[k]))
}
export function validateInput(raw: unknown, id?: string, profile?: string): CareerInput {
  const d = raw as CareerInput
  if (!d || !uuid(d.id) || id && d.id !== id || !uuid(d.profile_id) || profile && d.profile_id !== profile || !hex(d.job_hash, 64) || !text(d.job_description) || d.manifest_version !== 1 || !['input_created','selection_saved','preparing','ready','partial','failed'].includes(d.status) || !d.discovery || !Array.isArray(d.discovery.items) || d.discovery.items.length > 120 || !['complete','bounded','partial','unavailable'].includes(d.discovery.status) || d.discovery.private_coverage !== 'not_accessed_or_enumerated' || !Array.isArray(d.selection) || d.selection.length > 10) invalid()
  for (const r of d.discovery.items) if (!r || !text(r.full_name) || typeof r.eligible !== 'boolean') invalid()
  if (new Set(d.discovery.items.map(r => r.full_name)).size !== d.discovery.items.length || new Set(d.selection.map(r => r.full_name)).size !== d.selection.length) invalid()
  for (const r of d.selection) {
    if (!d.discovery.items.some(x => x.full_name === r.full_name && x.eligible) || !['not_started','pending','processing','awaiting_pin','ready','failed','cancelled'].includes(r.status)) invalid()
    if (r.status === 'ready' && (!uuid(r.repository_id) || !uuid(r.revision_id) || !hex(r.commit_sha, 40))) invalid()
  }
  if (d.status === 'ready' && (!d.selection.length || d.selection.some(r => r.status !== 'ready'))) invalid()
  return d
}
export function validateCitation(c: CareerCitation, manifest: Manifest): void {
  if (!c || !uuid(c.repository_id) || !uuid(c.revision_id) || !uuid(c.source_id) || !uuid(c.chunk_id) || !hex(c.commit_sha,40) || !manifest.revisions.some(r => r.repository_id === c.repository_id && r.revision_id === c.revision_id && r.commit_sha === c.commit_sha) || !text(c.path) || !c.path || c.path.includes('\\') || c.path.startsWith('/') || c.path.split('/').some(p => !p || p === '..' || p === '.') || c.path.includes(':') || !integer(c.line_start,1,1e8) || !integer(c.line_end,c.line_start,1e8) || !text(c.quote) || !c.quote.trim() || !['code','test','configuration','documentation'].includes(c.evidence_kind)) invalid()
}
export function validateAssessment(raw: unknown, input: CareerInput, id?: string): CareerAssessment {
  const d = raw as CareerAssessment
  if (!d || !uuid(d.id) || id && d.id !== id || d.input_id !== input.id || !d.manifest || d.manifest.input_id !== input.id || d.manifest.job_hash !== input.job_hash || d.manifest.version !== 1 || !Array.isArray(d.manifest.revisions) || !integer(d.manifest.revisions.length,1,5) || !['extracting','review','matching','complete','failed','cancelled'].includes(d.status) || d.policy_version !== 'career-evidence-v1' || d.model !== 'gemini-3.5-flash-lite' || !['keyword','semantic','hybrid'].includes(d.retrieval_mode) || typeof d.operation_expired !== 'boolean' || !integer(d.inference_spent,0,12) || !Array.isArray(d.requirements) || d.requirements.length > 4 || !Array.isArray(d.matches) || d.matches.length > d.requirements.length || !text(d.coverage_notice)) invalid()
  if (input.selection.length !== d.manifest.revisions.length || new Set(d.manifest.revisions.map(r => r.repository_id)).size !== d.manifest.revisions.length) invalid()
  for (const r of d.manifest.revisions) if (!input.selection.some(s => s.status === 'ready' && s.repository_id === r.repository_id && s.revision_id === r.revision_id && s.commit_sha === r.commit_sha && s.full_name === r.full_name)) invalid()
  if (new Set(d.requirements.map(r => r.id)).size !== d.requirements.length) invalid()
  for (const r of d.requirements) if (!r || !/^R[0-9a-f]{12}$/.test(r.id) || !text(r.quote) || !text(r.skill) || !Array.isArray(r.aliases) || r.aliases.length > 3 || r.aliases.some(a => !text(a)) || !text(r.depth) || !['required','preferred'].includes(r.priority) || !integer(r.start,0,Array.from(input.job_description).length) || !integer(r.end,r.start+1,Array.from(input.job_description).length) || Array.from(input.job_description).slice(r.start,r.end).join('') !== r.quote || r.job_hash !== input.job_hash) invalid()
  for (const [i,m] of d.matches.entries()) {
    if (!m || m.requirement_id !== d.requirements[i].id || !classes.includes(m.classification) || !text(m.explanation) || !text(m.uncertainty) || !Array.isArray(m.citations) || m.citations.length > 5 || !m.coverage || typeof m.coverage !== 'object' || !['supported','contradicted','insufficient','invalid'].includes(m.support_verdict)) invalid()
    m.citations.forEach(c => validateCitation(c,d.manifest))
    if (m.classification !== 'not evidenced' && !m.citations.length) invalid()
  }
  if (d.status === 'complete' && (!d.requirements.length || d.matches.length !== d.requirements.length) || ['failed','cancelled'].includes(d.status) && d.matches.length) invalid()
  return d
}
export function validateRoadmap(raw: unknown, parent: CareerAssessment): CareerRoadmap {
  const d = raw as CareerRoadmap
  if (!d || !uuid(d.id) || d.assessment_id !== parent.id || d.version !== 1 || !hex(d.source_signature,64) || !equal(d.manifest,parent.manifest) || !['generating','complete','failed','cancelled'].includes(d.status) || !integer(d.inference_spent,parent.inference_spent,12) || !integer(d.roadmap_inference_spent,0,2) || d.policy_version !== 'career-roadmap-v1' || d.model !== 'gemini-3.5-flash-lite' || typeof d.operation_expired !== 'boolean') invalid()
  if (d.status !== 'complete') { if (d.result !== null) invalid(); return d }
  if (!d.result || !Array.isArray(d.result.recommendations) || d.result.recommendations.length > 4 || !text(d.result.coverage_notice)) invalid()
  const expected = parent.matches.filter(m => m.classification !== 'demonstrated').map(m => m.requirement_id)
  const ids = d.result.recommendations.map(r => r.requirement_id)
  if (ids.length !== expected.length || new Set(ids).size !== ids.length || ids.some(id => !expected.includes(id)) || !equal([...d.result.demonstrated_requirement_ids].sort(),parent.matches.filter(m => m.classification === 'demonstrated').map(m => m.requirement_id).sort())) invalid()
  for (const r of d.result.recommendations) {
    const req = parent.requirements.find(x => x.id === r.requirement_id), match = parent.matches.find(x => x.requirement_id === r.requirement_id)
    if (!req || !match || !equal(req,r.requirement) || !equal(match,r.observation) || !['learning','coding','testing','documenting','deployment'].includes(r.action_type) || ![r.action,r.gap,r.rationale,r.completion_proof,r.priority_rationale,r.effort_uncertainty].every(x => text(x) && x.trim()) || r.proposal_notice !== 'Proposed future work; not existing portfolio capability.' || r.effort !== 'unknown' || r.priority !== (req.priority === 'required' ? 'high' : 'medium')) invalid()
    if (r.project_citation === null) { if (r.project !== null) invalid() }
    else {
      if (!integer(r.project_citation,0,match.citations.length-1)) invalid()
      const c = match.citations[r.project_citation]
      if (!equal(r.project,parent.manifest.revisions.find(x => x.repository_id === c.repository_id && x.revision_id === c.revision_id))) invalid()
    }
  }
  return d
}
export function validateInputs(raw: unknown): {items: CareerInputSummary[]; next_offset: number | null} {
  const d = raw as {items: CareerInputSummary[]; next_offset: number | null}
  if (!d || !Array.isArray(d.items) || d.items.length > 20 || d.items.some(r => !uuid(r.id) || !uuid(r.profile_id) || !text(r.account) || !text(r.created_at) || typeof r.started !== 'boolean') || d.next_offset !== null && !integer(d.next_offset,1,10000)) invalid()
  return d
}
export function validateHistory(raw: unknown): {items: AssessmentSummary[]} {
  const d = raw as {items: AssessmentSummary[]}
  if (!d || !Array.isArray(d.items) || d.items.length > 20 || d.items.some(r => !uuid(r.id) || !['extracting','review','matching','complete','failed','cancelled'].includes(r.status) || !integer(r.inference_spent,0,12))) invalid()
  return d
}
export function careerError(error: unknown): string {
  const e = error as {status?:number; message?:string}
  if (e?.status === 423) return 'Saved demo is read-only. Open a saved assessment; new discovery, preparation and AI generation require separate authorization.'
  if (e?.status === 401) return 'Your session expired. Sign in again; saved work has not been resubmitted.'
  if ([403,404].includes(e?.status ?? 0)) return 'This saved job, assessment or portfolio is unavailable to your account.'
  if (e?.status === 409) return /provider|hosted processing/i.test(e.message ?? '') ? 'Career Gemini processing is disabled, unavailable or its policy changed. No automatic retry will occur.' : 'This stage is unavailable, its budget is insufficient, or portfolio evidence changed. Refresh saved state before continuing.'
  if ([413,422].includes(e?.status ?? 0)) return 'The input exceeds a limit or is invalid. Check the GitHub profile, job text and explicit consent.'
  if (e?.status === 502) return 'The response failed validation and has been withheld. Inspect saved status; do not assume completion.'
  if ([429,503].includes(e?.status ?? 0)) return 'The provider, quota or service is unavailable. Inspect saved status before any explicit further action.'
  return 'Connection interrupted; the outcome may be unknown. Refresh saved history before submitting again. Nothing is retried automatically.'
}
