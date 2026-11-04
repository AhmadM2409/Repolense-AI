import { useCallback, useState } from 'react'
import { Link } from 'react-router-dom'
import type { CareerAssessment, CareerRoadmap as Roadmap } from '../types/career'
import { apiClient } from '../services/api'
import { useCareerAction, useCareerRead } from '../hooks/useCareer'
import { CareerObservation } from './CareerEvidence'
import { CareerButton, CareerConsent, CareerError, PollNotice } from './CareerControls'
const polling = (row:Roadmap|null, forced:boolean) => row ? row.status === 'generating' && !row.operation_expired : forced
export function CareerRoadmap({assessment,onChanged}:{assessment:CareerAssessment;onChanged:()=>void}) {
  const action=useCareerAction(), cancellation=useCareerAction()
  const [consent,setConsent]=useState(false)
  const load=useCallback((signal:AbortSignal)=>apiClient.careerRoadmap(assessment,signal),[assessment])
  const read=useCareerRead(load,polling,action.busy), row=read.value
  const spent=Math.max(assessment.inference_spent,row?.inference_spent ?? 0)
  const exhausted=12-spent<2 && assessment.matches.some(m=>m.classification!=='demonstrated')
  return <section className="ai-card mt-6" aria-label="Improvement roadmap"><h2 className="text-xl font-semibold">Personalized improvement roadmap</h2><p className="mt-2 text-sm text-slate-600">Future proposals tied to this job and inspected evidence. Effort is unknown without a verified delivery baseline.</p><p className="mt-2 text-sm">Shared assessment budget: {spent}/12 attempts spent. New roadmap generation needs up to two remaining attempts; owner cap 40, aggregate cap 100, rate cap 10/minute. Server quotas remain authoritative.</p>
    <CareerError message={read.error||action.error||cancellation.error}/><PollNotice paused={read.paused}/>
    <button type="button" className="mt-3 text-sm text-blue-700 underline" onClick={read.refresh}>Refresh saved roadmap status</button>
    {read.loading && !row ? <p role="status">Loading saved roadmap…</p> : null}
    {row ? <div className="mt-4"><p role="status" className="font-medium">Roadmap: {row.status}{row.operation_expired ? ' · operation expired; no automatic resend' : ''}</p>{row.error_category ? <p className="text-sm">Reason: {row.error_category}. No successful roadmap was published.</p> : null}
      {row.status==='generating' ? <CareerButton disabled={cancellation.busy} onClick={()=>void cancellation.run(s=>apiClient.cancelCareerRoadmap(assessment,row.id,s),()=>{read.refresh();onChanged()})}>Cancel roadmap</CareerButton> : null}
      {row.status==='complete' && row.result ? <div className="mt-4 space-y-5"><p className="text-sm text-slate-600">{row.result.coverage_notice}</p>{!row.result.recommendations.length ? <p>No mandatory improvements were identified for the demonstrated requirements.</p> : row.result.recommendations.map(item=><article key={item.requirement_id} className="rounded-xl border border-slate-200 p-4"><p className="text-xs font-semibold uppercase tracking-wide text-blue-700">{item.priority} priority · {item.action_type}</p><h3 className="mt-2 font-semibold">{item.action}</h3><blockquote className="my-3 border-l-2 border-blue-300 pl-3 text-sm">{item.requirement.quote}</blockquote><p className="text-sm">Requirement {item.requirement_id} · {item.priority_rationale}</p><p className="mt-2"><strong>Evidence gap:</strong> {item.gap}</p><p className="mt-2"><strong>Why this job:</strong> {item.rationale}</p><p className="mt-2"><strong>Proof of completion:</strong> {item.completion_proof}</p><p className="mt-2 text-sm"><strong>Effort:</strong> {item.effort}. {item.effort_uncertainty}</p><p className="mt-2 text-sm">{item.proposal_notice}</p>{item.project ? <Link className="mt-2 block text-sm text-blue-700 underline" to={`/workspace?repository=${item.project.repository_id}&revision=${item.project.revision_id}`}>Project to extend: {item.project.full_name}</Link> : <p className="mt-2 text-sm">No existing project was selected without supporting evidence.</p>}<details className="mt-4"><summary className="cursor-pointer text-sm font-medium">Underlying portfolio observation</summary><CareerObservation match={item.observation} manifest={row.manifest}/></details></article>)}</div> : null}
    </div> : null}
    {!read.loading && !read.error && !row ? <div><p className="mt-4 text-sm">No saved roadmap yet. Career processing must be enabled on the server; availability and quota are checked on submission.</p>{exhausted ? <p role="status" className="mt-3 text-amber-800">Insufficient remaining assessment budget for a roadmap. Limits will not be reset or bypassed.</p> : null}<CareerConsent value={consent} onChange={setConsent}/><CareerButton disabled={!consent||exhausted||action.busy} onClick={()=>void action.run(s=>apiClient.generateCareerRoadmap(assessment,s),()=>{setConsent(false);read.refresh();onChanged()})}>{action.busy?'Waiting for roadmap response…':'Generate improvement roadmap'}</CareerButton></div> : null}
    {action.busy ? <p role="status" className="mt-3 text-sm">Request in progress. Status reads do not spend inference. Leaving this page does not cancel server work; refresh saved status to locate an in-flight roadmap.</p> : null}
  </section>
}
