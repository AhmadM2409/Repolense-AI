import { act, cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { RepositoryAnalysisPage } from '../RepositoryAnalysisPage'
const span={source_id:'source',repository_id:'repo',commit_sha:'a'.repeat(40),path:'app.py',line_start:2,line_end:2,quote:'subprocess.run(value, shell=True)'}
const summary={scope:'verified ingested selection only',file_count:2,languages:{Python:1},major_directories:{'(root)':2},
 documentation:{count:1,selected_documents:['README.md']},dependencies:[{name:'flask',declaration:'>=3',ecosystem:'python',section:'requirements',evidence:[span]}],
 framework_indicators:[{name:'Flask',basis:'dependency declaration only'}],test_locations:[],ci_configurations:[],deployment_configurations:[],entrypoint_candidates:[],diagnostics:[]}
const report={review_id:'job',revision_id:'revision',commit_sha:'a'.repeat(40),repository:'example/project',engine_version:'deterministic-v1',producer:'deterministic',summary,
 findings:[{id:'finding',rule_id:'PY-SHELL-TRUE',title:'Subprocess shell enabled',detail:'Review input trust.',severity:'high',category:'security',certainty:'potential',evidence:[span]}]}
const response=(value:unknown,status=200)=>new Response(JSON.stringify(value),{status})
function show() {render(<MemoryRouter initialEntries={['/analyses/job']}><Routes><Route path="/analyses/:jobId" element={<RepositoryAnalysisPage/>}/></Routes></MemoryRouter>)}
beforeEach(()=>{vi.useFakeTimers();localStorage.setItem('token','fixture-user-token')})
afterEach(()=>{cleanup();vi.useRealTimers();vi.unstubAllGlobals();localStorage.clear()})
it('polls and displays observations, severity and exact evidence',async()=>{
 const fetch=vi.fn().mockResolvedValueOnce(response({id:'job',status:'processing'})).mockResolvedValueOnce(response({id:'job',status:'complete'})).mockResolvedValueOnce(response(report))
 vi.stubGlobal('fetch',fetch);show();await act(async()=>{})
 expect(screen.getByRole('status')).toHaveTextContent('processing')
 await act(async()=>{await vi.advanceTimersByTimeAsync(1000)})
 expect(screen.getByText('Python: 1')).toBeInTheDocument()
 expect(screen.getByText(/Potential risk; exploitability/)).toBeInTheDocument()
 expect(screen.getAllByText('subprocess.run(value, shell=True)')).toHaveLength(2)
 expect(screen.getAllByText(/app.py:2/).length).toBeGreaterThan(0)
 expect(fetch.mock.calls[0][1].headers.Authorization).toBe('Bearer fixture-user-token')
})
it('shows empty findings without claiming safety',async()=>{
 vi.stubGlobal('fetch',vi.fn().mockResolvedValueOnce(response({status:'complete'})).mockResolvedValueOnce(response({...report,findings:[]})))
 show();await act(async()=>{})
 expect(screen.getByText(/No supported issues were detected/)).toBeInTheDocument()
 expect(screen.queryByText(/score/i)).toBeNull()
})
it('shows failed and expired jobs without fabricated findings',async()=>{
 vi.stubGlobal('fetch',vi.fn().mockResolvedValue(response({status:'failed',error_message:'Analysis lease expired; submit again'})))
 show();await act(async()=>{})
 expect(screen.getByRole('alert')).toHaveTextContent('lease expired')
 expect(screen.queryByText('Source-backed findings')).toBeNull()
})
it('rejects report integrity failures visibly',async()=>{
 vi.stubGlobal('fetch',vi.fn().mockResolvedValueOnce(response({status:'complete'})).mockResolvedValueOnce(response({detail:'Analysis evidence integrity check failed'},409)))
 show();await act(async()=>{})
 expect(screen.getByRole('alert')).toHaveTextContent('integrity')
 expect(screen.queryByText('Subprocess shell enabled')).toBeNull()
})
it('stops polling on unmount',async()=>{
 const fetch=vi.fn().mockResolvedValue(response({status:'processing'}))
 vi.stubGlobal('fetch',fetch);show();await act(async()=>{});cleanup()
 await act(async()=>{await vi.advanceTimersByTimeAsync(2000)})
 expect(fetch).toHaveBeenCalledTimes(1)
})
it('escapes evidence content',async()=>{
 vi.stubGlobal('fetch',vi.fn().mockResolvedValueOnce(response({status:'complete'})).mockResolvedValueOnce(response({...report,
 findings:[{...report.findings[0],evidence:[{...span,quote:'<script>unsafe()</script>'}]}]})))
 show();await act(async()=>{})
 expect(screen.getByText('<script>unsafe()</script>')).toBeInTheDocument()
 expect(document.querySelector('script')).toBeNull()
})

it('shows queued worker-unavailable state without resubmitting', async () => {
 const fetch=vi.fn().mockResolvedValue(response({id:'job',status:'pending',progress_stage:'queued',attempt_count:0,max_attempts:3}))
 vi.stubGlobal('fetch',fetch);show();await act(async()=>{})
 expect(screen.getByRole('status')).toHaveTextContent('waiting for a worker')
 await act(async()=>{await vi.advanceTimersByTimeAsync(2000)})
 expect(fetch.mock.calls.every(call=>!call[1].method || call[1].method==='GET')).toBe(true)
 cleanup()
})

it('renders retry timing, actual stages and indeterminate progress', async () => {
 vi.stubGlobal('fetch',vi.fn().mockResolvedValue(response({id:'job',status:'pending',progress_stage:'retry_waiting',retry_at:'2026-10-30T22:00:00Z',attempt_count:1,max_attempts:3,progress_pct:null})))
 show();await act(async()=>{})
 expect(screen.getByRole('status')).toHaveTextContent('Retry waiting')
 expect(screen.getByText('Stage: retry_waiting')).toBeInTheDocument()
 expect(screen.getByText(/Next retry/)).toBeInTheDocument()
 expect(screen.queryByRole('progressbar')).toBeNull()
})

it('recovers polling after a temporary API failure', async () => {
 const fetch=vi.fn().mockRejectedValueOnce(new TypeError('Network unavailable'))
  .mockResolvedValueOnce(response({id:'job',status:'processing',progress_stage:'analyzing_sources'}))
  .mockResolvedValueOnce(response({id:'job',status:'complete',progress_pct:100}))
  .mockResolvedValueOnce(response(report))
 vi.stubGlobal('fetch',fetch);show();await act(async()=>{})
 expect(screen.getByRole('alert')).toHaveTextContent('Retrying connection')
 await act(async()=>{await vi.advanceTimersByTimeAsync(1000)})
 expect(screen.getByText('Stage: analyzing_sources')).toBeInTheDocument()
 expect(screen.queryByRole('alert')).toBeNull()
 await act(async()=>{await vi.advanceTimersByTimeAsync(1000)})
 expect(screen.getByText('Python: 1')).toBeInTheDocument()
 const count=fetch.mock.calls.length
 await act(async()=>{await vi.advanceTimersByTimeAsync(3000)})
 expect(fetch).toHaveBeenCalledTimes(count)
})

it('requests authenticated cancellation and stops after persisted cancelled status', async () => {
 const fetch=vi.fn().mockResolvedValueOnce(response({id:'job',status:'processing',progress_stage:'fetching_sources'}))
 .mockResolvedValueOnce(response({id:'job',status:'processing',cancel_requested_at:'2026-10-30T21:00:00Z'}))
 .mockResolvedValueOnce(response({id:'job',status:'cancelled',progress_stage:'cancelled'}))
 vi.stubGlobal('fetch',fetch);show();await act(async()=>{})
 await act(async()=>{screen.getByRole('button',{name:'Cancel job'}).click()})
 const request=fetch.mock.calls.find(call=>call[0]==='/api/jobs/job/cancel')
 expect(request?.[1].method).toBe('POST')
 expect(request?.[1].headers.Authorization).toBe('Bearer fixture-user-token')
 expect(screen.getByText(/Cancellation requested/)).toBeInTheDocument()
 await act(async()=>{await vi.advanceTimersByTimeAsync(1000)})
 expect(screen.getByRole('status')).toHaveTextContent('Cancelled')
 expect(screen.queryByRole('button',{name:'Cancel job'})).toBeNull()
 const count=fetch.mock.calls.length
 await act(async()=>{await vi.advanceTimersByTimeAsync(3000)})
 expect(fetch).toHaveBeenCalledTimes(count)
 expect(screen.queryByText('Source-backed findings')).toBeNull()
})

it('does not overlap pending status requests',async()=>{
 let finish: ((value: Response)=>void) | undefined
 const fetch=vi.fn().mockImplementation(()=>new Promise<Response>(resolve=>{finish=resolve}))
 vi.stubGlobal('fetch',fetch);show();await act(async()=>{})
 await act(async()=>{await vi.advanceTimersByTimeAsync(5000)})
 expect(fetch).toHaveBeenCalledTimes(1)
 cleanup();finish?.(response({id:'job',status:'processing'}));await act(async()=>{})
 expect(fetch).toHaveBeenCalledTimes(1)
})
