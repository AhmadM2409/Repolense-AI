import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, useLocation } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { JobFitAnalyzerPage } from '../JobFitAnalyzerPage'
import { assessment, copy, ids, input, roadmap } from '../../test/careerFixtures'
import type { CareerAssessment, CareerInput, CareerRoadmap } from '../../types/career'
const response=(data:unknown,status=200)=>new Response(JSON.stringify(data),{status})
let current:CareerInput,a:CareerAssessment|null,road:CareerRoadmap|null,saved:boolean,prepareCount:number,errorStatus:number,pendingPreparation:boolean
let fetcher:ReturnType<typeof vi.fn>
function Location(){return <output aria-label="Current address">{useLocation().search}</output>}
function show(path='/career'){return render(<MemoryRouter initialEntries={[path]}><JobFitAnalyzerPage/><Location/></MemoryRouter>)}
const posts=(suffix:string)=>fetcher.mock.calls.filter(c=>c[0].split('?')[0].endsWith(suffix)&&c[1]?.method==='POST')
async function openReady(){saved=true;show(`/career?input=${ids.input}`);await screen.findByRole('heading',{name:'Extract job requirements'})}
async function openAssessment(count=0){saved=true;a=assessment(count);show(`/career?input=${ids.input}&assessment=${ids.assessment}`);await screen.findByText('Assessment: '+a.status)}
function consent(){fireEvent.click(screen.getByLabelText(/I explicitly consent to Google Gemini/))}
beforeEach(()=>{
 current=copy(input);a=null;road=null;saved=false;prepareCount=0;errorStatus=0;pendingPreparation=false;localStorage.setItem('token','synthetic-session')
 fetcher=vi.fn(async(url:string,options?:RequestInit)=>{
  const path=url.split('?')[0],method=options?.method??'GET'
  if(path==='/api/profiles'&&method==='POST')return response({id:ids.profile})
  if(path==='/api/career-inputs'&&method==='POST'){saved=true;current={...copy(input),selection:[],status:'input_created'};return response(current,201)}
  if(path==='/api/career-inputs')return response({items:saved?[{id:ids.input,profile_id:ids.profile,account:'example',created_at:input.created_at,started:current.status!=='input_created'}]:[],next_offset:null})
  if(path===`/api/career-inputs/${ids.input}`)return response(current)
  if(path.endsWith('/selection')){const names=JSON.parse(options!.body as string).repositories;current={...copy(input),status:'selection_saved',selection:input.selection.filter(r=>names.includes(r.full_name)).map(r=>({...r,status:'not_started',job_id:null,repository_id:null,revision_id:null,commit_sha:null}))};return response(current)}
  if(path.endsWith('/prepare')){prepareCount++;current=prepareCount===1?{...copy(input),status:'preparing',selection:input.selection.map((r,i)=>({...r,status:i===0?(pendingPreparation?'pending':'awaiting_pin'):'not_started',repository_id:null,revision_id:null,commit_sha:null}))}:copy(input);return response(current,202)}
  if(path.endsWith('/assessments')&&method==='POST'){if(errorStatus)return response({detail:'Career hosted processing disabled'},errorStatus);a=assessment();return response(a,201)}
  if(path.endsWith('/assessments'))return response({items:a?[{id:a.id,status:a.status,created_at:a.created_at,inference_spent:a.inference_spent}]:[]})
  if(path===`/api/career-assessments/${ids.assessment}`)return response(a)
  if(path.endsWith('/match')){a=assessment((a?.matches.length??0)+1);return response(a)}
  if(path===`/api/career-assessments/${ids.assessment}/cancel`){a={...a!,status:'cancelled',matches:[]};return response(a)}
  if(path===`/api/career-roadmaps/${ids.roadmap}/cancel`){road={...road!,status:'cancelled',result:null};return response(road)}
  if(path.endsWith('/roadmap')&&method==='POST'){road=roadmap(a!);a={...a!,inference_spent:road.inference_spent};return response(road,201)}
  if(path.endsWith('/roadmap'))return road?response(road):response({detail:'Career roadmap unavailable'},404)
  throw new Error('Unexpected synthetic route '+path)
 });vi.stubGlobal('fetch',fetcher)
})
afterEach(()=>{vi.useRealTimers();vi.unstubAllGlobals();localStorage.clear()})
describe('Job Fit Analyzer — synthetic APIs only',()=>{
 it('connects input, selection, explicit sequential preparation, four matches and roadmap',async()=>{
  show();await screen.findByText('No saved job inputs on this page.')
  fireEvent.change(screen.getByLabelText('GitHub profile URL'),{target:{value:input.github_profile_url}});fireEvent.change(screen.getByLabelText('Pasted job description'),{target:{value:input.job_description}});fireEvent.click(screen.getByRole('button',{name:'Save job and discover repositories'}));await screen.findByText('Portfolio: input_created')
  fireEvent.click(screen.getByLabelText(/example\/alpha/));fireEvent.click(screen.getByLabelText(/example\/beta/));fireEvent.click(screen.getByRole('button',{name:'Save repository selection'}));await screen.findByText('Portfolio: selection_saved')
  fireEvent.click(screen.getByRole('button',{name:'Advance repository preparation'}));await screen.findByText('Portfolio: preparing');expect(prepareCount).toBe(1);expect(screen.getByText(/awaiting_pin/)).toBeInTheDocument()
  fireEvent.click(screen.getByRole('button',{name:'Advance repository preparation'}));await screen.findByRole('heading',{name:'Extract job requirements'});expect(prepareCount).toBe(2)
  expect(screen.getByRole('button',{name:'Extract requirements with Gemini'})).toBeDisabled();consent();fireEvent.click(screen.getByRole('button',{name:'Extract requirements with Gemini'}));await screen.findByText('Assessment: review');expect(posts('/assessments')).toHaveLength(1)
  for(let i=0;i<4;i++){
   expect(posts('/match')).toHaveLength(i);fireEvent.click(screen.getByLabelText(/I reviewed the next requirement/));consent();fireEvent.click(screen.getByRole('button',{name:'Match next reviewed requirement'}));await waitFor(()=>expect(posts('/match')).toHaveLength(i+1));if(i<3)await waitFor(()=>expect(screen.getByLabelText(/I reviewed the next requirement/)).not.toBeChecked())
  }
  await screen.findByText('Assessment: complete');await screen.findByText(/No saved roadmap yet/)
  for(const label of ['demonstrated','documented/claimed','not evidenced','partially demonstrated'])expect(screen.getByText(label,{exact:true})).toBeInTheDocument()
  consent();fireEvent.click(screen.getByRole('button',{name:'Generate improvement roadmap'}));await screen.findByText('Roadmap: complete');expect(posts('/roadmap')).toHaveLength(1);expect(screen.getByText('Proposed kubernetes verification exercise')).toBeInTheDocument();expect(screen.getByLabelText('Current address').textContent).toContain('assessment='+ids.assessment)
  expect(screen.getAllByText(/Project to extend:/)[0].closest('a')?.getAttribute('href')).toContain('/workspace?repository=');fireEvent.click(screen.getAllByText(/README.md:1–2/)[0]);expect(screen.getAllByText(/SQL support is documented/).length).toBeGreaterThan(0)
 })
 it('caps selection at five and excludes forked sources',async()=>{current={...copy(input),selection:[],status:'input_created'};saved=true;show(`/career?input=${ids.input}`);await screen.findByText('Portfolio: input_created');for(const n of ['alpha','beta','gamma','delta','epsilon'])fireEvent.click(screen.getByLabelText(new RegExp('example/'+n)));expect(screen.getByLabelText(/example\/zeta/)).toBeDisabled();expect(screen.getByLabelText(/example\/fork/)).toBeDisabled();expect(posts('/prepare')).toHaveLength(0)})
 it('rejects invalid URL and excessive UTF-8 bytes before saving',async()=>{show();fireEvent.change(screen.getByLabelText('GitHub profile URL'),{target:{value:'https://evil.example/user'}});fireEvent.change(screen.getByLabelText('Pasted job description'),{target:{value:input.job_description}});fireEvent.submit(screen.getByRole('button',{name:'Save job and discover repositories'}).closest('form')!);await screen.findByText(/Enter a standard https/);expect(posts('/profiles')).toHaveLength(0);for(const bad of ['https://github.com:443/example','https://github.com/example?','https://github.com/example#']){fireEvent.change(screen.getByLabelText('GitHub profile URL'),{target:{value:bad}});fireEvent.submit(screen.getByRole('button',{name:'Save job and discover repositories'}).closest('form')!);expect(posts('/profiles')).toHaveLength(0)}fireEvent.change(screen.getByLabelText('GitHub profile URL'),{target:{value:input.github_profile_url}});fireEvent.change(screen.getByLabelText('Pasted job description'),{target:{value:'技'.repeat(501)}});fireEvent.submit(screen.getByRole('button',{name:'Save job and discover repositories'}).closest('form')!);await screen.findByText(/Paste meaningful job text/);expect(posts('/career-inputs')).toHaveLength(0)})
 it('reopens saved assessments with GET only',async()=>{await openAssessment(4);await screen.findByText(/No saved roadmap yet/);expect(fetcher.mock.calls.every(c=>!c[1]?.method||c[1]?.method==='GET')).toBe(true);fireEvent.click(screen.getByRole('button',{name:'Refresh saved assessment'}));await screen.findByText('Assessment: complete');expect(posts('/assessments')).toHaveLength(0);expect(posts('/match')).toHaveLength(0)})
 it('resets consent and review between explicit matching steps',async()=>{await openAssessment();expect(screen.getByRole('button',{name:'Match next reviewed requirement'})).toBeDisabled();consent();expect(screen.getByRole('button',{name:'Match next reviewed requirement'})).toBeDisabled();fireEvent.click(screen.getByLabelText(/I reviewed the next requirement/));fireEvent.click(screen.getByRole('button',{name:'Match next reviewed requirement'}));await screen.findByText('demonstrated',{exact:true});await waitFor(()=>expect(screen.getByLabelText(/I reviewed the next requirement/)).not.toBeChecked());expect(screen.getByLabelText(/I explicitly consent/)).not.toBeChecked();expect(posts('/match')).toHaveLength(1)})
 it('shows provider disabled without an automatic retry',async()=>{await openReady();errorStatus=409;consent();fireEvent.click(screen.getByRole('button',{name:'Extract requirements with Gemini'}));await screen.findByText(/Career Gemini processing is disabled/);expect(posts('/assessments')).toHaveLength(1);expect(screen.getByRole('button',{name:'Extract requirements with Gemini'})).toBeDisabled()})
 it.each(['failed','cancelled','partial'])('renders truthful %s state',async kind=>{saved=true;if(kind==='partial'){current={...copy(input),status:'partial',selection:input.selection.map((r,i)=>i?{...r,status:'failed'}:r)};show(`/career?input=${ids.input}`);await screen.findByText(/Preparation is incomplete/);expect(screen.queryByRole('button',{name:'Extract requirements with Gemini'})).toBeNull()}else{a={...assessment(),status:kind as 'failed'|'cancelled',error_category:kind==='failed'?'validation_failure':null};show(`/career?input=${ids.input}&assessment=${ids.assessment}`);await screen.findByText('Assessment: '+kind);expect(screen.queryByRole('button',{name:'Match next reviewed requirement'})).toBeNull()}})
 it('handles empty public discovery',async()=>{current={...copy(input),selection:[],status:'input_created',discovery:{...input.discovery,items:[],status:'complete',reason:null}};saved=true;show(`/career?input=${ids.input}`);await screen.findByText(/No public repositories were returned/);expect(screen.getByRole('button',{name:'Save repository selection'})).toBeDisabled()})
 it('does not resend expired operations',async()=>{a={...assessment(),status:'matching',operation_expired:true};saved=true;show(`/career?input=${ids.input}&assessment=${ids.assessment}`);await screen.findByText(/operation expired; no automatic resend/);expect(screen.queryByRole('button',{name:'Match next reviewed requirement'})).toBeNull();expect(posts('/match')).toHaveLength(0)})
 it('cancels only by explicit action',async()=>{await openAssessment();expect(posts('/cancel')).toHaveLength(0);fireEvent.click(screen.getByRole('button',{name:'Cancel assessment'}));await screen.findByText('Assessment: cancelled');expect(posts('/cancel')).toHaveLength(1)})
 it('blocks roadmap generation when the shared assessment budget is exhausted',async()=>{saved=true;a={...assessment(4),inference_spent:11};show(`/career?input=${ids.input}&assessment=${ids.assessment}`);await screen.findByText(/Insufficient remaining assessment budget/);consent();expect(screen.getByRole('button',{name:'Generate improvement roadmap'})).toBeDisabled();expect(posts('/roadmap')).toHaveLength(0)})
 it('keeps the two reserved roadmap attempts after ten matching attempts',async()=>{saved=true;a={...assessment(4),inference_spent:10};show(`/career?input=${ids.input}&assessment=${ids.assessment}`);await screen.findByText(/10\/12 attempts spent/);expect(screen.queryByText(/Insufficient remaining assessment budget/)).toBeNull();expect(posts('/roadmap')).toHaveLength(0)})
 it.each([401,404])('hides unauthorized saved scope %s',async status=>{const ordinary=fetcher.getMockImplementation() as (url:string, options?:RequestInit)=>Promise<Response>;fetcher.mockImplementation((url,options)=>url.split('?')[0]===`/api/career-inputs/${ids.input}`?Promise.resolve(response({detail:'raw private message'},status)):ordinary(url,options));saved=true;show(`/career?input=${ids.input}&assessment=${ids.assessment}`);await screen.findByText(status===401?/Your session expired/:/This saved job, assessment or portfolio is unavailable/);expect(screen.queryByText(input.job_description)).toBeNull();expect(screen.queryByRole('button',{name:'Match next reviewed requirement'})).toBeNull()})

 it('prevents duplicate extraction clicks and never retries an unknown outcome',async()=>{
  await openReady();const ordinary=fetcher.getMockImplementation() as (url:string,options?:RequestInit)=>Promise<Response>
  let reject!:(error:Error)=>void
  fetcher.mockImplementation((url,options)=>url.endsWith('/assessments')&&options?.method==='POST'?new Promise<Response>((_,r)=>{reject=r}):ordinary(url,options))
  consent();const button=screen.getByRole('button',{name:'Extract requirements with Gemini'});fireEvent.click(button);fireEvent.click(button)
  expect(posts('/assessments')).toHaveLength(1)
  await act(async()=>{reject(new Error('connection lost'))})
  await screen.findByText(/Connection interrupted; the outcome may be unknown/)
  expect(posts('/assessments')).toHaveLength(1);expect(screen.getByRole('button',{name:'Extract requirements with Gemini'})).toBeDisabled()
 })
 it('resets extraction consent when the retrieval mode changes',async()=>{
  await openReady();consent();fireEvent.change(screen.getByLabelText('Retrieval mode'),{target:{value:'hybrid'}})
  expect(screen.getByLabelText(/I explicitly consent/)).not.toBeChecked();expect(screen.getByRole('button',{name:'Extract requirements with Gemini'})).toBeDisabled()
 })
 it('shows genuine pending preparation without advancing automatically',async()=>{
  saved=true;current={...copy(input),status:'preparing',selection:input.selection.map(r=>({...r,status:'pending',revision_id:null,repository_id:null,commit_sha:null}))};show(`/career?input=${ids.input}`)
  await screen.findByText('Portfolio: preparing');expect(screen.getByRole('button',{name:'Advance repository preparation'})).toBeDisabled();expect(posts('/prepare')).toHaveLength(0)
 })
 it.each(['failed','cancelled'])('withholds proposals from a %s roadmap',async kind=>{
  saved=true;a=assessment(4);road={...roadmap(a),status:kind as 'failed'|'cancelled',error_category:kind==='failed'?'request_budget':null,result:null};show(`/career?input=${ids.input}&assessment=${ids.assessment}`)
  await screen.findByText('Roadmap: '+kind);expect(screen.queryByText('Proposed kubernetes verification exercise')).toBeNull();expect(screen.queryByRole('button',{name:'Generate improvement roadmap'})).toBeNull()
 })
 it('cancels an in-flight roadmap explicitly',async()=>{
  saved=true;a=assessment(4);road={...roadmap(a),status:'generating',inference_spent:9,roadmap_inference_spent:1,result:null};show(`/career?input=${ids.input}&assessment=${ids.assessment}`)
  await screen.findByText('Roadmap: generating');expect(posts('/cancel')).toHaveLength(0);fireEvent.click(screen.getByRole('button',{name:'Cancel roadmap'}));await screen.findByText('Roadmap: cancelled');expect(posts('/cancel')).toHaveLength(1)
 })

 it('never prepares a different saved selection while checkbox changes are unsaved',async()=>{
  saved=true;current={...copy(input),status:'selection_saved',selection:input.selection.map(r=>({...r,status:'not_started',job_id:null,repository_id:null,revision_id:null,commit_sha:null}))};show(`/career?input=${ids.input}`)
  await screen.findByText('Portfolio: selection_saved');expect(screen.getByRole('button',{name:'Advance repository preparation'})).toBeEnabled()
  fireEvent.click(screen.getByLabelText(/example\/beta/));expect(screen.getByRole('button',{name:'Advance repository preparation'})).toBeDisabled()
  fireEvent.click(screen.getByRole('button',{name:'Save repository selection'}));await waitFor(()=>expect(screen.getByRole('button',{name:'Advance repository preparation'})).toBeEnabled());expect(posts('/prepare')).toHaveLength(0)
 })

 it('refreshes saved summaries with GET after a known matching mutation',async()=>{
  await openAssessment();await screen.findByRole('button',{name:/Open assessment.*review.*2\/12 attempts/})
  fireEvent.click(screen.getByLabelText(/I reviewed the next requirement/));consent();fireEvent.click(screen.getByRole('button',{name:'Match next reviewed requirement'}))
  await screen.findByRole('button',{name:/Open assessment.*review.*4\/12 attempts/})
  expect(posts('/match')).toHaveLength(1);expect(posts('/assessments')).toHaveLength(0)
 })
 it('withholds foreign citation scope',async()=>{a=copy(assessment(4));a.matches[0].citations[0].revision_id=ids.rev2;saved=true;show(`/career?input=${ids.input}&assessment=${ids.assessment}`);await screen.findByText(/response failed validation/);expect(screen.queryByText('The source defines a Python function.')).toBeNull()})
})


describe('Saved demo presentation',()=>{
 it('collapses a frozen discovery list while keeping readiness and revisions visible',async()=>{
  await openReady()
  const summary=screen.getByText(/Review frozen selection/)
  expect(summary.closest('details')).not.toHaveAttribute('open')
  expect(screen.getAllByText(/Verified commit/).length).toBeGreaterThan(0)
  fireEvent.click(summary)
  expect(screen.getByLabelText(/example\/alpha/)).toBeDisabled()
  expect(posts('/prepare')).toHaveLength(0)
 })
 it('explains a saved-demo mutation denial without retrying',async()=>{
  await openReady();errorStatus=423;consent()
  fireEvent.click(screen.getByRole('button',{name:'Extract requirements with Gemini'}))
  await screen.findByText(/Saved demo is read-only/)
  expect(posts('/assessments')).toHaveLength(1)
  expect(posts('/match')).toHaveLength(0)
 })
})

it('preserves a read-only profile-creation denial and sends no discovery request',async()=>{
 const ordinary=fetcher.getMockImplementation() as (url:string,options?:RequestInit)=>Promise<Response>
 fetcher.mockImplementation((url,options)=>url==='/api/profiles'&&options?.method==='POST'?Promise.resolve(response({detail:'Saved demo is read-only'},423)):ordinary(url,options))
 show('/career')
 fireEvent.change(screen.getByLabelText('GitHub profile URL'),{target:{value:'https://github.com/example'}})
 fireEvent.change(screen.getByLabelText('Pasted job description'),{target:{value:'Write and test Python code.'}})
 fireEvent.click(screen.getByRole('button',{name:'Save job and discover repositories'}))
 await screen.findByText(/Saved demo is read-only/)
 expect(posts('/profiles')).toHaveLength(1)
 expect(posts('/career-inputs')).toHaveLength(0)
 expect(screen.getByRole('button',{name:'Save job and discover repositories'})).toBeDisabled()
})
