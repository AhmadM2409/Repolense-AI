import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { apiClient } from './api'
import { checked, validateInput, validateAssessment, validateRoadmap, careerError } from './careerContracts'
import { assessment, copy, ids, input, roadmap } from '../test/careerFixtures'
import { CAREER_CONSENT } from '../types/career'
const response=(data:unknown,status=200)=>new Response(JSON.stringify(data),{status})
let fetcher:ReturnType<typeof vi.fn>
beforeEach(()=>{localStorage.setItem('token','synthetic-session');fetcher=vi.fn();vi.stubGlobal('fetch',fetcher)})
afterEach(()=>{vi.unstubAllGlobals();localStorage.clear()})
describe('Career API contracts — synthetic only',()=>{
 it('validates four classes, multi-repository citations and versioned roadmap',()=>{expect(validateInput(input).id).toBe(ids.input);const a=validateAssessment(assessment(4),input,ids.assessment);expect(a.matches.map(m=>m.classification)).toEqual(['demonstrated','documented/claimed','not evidenced','partially demonstrated']);expect(validateRoadmap(roadmap(a),a).result?.recommendations).toHaveLength(3)})
 it.each(['input','revision','repository','sha','path','range','source','quote','class','requirement','jobquote','version'])('rejects invalid %s',kind=>{
  const a=copy(assessment(4)),c=a.matches[0].citations[0]
  if(kind==='input')a.input_id=ids.profile
  if(kind==='revision')c.revision_id=ids.rev2
  if(kind==='repository')c.repository_id=ids.repo2
  if(kind==='sha')c.commit_sha='f'.repeat(40)
  if(kind==='path')c.path='../outside.py'
  if(kind==='range')c.line_end=0
  if(kind==='source')c.source_id='invalid'
  if(kind==='quote')c.quote=''
  if(kind==='class')a.matches[0].classification='invented' as never
  if(kind==='requirement')a.matches[0].requirement_id='Rffffffffffff'
  if(kind==='jobquote')a.requirements[0].quote='Invented job requirement.'
  if(kind==='version')a.policy_version='fake' as never
  expect(()=>checked(()=>validateAssessment(a,input))).toThrow('Invalid career response')
 })
 it('uses Python Unicode code-point offsets',()=>{const i=copy(input),a=copy(assessment());i.job_description='🚀 '+i.job_description;for(const r of a.requirements){r.start+=2;r.end+=2}expect(validateAssessment(a,i).requirements[0].quote).toBe('Python coding required.')})
 it.each(['requirement','observation','project','incomplete','effort','proposal','scope'])('rejects invalid roadmap %s',kind=>{
  const a=assessment(4),r=copy(roadmap(a)),item=r.result!.recommendations[0]
  if(kind==='requirement')item.requirement.quote='Wrong requirement.'
  if(kind==='observation')item.observation.explanation='Invented capability.'
  if(kind==='project')item.project={...a.manifest.revisions[0]}
  if(kind==='incomplete')r.result!.recommendations=[]
  if(kind==='effort')item.effort='three hours' as never
  if(kind==='proposal')item.proposal_notice='Implemented' as never
  if(kind==='scope')r.manifest.input_id=ids.profile
  expect(()=>checked(()=>validateRoadmap(r,a))).toThrow()
 })
 it('sends bearer auth, exact consent and requirement-ID-bound bodies',async()=>{
  fetcher.mockResolvedValueOnce(response(assessment()));await apiClient.extractCareer(input,ids.profile,'keyword')
  expect(fetcher.mock.calls[0][0]).toBe(`/api/career-inputs/${ids.input}/assessments`)
  expect(fetcher.mock.calls[0][1].headers.Authorization).toBe('Bearer synthetic-session')
  expect(JSON.parse(fetcher.mock.calls[0][1].body)).toEqual({request_id:ids.profile,retrieval_mode:'keyword',hosted_processing_consent:CAREER_CONSENT})
  fetcher.mockResolvedValueOnce(response(assessment(1)));await apiClient.matchCareer(input,assessment(),assessment().requirements[0].id)
  expect(JSON.parse(fetcher.mock.calls[1][1].body)).toEqual({requirement_id:assessment().requirements[0].id,reviewed:true,hosted_processing_consent:CAREER_CONSENT})
 })
 it('retains saved failure IDs on 502 and 503',async()=>{
  const a=copy(assessment());a.status='failed';a.error_category='validation_failure';fetcher.mockResolvedValueOnce(response(a,502));expect((await apiClient.extractCareer(input,ids.profile,'keyword')).status).toBe('failed')
  const r=roadmap(assessment(4));r.status='failed';r.result=null;fetcher.mockResolvedValueOnce(response(r,503));expect((await apiClient.generateCareerRoadmap(assessment(4))).status).toBe('failed')
 })
 it('distinguishes missing roadmap from revoked ownership',async()=>{fetcher.mockResolvedValueOnce(response({detail:'Career roadmap unavailable'},404));expect(await apiClient.careerRoadmap(assessment(4))).toBeNull();fetcher.mockResolvedValueOnce(response({detail:'Career assessment unavailable'},404));await expect(apiClient.careerRoadmap(assessment(4))).rejects.toMatchObject({status:404})})
 it('never fetches an aborted request or automatically retries a POST',async()=>{const controller=new AbortController();controller.abort();await expect(apiClient.prepareCareer(input,controller.signal)).rejects.toThrow();expect(fetcher).not.toHaveBeenCalled();fetcher.mockRejectedValueOnce(new TypeError('offline'));await expect(apiClient.extractCareer(input,ids.profile,'keyword')).rejects.toThrow();expect(fetcher).toHaveBeenCalledTimes(1)})
 it.each([401,404,409,422,502,503,0])('sanitizes errors %s',status=>{expect(careerError({status,message:'raw private provider payload'})).not.toContain('raw private provider payload')})
})
