import * as career from './careerContracts'
import type { CareerInput, CareerAssessment, RetrievalMode } from '../types/career'
import { CAREER_CONSENT } from '../types/career'
import { validateAnswer, validateInvestigation } from './aiContracts'
import { AuthResponse, Profile, Review, ReviewListResponse, ReviewStatus, IngestionJob, IngestedSourceDetail, AnalysisStatus, AnalysisReport, DurableJob } from '../types'

const API_BASE = '/api'

class ApiClient {
  async cancelJob(id: string, signal?: AbortSignal): Promise<DurableJob> {
    return this.request('/jobs/' + id + '/cancel', { method: 'POST', signal })
  }

  private getAuthHeader(): { Authorization: string } | {} {
    const token = localStorage.getItem('token')
    if (!token) return {}
    return { Authorization: `Bearer ${token}` }
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {},
    acceptCareerFailure = false
  ): Promise<T> {
    options.signal?.throwIfAborted()
    const url = `${API_BASE}${endpoint}`
    const headers = {
      'Content-Type': 'application/json',
      ...this.getAuthHeader(),
      ...options.headers
    }

    const response = await fetch(url, {
      ...options,
      headers
    })

    if (!response.ok) {
      const error = await response.json().catch(() => ({}))
      if (acceptCareerFailure && [502,503].includes(response.status) && error?.status === 'failed') return error as T
      throw Object.assign(new Error(typeof error.detail === "string" ? error.detail : `Request failed with status ${response.status}`), { status: response.status })
    }

    if (response.status === 204) return undefined as T
    return response.json()
  }

  async login(email: string, password: string): Promise<string> {
    const formData = new URLSearchParams()
    formData.append('username', email)
    formData.append('password', password)

    const response = await fetch(`${API_BASE}/auth/login`, {
      method: 'POST',
      body: formData,
      headers: {
        'Accept': 'application/json'
      }
    })

    if (!response.ok) {
      throw new Error('Invalid email or password')
    }

    const data: AuthResponse = await response.json()
    return data.access_token
  }

  async register(email: string, password: string): Promise<string> {
    const response = await fetch(`${API_BASE}/auth/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password })
    })

    if (!response.ok) {
      const error = await response.json().catch(() => ({}))
      throw new Error(error.detail || 'Registration failed')
    }

    const data: AuthResponse = await response.json()
    return data.access_token
  }

  async createProfile(formData: FormData, signal?: AbortSignal): Promise<Profile> {
    signal?.throwIfAborted()
    const url = `${API_BASE}/profiles`
    const headers = this.getAuthHeader()

    const response = await fetch(url, {
      method: 'POST',
      headers,
      signal,
      body: formData
    })

    if (!response.ok) {
      const error = await response.json().catch(() => ({}))
      throw Object.assign(new Error(typeof error.detail === 'string' ? error.detail : 'Failed to create profile'), { status: response.status })
    }

    return response.json()
  }

  async getProfile(id: string): Promise<Profile> {
    return this.request(`/profiles/${id}`)
  }

  async createReview(profileId: string): Promise<Review> {
    return this.request('/reviews', {
      method: 'POST',
      body: JSON.stringify({ profile_id: profileId })
    })
  }

  async getReview(id: string): Promise<Review> {
    return this.request(`/reviews/${id}`)
  }

  async getReviewStatus(id: string): Promise<ReviewStatus> {
    return this.request(`/reviews/${id}/status`)
  }

  async listReviews(page: number = 1, pageSize: number = 10): Promise<ReviewListResponse> {
    return this.request(`/reviews?page=${page}&page_size=${pageSize}`)
  }

  async submitRepository(profileId: string, repository: string, signal?: AbortSignal): Promise<IngestionJob> {
    return this.request('/repository-ingestions', {
      method: 'POST', body: JSON.stringify({ profile_id: profileId, repository }), signal
    })
  }

  async getIngestion(id: string, signal?: AbortSignal): Promise<IngestionJob> {
    return this.request('/repository-ingestions/' + id, { signal })
  }

  async getIngestedSource(jobId: string, sourceId: string, signal?: AbortSignal): Promise<IngestedSourceDetail> {
    return this.request('/repository-ingestions/' + jobId + '/sources/' + sourceId, { signal })
  }

  async startAnalysis(ingestionId: string, signal?: AbortSignal): Promise<AnalysisStatus> {
    return this.request('/analyses', { method: 'POST', body: JSON.stringify({ ingestion_id: ingestionId }), signal })
  }
  async getAnalysis(id: string, signal?: AbortSignal): Promise<AnalysisStatus> {
    return this.request('/analyses/' + id, { signal })
  }
  async getAnalysisReport(id: string, signal?: AbortSignal): Promise<AnalysisReport> {
    return this.request('/analyses/' + id + '/report', { signal })
  }

  async questionPolicy(repositoryId: string, revisionId: string, signal?: AbortSignal): Promise<import('../types').QuestionPolicy> {
    return this.request(`/repositories/${repositoryId}/revisions/${revisionId}/question-policy`, { signal })
  }

  async askRepository(repositoryId: string, revisionId: string, data: import('../types').RepositoryQuestion, signal?: AbortSignal): Promise<import('../types').RepositoryAnswer> {
    return validateAnswer(await this.request(`/repositories/${repositoryId}/revisions/${revisionId}/questions`, { method: 'POST', body: JSON.stringify(data), signal }), repositoryId, revisionId)
  }

  async repositories(offset = 0, signal?: AbortSignal): Promise<import('../types').Page<import('../types').RepositoryIdentity>> {
    return this.request(`/repositories?limit=20&offset=${offset}`, { signal })
  }
  async revisions(repo: string, offset = 0, signal?: AbortSignal): Promise<import('../types').Page<import('../types').RepositoryRevision>> {
    return this.request(`/repositories/${encodeURIComponent(repo)}/revisions?limit=20&offset=${offset}`, { signal })
  }
  async investigationHistory(repo: string, rev: string, offset = 0, signal?: AbortSignal): Promise<import('../types').Page<import('../types').InvestigationHistoryItem>> {
    return this.request(`/repositories/${repo}/revisions/${rev}/durable-investigations?limit=10&offset=${offset}`, { signal })
  }
  async startInvestigation(repo: string, rev: string, objective: string, signal?: AbortSignal): Promise<{ id: string; status: 'pending' }> {
    const result = await this.request<{ id: string; status: 'pending' }>(`/repositories/${repo}/revisions/${rev}/durable-investigations`, { method: 'POST', body: JSON.stringify({ objective, generation_provider: 'gemini', hosted_processing_consent: 'gemini-source-processing-v1' }), signal })
    if (!result || !/^[a-f0-9-]{36}$/i.test(result.id) || result.status !== 'pending') throw Object.assign(new Error('Invalid submission response'), { status: 502 })
    return result
  }
  async investigation(repo: string, rev: string, id: string, signal?: AbortSignal): Promise<import('../types').InvestigationStatus> {
    return validateInvestigation(await this.request(`/repositories/${repo}/revisions/${rev}/durable-investigations/${id}`, { signal }), repo, rev, id)
  }
  async cancelInvestigation(repo: string, rev: string, id: string, signal?: AbortSignal): Promise<import('../types').InvestigationStatus> {
    return validateInvestigation(await this.request(`/repositories/${repo}/revisions/${rev}/durable-investigations/${id}/cancel`, { method: 'POST', signal }), repo, rev, id)
  }


  async careerInputs(offset = 0, signal?: AbortSignal) {
    const data = await this.request(`/career-inputs?limit=10&offset=${offset}`, {signal}); return career.checked(() => career.validateInputs(data))
  }
  async careerInput(id: string, signal?: AbortSignal) {
    const data = await this.request(`/career-inputs/${id}`, { signal }); return career.checked(() => career.validateInput(data,id))
  }
  async saveCareerInput(profile: string, github: string, job: string, signal?: AbortSignal) {
    const data = await this.request('/career-inputs', {method:'POST',body:JSON.stringify({profile_id:profile,github_profile_url:github,job_description:job}),signal}); return career.checked(() => career.validateInput(data,undefined,profile))
  }
  async selectCareer(input: CareerInput, repositories: string[], signal?: AbortSignal) {
    const data = await this.request(`/career-inputs/${input.id}/selection`, {method:'PUT',body:JSON.stringify({repositories}),signal}); return career.checked(() => career.validateInput(data,input.id))
  }
  async prepareCareer(input: CareerInput, signal?: AbortSignal) {
    const data = await this.request(`/career-inputs/${input.id}/prepare`, {method:'POST',signal}); return career.checked(() => career.validateInput(data,input.id))
  }
  async careerHistory(input: string, signal?: AbortSignal) {
    const data = await this.request(`/career-inputs/${input}/assessments`, {signal}); return career.checked(() => career.validateHistory(data))
  }
  async careerAssessment(input: CareerInput, id: string, signal?: AbortSignal) {
    const data = await this.request(`/career-assessments/${id}`, {signal}); return career.checked(() => career.validateAssessment(data,input,id))
  }
  async extractCareer(input: CareerInput, requestId: string, mode: RetrievalMode, signal?: AbortSignal) {
    const data = await this.request(`/career-inputs/${input.id}/assessments`, {method:'POST',body:JSON.stringify({request_id:requestId,retrieval_mode:mode,hosted_processing_consent:CAREER_CONSENT}),signal},true); return career.checked(() => career.validateAssessment(data,input))
  }
  async matchCareer(input: CareerInput, assessment: CareerAssessment, requirement: string, signal?: AbortSignal) {
    const data = await this.request(`/career-assessments/${assessment.id}/match`, {method:'POST',body:JSON.stringify({requirement_id:requirement,reviewed:true,hosted_processing_consent:CAREER_CONSENT}),signal},true); return career.checked(() => career.validateAssessment(data,input,assessment.id))
  }
  async cancelCareer(input: CareerInput, assessment: CareerAssessment, signal?: AbortSignal) {
    const data = await this.request(`/career-assessments/${assessment.id}/cancel`, {method:'POST',signal}); return career.checked(() => career.validateAssessment(data,input,assessment.id))
  }
  async careerRoadmap(parent: CareerAssessment, signal?: AbortSignal) {
    let data: unknown
    try { data = await this.request(`/career-assessments/${parent.id}/roadmap`, {signal}) }
    catch (error) { if ((error as {status?:number}).status === 404 && (error as Error).message === 'Career roadmap unavailable') return null; throw error }
    return career.checked(() => career.validateRoadmap(data,parent))
  }
  async generateCareerRoadmap(parent: CareerAssessment, signal?: AbortSignal) {
    const data = await this.request(`/career-assessments/${parent.id}/roadmap`, {method:'POST',body:JSON.stringify({hosted_processing_consent:CAREER_CONSENT}),signal},true); return career.checked(() => career.validateRoadmap(data,parent))
  }
  async cancelCareerRoadmap(parent: CareerAssessment, id: string, signal?: AbortSignal) {
    const data = await this.request(`/career-roadmaps/${id}/cancel`, {method:'POST',signal}); return career.checked(() => career.validateRoadmap(data,parent))
  }

  async deleteProfile(id: string): Promise<void> {
    return this.request(`/profiles/${id}`, { method: 'DELETE' })
  }
}

export const apiClient = new ApiClient()
