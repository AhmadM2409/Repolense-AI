export const CAREER_CONSENT = 'career-gemini-job-and-source-v1' as const
export type RetrievalMode = 'keyword' | 'semantic' | 'hybrid'
export type Classification = 'demonstrated' | 'partially demonstrated' | 'documented/claimed' | 'not evidenced'
export interface RevisionScope { repository_id: string; revision_id: string; commit_sha: string; full_name: string; digest: string; index_id?: string; index_signature?: string }
export interface Manifest { input_id: string; job_hash: string; revisions: RevisionScope[]; version: 1; discovery_status: string }
export interface CareerInput {
  id: string; profile_id: string; github_profile_url: string; job_description: string; job_hash: string; manifest_version: 1
  discovery: { version: 1; status: 'complete' | 'bounded' | 'partial' | 'unavailable'; reason: string | null; private_coverage: 'not_accessed_or_enumerated'; requests: number; items: { full_name: string; eligible: boolean; excluded_reason: string | null; description: string | null; language: string | null }[] }
  selection: { full_name: string; position: number; job_id: string | null; status: 'not_started' | 'pending' | 'processing' | 'awaiting_pin' | 'ready' | 'failed' | 'cancelled'; error_category: string | null; repository_id: string | null; revision_id: string | null; commit_sha: string | null }[]
  status: 'input_created' | 'selection_saved' | 'preparing' | 'ready' | 'partial' | 'failed'; created_at: string; updated_at: string; future_hosted_consent: 'not_granted'
}
export interface CareerInputSummary { id: string; profile_id: string; account: string; created_at: string; started: boolean }
export interface Requirement { id: string; quote: string; skill: string; aliases: string[]; priority: 'required' | 'preferred'; depth: string; start: number; end: number; job_hash: string }
export interface CareerCitation { evidence_id: string; repository_id: string; revision_id: string; commit_sha: string; source_id: string; chunk_id: string; path: string; line_start: number; line_end: number; quote: string; evidence_kind: 'code' | 'test' | 'configuration' | 'documentation' }
export interface RequirementMatch { requirement_id: string; classification: Classification; explanation: string; uncertainty: string; citations: CareerCitation[]; support_verdict: 'supported' | 'contradicted' | 'insufficient' | 'invalid'; coverage: Record<string, unknown> }
export interface CareerAssessment {
  id: string; input_id: string; status: 'extracting' | 'review' | 'matching' | 'complete' | 'failed' | 'cancelled'; error_category: string | null
  manifest: Manifest; requirements: Requirement[]; matches: RequirementMatch[]; inference_spent: number; policy_version: 'career-evidence-v1'; model: 'gemini-3.5-flash-lite'; retrieval_mode: RetrievalMode; created_at: string; operation_expired: boolean; coverage_notice: string
}
export interface AssessmentSummary { id: string; status: CareerAssessment['status']; created_at: string; inference_spent: number }
export interface Recommendation {
  requirement_id: string; action_type: 'learning' | 'coding' | 'testing' | 'documenting' | 'deployment'; action: string; gap: string; rationale: string; completion_proof: string; project_citation: number | null
  proposal_notice: 'Proposed future work; not existing portfolio capability.'; requirement: Requirement; observation: RequirementMatch; project: RevisionScope | null; priority: 'high' | 'medium'; priority_rationale: string; effort: 'unknown'; effort_uncertainty: string
}
export interface CareerRoadmap {
  id: string; assessment_id: string; version: 1; status: 'generating' | 'complete' | 'failed' | 'cancelled'; error_category: string | null
  result: { recommendations: Recommendation[]; demonstrated_requirement_ids: string[]; coverage_notice: string } | null
  source_signature: string; manifest: Manifest; inference_spent: number; roadmap_inference_spent: number; policy_version: 'career-roadmap-v1'; model: 'gemini-3.5-flash-lite'; created_at: string; operation_expired: boolean
}
