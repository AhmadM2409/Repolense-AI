export interface User {
  id: string
  email: string
}

export interface Profile {
  id: string
  github_username: string | null
  user_id: string
  resume_filename: string | null
  portfolio_url?: string
  created_at: string
  updated_at: string
}

export interface FeedbackSection {
  section_name: string
  content: string
  suggestions: string[]
  confidence: number
}

export interface Review {
  id: string
  profile_id: string
  status: 'pending' | 'processing' | 'complete' | 'failed' | 'cancelled'
  revision_id?: string | null
  report_kind?: 'legacy_unverified' | 'demo' | 'evidence'
  overall_score?: number | null
  sections?: FeedbackSection[] | null
  error_message?: string | null
  created_at: string
  updated_at: string
}

export interface ReviewListResponse {
  items: Review[]
  total: number
  page: number
  page_size: number
}

export interface AuthResponse {
  access_token: string
  token_type: string
}


export interface ReviewStatus {
  review_id: string
  status: Review["status"]
  progress_pct: number | null
  error_message?: string | null
}


export interface IngestionResult {
  repository_id?: string
  commit_sha?: string
  file_count?: number
  chunk_count?: number
  deduplicated?: boolean
  error_code?: string
  files?: { id: string; path: string; content_hash: string; chunk_count: number }[]
  skipped?: { path: string; reason: string }[]
}
export interface IngestionJob extends DurableJob {
  id: string
  profile_id: string
  requested_repository: string
  revision_id: string | null
  status: Review["status"]
  ingestion_result: IngestionResult | null
  error_message: string | null
}
export interface IngestedSourceDetail {
  source: { id: string; path: string; text: string; content_hash: string }
  chunks: { id: string; text: string; line_start: number; line_end: number }[]
}

export interface SourceEvidence {
  source_id: string; repository_id: string; commit_sha: string; path: string
  line_start: number; line_end: number; quote: string
}
export interface SourceFinding {
  id: string; title: string; detail: string; rule_id: string
  severity: string; category: string; certainty: 'observation' | 'potential'
  evidence: SourceEvidence[]
}
export interface AnalysisStatus extends DurableJob {
  id: string; profile_id: string; revision_id: string
  status: Review["status"]; error_message: string | null; progress_pct: number | null; engine_version: string
}
export interface AnalysisReport {
  review_id: string; revision_id: string; commit_sha: string; repository: string; engine_version: string
  producer: 'deterministic'; findings: SourceFinding[]
  summary: {
    limitations?: string[]
    scope: string; file_count: number; languages: Record<string,number>; major_directories: Record<string,number>
    documentation: { count: number; selected_documents: string[] }
    dependencies: { name: string; declaration: string; ecosystem: string; section: string; evidence: SourceEvidence[] }[]
    framework_indicators: { name: string; basis: string }[]
    test_framework_indicators?: { name: string; basis: string; evidence: SourceEvidence[] }[]
    test_locations: string[]; ci_configurations: { path: string }[]; deployment_configurations: { path: string }[]
    entrypoint_candidates: { name: string; target: string }[]; diagnostics: { path: string; code: string }[]
  }
}

export interface DurableJob {
  id: string
  status: Review["status"]
  progress_stage?: string
  progress_pct?: number | null
  attempt_count?: number
  max_attempts?: number
  retry_at?: string | null
  cancel_requested_at?: string | null
  error_category?: string | null
  error_message?: string | null
}

export interface QuestionPolicy { default_provider: 'lmstudio' | 'gemini'; hosted_available: boolean; consent_version: 'gemini-source-processing-v1'; notice: string; durable_investigations_available?: boolean }
export interface RepositoryQuestion { question: string; retrieval_mode: 'keyword' | 'hybrid' | 'semantic'; generation_provider: 'lmstudio' | 'gemini'; hosted_processing_consent?: 'gemini-source-processing-v1' }
export interface RepositoryAnswer { status: string; answer: string; uncertainty: string; model: string; revision_id: string; citations: (SourceEvidence & { chunk_id: string; revision_id: string })[] }


export interface Page<T> { items: T[]; next_offset: number | null }
export interface RepositoryIdentity { id: string; full_name: string; provider: string }
export interface RepositoryRevision { id: string; repository_id: string; commit_sha: string }
export interface InvestigationHistoryItem { id: string; objective: string; status: string; outcome: string | null; created_at: string; completed_at: string | null }
export interface InvestigationStep { number: number; operation: 'search' | 'answer' | 'stop'; origin: 'planner' | 'readiness'; evidence_paths: string[] }
export interface InvestigationResult { execution_id: string; repository_id: string; revision_id: string; status: string; steps: InvestigationStep[]; final: RepositoryAnswer | null }
export interface InvestigationStatus { id: string; repository_id: string; revision_id: string; status: 'pending' | 'processing' | 'complete' | 'failed' | 'cancelled'; outcome: string | null; deadline_at: string; inference_spent: number; checkpoint_count: number; cancel_requested: boolean; result: InvestigationResult | null }
