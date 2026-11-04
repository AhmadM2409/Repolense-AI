import { useEffect, useState } from 'react'

import { useParams, Link } from 'react-router-dom'

import { apiClient } from '../services/api'

import { AnalysisReport } from '../types'

import { useDurableJob } from '../hooks/useDurableJob'

import { JobProgress } from '../components/JobProgress'

const loadAnalysis = (id: string, signal: AbortSignal) => apiClient.getAnalysis(id, signal)



export function RepositoryAnalysisPage() {

  const { jobId = '' } = useParams<{ jobId: string }>()

  const { job: status, error: pollingError, cancel, cancelling, retrying } = useDurableJob(jobId, loadAnalysis)

  const [report, setReport] = useState<AnalysisReport | null>(null)

  const [error, setError] = useState('')

  const [reload, setReload] = useState(0)

  useEffect(() => {

    const active = new AbortController()

    setReport(null); setError('')

    let timer: ReturnType<typeof setTimeout> | undefined

    if (status?.status === 'complete') {

      apiClient.getAnalysisReport(jobId, active.signal).then(data => {

        if (!active.signal.aborted) setReport(data)

      }).catch(cause => {

        if (!active.signal.aborted) {

          setError(cause instanceof Error ? cause.message : 'Report unavailable')

          if (!(cause.status >= 400 && cause.status < 500)) timer = setTimeout(() => setReload(value => value + 1), 2000)

        }

      })

    }

    return () => { active.abort(); clearTimeout(timer) }

  }, [jobId, status?.status, reload])

  return <main className="max-w-5xl mx-auto p-8 space-y-6">

    <Link to="/repositories/new" className="text-blue-600">Repository ingestion</Link>

    <h1 className="text-2xl font-bold">Deterministic repository analysis</h1>

    <p>Source observations and potential risks from versioned Python rules. Repository code is not executed.</p>

    {pollingError ? <p role="alert">{pollingError}{retrying ? ". Retrying connection…" : ""}</p> : null}

    {status ? <JobProgress job={status} cancel={cancel} cancelling={cancelling} /> : null}

    {error ? <p role="alert" className="text-red-700">{error}</p> : null}



    {status?.status === 'failed' ? <p role="alert">{status.error_message || 'Analysis failed'}</p> : null}

    {report ? <>

      <p>Repository: {report.repository}</p>

      <p className="break-all">Commit: <code>{report.commit_sha}</code> Â· Rules: {report.engine_version}</p>

      <p>{report.summary.scope}</p>

      <ul>{(report.summary.limitations || []).map(item => <li key={item}>{item}</li>)}</ul>

      <section><h2 className="text-xl font-semibold">Architecture and technology observations</h2>

        <p>{report.summary.file_count} selected files; {report.summary.documentation.count} documentation files.</p>

        <h3 className="font-semibold mt-3">Source languages (observed file counts)</h3>

        <ul>{Object.entries(report.summary.languages).map(([name, count]) => <li key={name}>{name}: {count}</li>)}</ul>

        <h3 className="font-semibold mt-3">Major directories (observed)</h3>

        <ul>{Object.entries(report.summary.major_directories).map(([name,count]) => <li key={name}>{name}: {count}</li>)}</ul>

        <h3 className="font-semibold mt-3">Dependency declarations</h3>

        <p>Declarations do not prove installation or use.</p>

        <ul>{report.summary.dependencies.map((item,index) => <li key={index}>

          {item.name}: {item.declaration} ({item.ecosystem}, {item.section})

          {item.evidence.map((span,i) => <EvidenceView key={i} span={span} />)}

        </li>)}</ul>

        <h3 className="font-semibold mt-3">Framework indicators (inferred)</h3>

        <ul>{report.summary.framework_indicators.map((item,index) => <li key={index}>{item.name}: {item.basis}</li>)}</ul>

        <h3 className="font-semibold mt-3">Test framework imports (observed)</h3>

        <ul>{(report.summary.test_framework_indicators || []).map((item,index) => <li key={index}>{item.name}: {item.basis}

          {item.evidence.map((span,i) => <EvidenceView key={i} span={span} />)}

        </li>)}</ul>

        <p>Test locations: {report.summary.test_locations.join(', ') || 'None in the analyzed selection'}</p>

        <p>CI configuration files: {report.summary.ci_configurations.map(item => item.path).join(', ') || 'None in the analyzed selection'}</p>

        <p>Deployment configuration files: {report.summary.deployment_configurations.map(item => item.path).join(', ') || 'None in the analyzed selection'}</p>

        <h3 className="font-semibold mt-3">Entrypoint candidates (inferred, not executed)</h3>

        <ul>{report.summary.entrypoint_candidates.map((item,index) => <li key={index}>{item.name}: {item.target}</li>)}</ul>

        {report.summary.diagnostics.length ? <><h3 className="font-semibold">Coverage diagnostics</h3>

          <ul>{report.summary.diagnostics.map((item,index) => <li key={index}>{item.path}: {item.code}</li>)}</ul></> : null}

      </section>

      <section><h2 className="text-xl font-semibold">Source-backed findings</h2>

        {report.findings.length === 0 ? <p>No supported issues were detected in the analyzed selection. This is not a security certification.</p> : null}

        {['critical', 'high', 'medium', 'low', 'info'].map(severity => {

          const matches = report.findings.filter(finding => finding.severity === severity)

          return matches.length ? <section key={severity}><h3 className="text-lg font-semibold capitalize">{severity}</h3>

            {['security','configuration','maintainability'].map(category => <div key={category}>

              {matches.some(item => item.category === category) ? <h4 className="font-semibold capitalize">{category}</h4> : null}

              {matches.filter(item => item.category === category).map(finding => <article key={finding.id} className="border rounded p-4 my-3">

                <h5 className="font-semibold">{finding.title} Â· {finding.rule_id}</h5>

                <p>{finding.certainty === 'potential' ? 'Potential risk; exploitability is not established.' : 'Confirmed source observation.'}</p>

                <p>{finding.detail}</p>

                {finding.evidence.map((span,index) => <EvidenceView key={index} span={span} />)}

              </article>)}

            </div>)}

          </section> : null

        })}

      </section>

    </> : null}

  </main>

}

function EvidenceView({ span }: { span: import('../types').SourceEvidence }) {

  return <details className="my-2">

    <summary>{span.path}:{span.line_start}â€“{span.line_end} Â· {span.commit_sha}</summary>

    <pre className="overflow-auto bg-gray-50 border p-3 text-sm">{span.quote}</pre>

  </details>

}
