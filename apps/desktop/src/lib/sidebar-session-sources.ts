import type { SessionSourceFilter } from '../hermes'

import { MESSAGING_SESSION_SOURCE_IDS, normalizeSessionSource } from './session-source'

// Keep the historical sidebar behavior as the empty-preference default: main
// recents hide scheduler/tool/subagent rows and every external messaging
// platform, which is fetched into separate sidebar sections instead.
export const SIDEBAR_DEFAULT_EXCLUDED_SOURCE_IDS = ['cron', 'subagent', 'tool', ...MESSAGING_SESSION_SOURCE_IDS]

export const SIDEBAR_LOCAL_CHAT_SOURCE_IDS = ['cli', 'tui', 'webui']

export const SIDEBAR_SOURCE_OPTION_IDS = [
  'cli',
  'tui',
  'webui',
  'desktop',
  'codex',
  'acp',
  'profile-delegate',
  'api_server',
  'gateway',
  'local'
]

export function normalizeSidebarSourceIds(sources: readonly string[] | null | undefined): string[] {
  const seen = new Set<string>()
  const normalized: string[] = []

  for (const source of sources ?? []) {
    const id = normalizeSessionSource(source)

    if (!id || seen.has(id)) {
      continue
    }

    seen.add(id)
    normalized.push(id)
  }

  return normalized
}

export function sidebarSessionSourceFilter(sources: readonly string[] | null | undefined): SessionSourceFilter {
  const normalized = normalizeSidebarSourceIds(sources)

  if (normalized.length > 0) {
    return { sources: normalized }
  }

  return { excludeSources: SIDEBAR_DEFAULT_EXCLUDED_SOURCE_IDS }
}

export function sessionMatchesSidebarSources(
  source: null | string | undefined,
  sources: readonly string[] | null | undefined
): boolean {
  const normalized = normalizeSidebarSourceIds(sources)

  if (normalized.length === 0) {
    return true
  }

  const sourceId = normalizeSessionSource(source)

  return sourceId != null && normalized.includes(sourceId)
}
