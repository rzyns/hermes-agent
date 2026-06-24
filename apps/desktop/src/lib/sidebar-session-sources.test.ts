import { describe, expect, it } from 'vitest'

import {
  normalizeSidebarSourceIds,
  sessionMatchesSidebarSources,
  SIDEBAR_DEFAULT_EXCLUDED_SOURCE_IDS,
  SIDEBAR_LOCAL_CHAT_SOURCE_IDS,
  sidebarSessionSourceFilter
} from './sidebar-session-sources'

describe('sidebar session source filters', () => {
  it('preserves the built-in sidebar grouping when no custom sources are selected', () => {
    expect(sidebarSessionSourceFilter([])).toEqual({
      excludeSources: SIDEBAR_DEFAULT_EXCLUDED_SOURCE_IDS
    })
    expect(SIDEBAR_DEFAULT_EXCLUDED_SOURCE_IDS).toContain('cron')
    expect(SIDEBAR_DEFAULT_EXCLUDED_SOURCE_IDS).toContain('subagent')
    expect(SIDEBAR_DEFAULT_EXCLUDED_SOURCE_IDS).toContain('tool')
    expect(SIDEBAR_DEFAULT_EXCLUDED_SOURCE_IDS).toContain('discord')
  })

  it('builds a server-side multi-source allowlist for CLI/TUI/WebUI filtering', () => {
    expect(sidebarSessionSourceFilter([' CLI ', 'tui', 'WEBUI', 'cli'])).toEqual({
      sources: SIDEBAR_LOCAL_CHAT_SOURCE_IDS
    })
  })

  it('normalizes and dedupes selected source ids without inventing defaults', () => {
    expect(normalizeSidebarSourceIds(['cli', ' CLI ', '', 'profile-delegate'])).toEqual(['cli', 'profile-delegate'])
  })

  it('filters preserved sidebar rows when an allowlist is active', () => {
    expect(sessionMatchesSidebarSources('cli', ['cli', 'tui', 'webui'])).toBe(true)
    expect(sessionMatchesSidebarSources('webui', ['cli', 'tui', 'webui'])).toBe(true)
    expect(sessionMatchesSidebarSources('acp', ['cli', 'tui', 'webui'])).toBe(false)
    expect(sessionMatchesSidebarSources('acp', [])).toBe(true)
  })
})
