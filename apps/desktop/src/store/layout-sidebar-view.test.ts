import { beforeEach, describe, expect, it } from 'vitest'

import {
  $sidebarFiltersActive,
  $sidebarGrouping,
  $sidebarOrdering,
  $sidebarRowMeta,
  $sidebarSourceFilter,
  $sidebarViewCustomized,
  resetSidebarView,
  setSidebarGrouping,
  setSidebarOrdering,
  toggleSidebarRowMeta,
  toggleSidebarSourceFilter,
  toggleSidebarStatusFilter
} from './layout'
import { $showAllProfiles } from './profile'

beforeEach(() => {
  $showAllProfiles.set(false)
  resetSidebarView()
})

describe('the sidebar as it ships', () => {
  it('groups by date, sorts by recency, and pins the timestamp and preview', () => {
    expect($sidebarGrouping.get()).toBe('date')
    expect($sidebarOrdering.get()).toBe('updated')
    expect($sidebarRowMeta.get()).toEqual(['preview', 'updated'])
  })

  it('offers no reset until something actually moves off the defaults', () => {
    expect($sidebarViewCustomized.get()).toBe(false)

    toggleSidebarRowMeta('tokens')

    expect($sidebarViewCustomized.get()).toBe(true)
  })

  it('is what reset puts back — every knob, not just the filters', () => {
    setSidebarGrouping('project')
    setSidebarOrdering('cost')
    toggleSidebarRowMeta('updated')
    toggleSidebarRowMeta('cost')
    toggleSidebarStatusFilter('working')

    resetSidebarView()

    expect($sidebarGrouping.get()).toBe('date')
    expect($sidebarOrdering.get()).toBe('updated')
    expect($sidebarRowMeta.get()).toEqual(['preview', 'updated'])
    expect($sidebarViewCustomized.get()).toBe(false)
  })

  it('ships by date in the all-profiles scope too, and resets back to it', () => {
    $showAllProfiles.set(true)
    setSidebarGrouping('profile')

    resetSidebarView()

    expect($sidebarGrouping.get()).toBe('date')
    expect($sidebarViewCustomized.get()).toBe(false)
  })

  it('resets the scope the user is not looking at, so flipping the rail cannot restore it', () => {
    setSidebarGrouping('status')
    $showAllProfiles.set(true)
    setSidebarGrouping('profile')

    resetSidebarView()
    $showAllProfiles.set(false)

    expect($sidebarGrouping.get()).toBe('date')
  })

  it('turns all-profiles on when the user groups by profile, since that is the ask', () => {
    setSidebarGrouping('profile')

    expect($showAllProfiles.get()).toBe(true)
    expect($sidebarGrouping.get()).toBe('profile')
  })

  it('counts a source selection as an active filter, so the funnel lights up', () => {
    expect($sidebarFiltersActive.get()).toBe(false)

    toggleSidebarSourceFilter('cli')

    expect($sidebarSourceFilter.get()).toEqual(['cli'])
    expect($sidebarFiltersActive.get()).toBe(true)
    expect($sidebarViewCustomized.get()).toBe(true)
  })

  it('clears the source selection on reset, back to the default taxonomy', () => {
    toggleSidebarSourceFilter('cli')
    toggleSidebarSourceFilter('webui')

    expect($sidebarSourceFilter.get()).toEqual(['cli', 'webui'])

    resetSidebarView()

    expect($sidebarSourceFilter.get()).toEqual([])
    expect($sidebarFiltersActive.get()).toBe(false)
  })
})
