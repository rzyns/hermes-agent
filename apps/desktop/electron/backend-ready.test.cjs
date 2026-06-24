/**
 * Tests for electron/backend-ready.cjs.
 *
 * Run with: node --test electron/backend-ready.test.cjs
 * (Wired into npm test:desktop:platforms in package.json.)
 *
 * Covers the cold-start port-announcement deadline (issue #50209): the clock
 * starts before the backend binds its port, so a tight 45s deadline killed a
 * healthy-but-still-compiling backend on cold Windows installs. The default is
 * now cold-start tolerant and overridable via
 * HERMES_DESKTOP_PORT_ANNOUNCE_TIMEOUT_MS, clamped to a 45s floor.
 */

const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const { EventEmitter } = require('node:events')

const {
  waitForDashboardPort,
  resolvePortAnnounceTimeoutMs,
  DEFAULT_PORT_ANNOUNCE_TIMEOUT_MS,
  MIN_PORT_ANNOUNCE_TIMEOUT_MS,
} = require('./backend-ready.cjs')

// A minimal stand-in for a spawned child process: an EventEmitter with stdout
// and stderr EventEmitters, matching the surface waitForDashboardPort consumes
// (child.<stream>.on('data'), child.on('exit'|'error') + the .off() teardown).
function makeFakeChild() {
  const child = new EventEmitter()
  child.stdout = new EventEmitter()
  child.stderr = new EventEmitter()
  return child
}

// ---------------------------------------------------------------------------
// resolvePortAnnounceTimeoutMs
// ---------------------------------------------------------------------------

test('default is cold-start tolerant (> the historical 45s floor)', () => {
  assert.equal(resolvePortAnnounceTimeoutMs({}), DEFAULT_PORT_ANNOUNCE_TIMEOUT_MS)
  assert.ok(
    DEFAULT_PORT_ANNOUNCE_TIMEOUT_MS > MIN_PORT_ANNOUNCE_TIMEOUT_MS,
    'cold-start default must exceed the warm-start floor'
  )
})

test('honors a valid HERMES_DESKTOP_PORT_ANNOUNCE_TIMEOUT_MS override', () => {
  const env = { HERMES_DESKTOP_PORT_ANNOUNCE_TIMEOUT_MS: '120000' }
  assert.equal(resolvePortAnnounceTimeoutMs(env), 120_000)
})

test('clamps an override below the floor up to the 45s minimum', () => {
  const env = { HERMES_DESKTOP_PORT_ANNOUNCE_TIMEOUT_MS: '1000' }
  assert.equal(resolvePortAnnounceTimeoutMs(env), MIN_PORT_ANNOUNCE_TIMEOUT_MS)
})

test('rounds a fractional override', () => {
  const env = { HERMES_DESKTOP_PORT_ANNOUNCE_TIMEOUT_MS: '60000.7' }
  assert.equal(resolvePortAnnounceTimeoutMs(env), 60_001)
})

test('falls back to the default for malformed / non-positive overrides', () => {
  for (const bad of ['', 'abc', '0', '-5', 'NaN', undefined]) {
    const env = bad === undefined ? {} : { HERMES_DESKTOP_PORT_ANNOUNCE_TIMEOUT_MS: bad }
    assert.equal(
      resolvePortAnnounceTimeoutMs(env),
      DEFAULT_PORT_ANNOUNCE_TIMEOUT_MS,
      `override ${JSON.stringify(bad)} should fall through to the default`
    )
  }
})

// ---------------------------------------------------------------------------
// waitForDashboardPort
// ---------------------------------------------------------------------------

test('resolves with the announced port', async () => {
  const child = makeFakeChild()
  const p = waitForDashboardPort(child, 1000)
  child.stdout.emit('data', 'noise before\nHERMES_DASHBOARD_READY port=54321\n')
  assert.equal(await p, 54321)
})

test('resolves when the announcement arrives on stderr', async () => {
  const child = makeFakeChild()
  const p = waitForDashboardPort(child, 1000)
  child.stderr.emit('data', 'HERMES_DASHBOARD_READY port=44787\n')
  assert.equal(await p, 44787)
})

test('parses the port even when the line arrives split across chunks', async () => {
  const child = makeFakeChild()
  const p = waitForDashboardPort(child, 1000)
  child.stdout.emit('data', 'HERMES_DASHBOARD_READY po')
  child.stdout.emit('data', 'rt=8080\n')
  assert.equal(await p, 8080)
})

test('rejects when the child exits before announcing', async () => {
  const child = makeFakeChild()
  const p = waitForDashboardPort(child, 1000)
  child.emit('exit', 1, null)
  await assert.rejects(p, /exited before port announcement/)
})

test('rejects on a child error event', async () => {
  const child = makeFakeChild()
  const p = waitForDashboardPort(child, 1000)
  child.emit('error', new Error('spawn ENOENT'))
  await assert.rejects(p, /spawn ENOENT/)
})

test('rejects with the timeout message after the deadline', async () => {
  const child = makeFakeChild()
  await assert.rejects(
    waitForDashboardPort(child, 20),
    /Timed out waiting for Hermes backend port announcement \(20ms\)/
  )
})

test('a late announcement after timeout does not throw (listeners torn down)', async () => {
  const child = makeFakeChild()
  await assert.rejects(waitForDashboardPort(child, 20), /Timed out/)
  // The orphaned backend may still print its READY line later; the watcher
  // must have detached so this emit is a no-op rather than a double-settle.
  assert.doesNotThrow(() => {
    child.stdout.emit('data', 'HERMES_DASHBOARD_READY port=9999\n')
  })
})

test('primary desktop backend arms the port watcher before boot progress can yield', () => {
  const source = fs.readFileSync(path.join(__dirname, 'main.cjs'), 'utf8')
  const primarySpawn = source.indexOf('hermesProcess = spawn(')
  assert.notEqual(primarySpawn, -1, 'primary backend spawn block should exist')

  const watcher = source.indexOf('waitForDashboardPort(hermesProcess)', primarySpawn)
  const yieldingProgress = source.indexOf("await advanceBootProgress('backend.port'", primarySpawn)

  assert.notEqual(watcher, -1, 'primary backend must create a waitForDashboardPort watcher')
  assert.notEqual(yieldingProgress, -1, 'primary backend must update backend.port boot progress')
  assert.ok(
    watcher < yieldingProgress,
    'port watcher must be armed before the awaited backend.port progress update; otherwise a warm backend can print READY while only rememberLog is listening'
  )
})
