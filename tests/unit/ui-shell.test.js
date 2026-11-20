/**
 * Browser-only tests of the shared UI shell (lib/ui/controls.js, lib/viz/canvas.js):
 * accessibility attributes, keyboard operation of canvas interactions, touch behaviour.
 * They need a DOM, so they are `browserOnly`: they run in tests/test.html (and thereby in
 * tools/smoke.sh) and are ignored under Deno.
 */
import { test, assert } from '../harness.js';
import { segmented, playControls, widgetLayout, slider, select } from '../../lib/ui/controls.js';
import { createCanvas, onPointer, theme } from '../../lib/viz/canvas.js';

/** A detached-looking but rendered sandbox figure; remove() it at the end of the test. */
function sandbox() {
  const fig = document.createElement('figure');
  fig.className = 'widget';
  fig.style.cssText = 'position:absolute; left:0; top:0; width:420px;';
  fig.innerHTML = '<figcaption><b>Sandbox figure.</b> caption text</figcaption>';
  document.body.appendChild(fig);
  return fig;
}
const key = (el, k, o = {}) => el.dispatchEvent(new KeyboardEvent('keydown', { key: k, bubbles: true, cancelable: true, ...o }));

test('canvas: accessible name/description; touch scrolling is blocked only on interactive canvases', () => {
  const fig = sandbox();
  try {
    const L = widgetLayout(fig), p = L.panel('solution $u_h$');
    const s = createCanvas(p, { aspect: 0.5 });
    const c = s.canvas, lab = p.querySelector('.canvas-label'), cap = fig.querySelector('figcaption');
    assert(c.getAttribute('role') === 'img', 'role=img');
    assert(lab.id && c.getAttribute('aria-labelledby') === lab.id, 'named by the panel label');
    assert(cap.id && c.getAttribute('aria-describedby') === cap.id, 'described by the caption');
    assert(c.textContent.length > 20, 'fallback text inside <canvas>');
    assert(getComputedStyle(c).touchAction !== 'none', `a plain canvas must not block touch scrolling (touch-action: ${getComputedStyle(c).touchAction})`);
    assert(c.tabIndex < 0, 'a plain canvas is not a tab stop');
    const s2 = createCanvas(L.panel(), { label: 'explicit name', description: 'explicit description' });
    assert(s2.canvas.getAttribute('aria-label') === 'explicit name' && s2.canvas.textContent === 'explicit description', 'explicit label/description');
    onPointer(s2.canvas, { down: () => {} });
    assert(getComputedStyle(s2.canvas).touchAction === 'none' && s2.canvas.tabIndex === 0, 'interactive canvas: touch-action none, focusable');
    assert(typeof theme().onAccent === 'string' && theme().onAccent.length > 0, 'theme().onAccent');
  } finally { fig.remove(); }
}, { browserOnly: true });

test('onPointer: keyboard virtual pointer drives the same down / move / up / hover callbacks as the mouse', () => {
  const fig = sandbox();
  try {
    const s = createCanvas(widgetLayout(fig).panel('drag me'), { height: 200 });
    const c = s.canvas, log = [];
    let grab = true;
    const api = onPointer(c, {
      down: (x, y, e) => { log.push(['down', x, y, e.pointerType]); return grab ? undefined : false; },
      move: (x, y) => log.push(['move', x, y]), up: () => log.push(['up']), hover: (x, y) => log.push(['hover', x, y]),
      handles: () => [[40, 50], [120, 90]],
    });
    assert(c.getAttribute('role') === 'application' && /arrow keys/.test(c.textContent), 'role + keyboard help in the fallback text');
    c.focus();
    assert(key(c, 'ArrowRight') === false, 'arrow key is consumed (no page scroll)'); // starts at the first handle (40, 50)
    assert(log.length === 1 && log[0][0] === 'hover' && log[0][1] === 48 && log[0][2] === 50, JSON.stringify(log));
    key(c, ' ');                      // press
    key(c, 'ArrowDown');              // drag by 8 px
    key(c, 'ArrowLeft', { shiftKey: true }); // fine step: 1 px
    key(c, 'Enter');                  // release
    assert(JSON.stringify(log.slice(1)) === JSON.stringify([['down', 48, 50, 'keyboard'], ['move', 48, 58], ['move', 47, 58], ['up']]), JSON.stringify(log));
    assert(api.position()[0] === 47 && api.position()[1] === 58, 'position()');
    assert(fig.querySelector('.kbd-cursor') && !fig.querySelector('.kbd-cursor').hidden, 'the virtual pointer is visible while the keyboard is used');
    log.length = 0;
    key(c, 'n'); key(c, 'n');         // jump to handle 0, then handle 1
    assert(log.length === 2 && log[1][1] === 120 && log[1][2] === 90, JSON.stringify(log));
    grab = false; log.length = 0;
    key(c, ' '); key(c, 'ArrowUp');   // down() returned false: a click, no drag → the arrow is a hover again
    assert(log[0][0] === 'down' && log[1][0] === 'hover', JSON.stringify(log));
    grab = true; log.length = 0;
    key(c, ' '); key(c, 'Escape');    // Escape releases
    assert(log.length === 2 && log[1][0] === 'up', JSON.stringify(log));
    key(c, ' '); c.dispatchEvent(new FocusEvent('blur')); // losing focus releases, too
    assert(log[log.length - 1][0] === 'up' && fig.querySelector('.kbd-cursor').hidden, 'blur releases and hides the cursor');
    assert(key(c, 'a') === true, 'other keys are not consumed');
    // real pointer events still work and are clamped to nothing (regression: same callbacks)
    log.length = 0;
    const r = c.getBoundingClientRect();
    c.dispatchEvent(new PointerEvent('pointermove', { clientX: r.left + 10, clientY: r.top + 20, bubbles: true }));
    assert(log.length === 1 && log[0][0] === 'hover' && Math.abs(log[0][1] - 10) < 1e-6 && Math.abs(log[0][2] - 20) < 1e-6, JSON.stringify(log));
  } finally { fig.remove(); }
}, { browserOnly: true });

test('controls: segmented buttons expose their state (aria-pressed), playControls follows the animator', () => {
  const fig = sandbox();
  try {
    const L = widgetLayout(fig);
    let got = null;
    const seg = segmented(L.controls, { label: 'layout $x$', value: 'a', options: [{ value: 'a', label: 'A' }, { value: 'b', label: 'B' }], onChange: (v) => { got = v; } });
    const [ba, bb] = seg.el.querySelectorAll('button');
    assert(seg.el.querySelector('.seg').getAttribute('role') === 'group' && seg.el.querySelector('.seg').getAttribute('aria-label') === 'layout x', 'group label');
    assert(ba.getAttribute('aria-pressed') === 'true' && bb.getAttribute('aria-pressed') === 'false', 'initial state');
    bb.click();
    assert(got === 'b' && seg.value === 'b' && ba.getAttribute('aria-pressed') === 'false' && bb.getAttribute('aria-pressed') === 'true' && bb.classList.contains('on'), 'after click');
    seg.set('a');
    assert(ba.getAttribute('aria-pressed') === 'true' && got === 'b', 'set() without fire');
    // playControls
    const anim = { o: {}, running: false, toggle() { this.running = !this.running; this.o.onState?.(this.running); } };
    let resets = 0;
    const play = playControls(L.controls, anim, () => { resets++; });
    const [bp, br] = play.row.querySelectorAll('button');
    assert(/Play/.test(bp.textContent) && /Reset/.test(br.textContent), 'labels');
    bp.click(); assert(anim.running && /Pause/.test(bp.textContent), 'play → pause label');
    bp.click(); assert(!anim.running && /Play/.test(bp.textContent), 'pause → play label');
    br.click(); assert(resets === 1, 'reset callback');
    // existing controls keep their API
    const sl = slider(L.controls, { label: 'p', min: 1, max: 3, step: 1, value: 2 }), se = select(L.controls, { label: 's', value: 'x', options: [{ value: 'x', label: 'X' }] });
    assert(sl.value === 2 && se.value === 'x' && sl.input.getAttribute('aria-label') === 'p', 'slider/select');
  } finally { fig.remove(); }
}, { browserOnly: true });
