/**
 * Accessibility regression test: the text/background colour pairs of css/style.css must reach
 * WCAG 2.x AA contrast (4.5:1 for normal text) in the light AND the dark scheme.
 * The colours are read from the stylesheet, so changing a variable there re-runs the check.
 */
import { test, assert } from '../harness.js';

/** WCAG relative luminance of '#rrggbb'. */
const lum = (h) => {
  const c = [1, 3, 5].map((k) => parseInt(h.slice(k, k + 2), 16) / 255).map((v) => (v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4));
  return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
};
const contrast = (a, b) => { const x = lum(a), y = lum(b); return (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05); };

/** Parse `--name: #rrggbb;` declarations of a CSS block. */
const vars = (block) => Object.fromEntries([...block.matchAll(/--([\w-]+):\s*(#[0-9a-fA-F]{6})\b/g)].map((m) => [m[1], m[2]]));

/** [text colour, background] pairs that occur in the stylesheet / widgets, by variable name. */
const PAIRS = [
  ['ink', 'bg'], ['ink', 'bg-elev'], ['ink-soft', 'bg'], ['ink-soft', 'bg-sunk'], ['ink-soft', 'bg-elev'],
  ['ink-faint', 'bg'], ['ink-faint', 'bg-sunk'], ['ink-faint', 'bg-elev'],            // chapter numbers, sidebar headings, footers
  ['accent', 'bg'], ['accent', 'bg-elev'], ['accent', 'bg-sunk'], ['accent', 'note-bg'], // links, box titles
  ['accent-2', 'warn-bg'], ['accent-2', 'bg-elev'], ['accent-3', 'good-bg'], ['accent-3', 'bg-elev'], ['accent-4', 'bg-elev'],
  ['on-accent', 'accent'],                                                             // primary buttons, active segmented buttons
  ['on-accent', 'accent-2'], ['on-accent', 'accent-3'],                                // labels on accent discs in canvases
];

test('css/style.css: text/background pairs reach WCAG AA contrast (≥ 4.5:1) in light and dark mode', async () => {
  const css = await Deno.readTextFile(new URL('../../css/style.css', import.meta.url));
  const light = vars(/:root\s*\{([^}]*)\}/.exec(css)[1]);
  const dark = { ...light, ...vars(/@media \(prefers-color-scheme: dark\)\s*\{\s*:root\s*\{([^}]*)\}/.exec(css)[1]) };
  assert(Object.keys(light).length >= 15 && dark.bg !== light.bg, 'colour variables parsed');
  for (const [name, T] of [['light', light], ['dark', dark]]) for (const [fg, bg] of PAIRS) {
    assert(T[fg] && T[bg], `missing variable --${fg} or --${bg}`);
    const c = contrast(T[fg], T[bg]);
    assert(c >= 4.5, `${name}: --${fg} ${T[fg]} on --${bg} ${T[bg]} has contrast ${c.toFixed(2)} < 4.5`);
  }
  assert(Math.abs(contrast('#ffffff', '#000000') - 21) < 1e-9, 'sanity: white on black is 21:1');
  // buttons must take their text colour from --on-accent, not a hard-coded white
  assert(!/\.primary[^}]*color:\s*#fff/.test(css) && !/\.seg button\.on[^}]*color:\s*#fff/.test(css), 'hard-coded #fff on accent backgrounds');
  // canvases must not block touch scrolling globally (onPointer sets it per interactive canvas)
  assert(!/canvas[^}]*\{[^}]*touch-action:\s*none/.test(css.replace(/\/\*[\s\S]*?\*\//g, '')), 'global touch-action: none on canvases');
  assert(/prefers-reduced-motion:\s*reduce/.test(css), 'reduced-motion rules present');
}, { denoOnly: true });
