import { test, assert } from '../harness.js';
import { TEST_FILES } from '../manifest.js';

test('manifest lists every *.test.js file', async () => {
  const found = [];
  const root = new URL('../', import.meta.url);
  for (const dir of ['unit', 'conv', 'gpu']) {
    try {
      for await (const e of Deno.readDir(new URL(dir + '/', root))) if (e.name.endsWith('.test.js')) found.push(`${dir}/${e.name}`);
    } catch (e) { if (!(e instanceof Deno.errors.NotFound)) throw e; }
  }
  found.sort();
  const listed = [...TEST_FILES].sort();
  assert(JSON.stringify(found) === JSON.stringify(listed), `manifest mismatch:\n found  ${found}\n listed ${listed}`);
}, { denoOnly: true });
