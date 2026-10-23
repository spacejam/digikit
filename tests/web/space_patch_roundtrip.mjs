// Acceptance test: applying the space 17.1 patch to the stock file reproduces the target image byte for byte.
//   node tests/web/space_patch_roundtrip.mjs STOCK.syx SITE_DIR [TARGET.syx]
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';
const [stockPath, site, targetPath] = process.argv.slice(2);
if (!stockPath || !site) { console.error('usage: node space_patch_roundtrip.mjs STOCK.syx SITE_DIR [TARGET.syx]'); process.exit(2); }
const S = createRequire(import.meta.url)(path.resolve(site, 'patcher.js'));
const manifest = JSON.parse(readFileSync(path.join(site, 'space17.json'), 'utf8'));
const stock = new Uint8Array(readFileSync(stockPath));
const t = Date.now();
const out = await S.applyPatch(stock, new Uint8Array(readFileSync(path.join(site, manifest.patch))), m => console.log(' ', m));
const sha = await S.sha256hex(out);
console.log('output', out.length, sha, ((Date.now() - t) / 1000).toFixed(1) + 's');
let ok = sha === manifest.output_sha256 && out.length === manifest.output_bytes;
if (targetPath) ok = ok && Buffer.compare(Buffer.from(out), readFileSync(targetPath)) === 0;
console.log(ok ? 'PASS: byte-identical to target' : 'FAIL');
process.exit(ok ? 0 : 1);
