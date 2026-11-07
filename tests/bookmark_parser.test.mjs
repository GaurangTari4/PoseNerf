import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(new URL("../posenerf/studio.py", import.meta.url), "utf8");
const parser = source.match(/function valid\(a\)\{.*?function loadBookmarks\(\)\{.*?return parsed\}/s)?.[0];
assert.ok(parser, "could not find the bookmark parser embedded in Studio");

function parseBookmarks(raw) {
  const select = { innerHTML: "", options: [], add(option) { this.options.push(option); } };
  const localStorage = { getItem: () => raw };
  const document = { querySelector: () => select };
  const Option = function Option(name, value) { return { name, value }; };
  const loadBookmarks = new Function("localStorage", "document", "Option", `const $=s=>document.querySelector(s),key=()=>"test";${parser};return loadBookmarks;`)(localStorage, document, Option);
  return { parsed: loadBookmarks(), select };
}

const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

test("corrupted bookmark JSON recovers without throwing", () => {
  const result = parseBookmarks("{not valid json");
  assert.deepEqual(result.parsed, []);
  assert.equal(result.select.options.length, 0);
});

test("missing, nonfinite, and invalid bookmark matrices are rejected", () => {
  const raw = JSON.stringify([
    {},
    { name: "missing-pose" },
    { name: "nan", pose: [...identity.slice(0, 15), null] },
    { name: "matrix-not-flat", pose: [identity, identity, identity, identity] },
    { name: "good", pose: identity },
  ]);
  const result = parseBookmarks(raw);
  assert.deepEqual(result.parsed.map((item) => item.name), ["good"]);
  assert.deepEqual(result.select.options.map((item) => item.name), ["good"]);
});
