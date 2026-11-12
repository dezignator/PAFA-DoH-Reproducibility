'use strict';
const test = require('node:test');
const assert = require('node:assert');
const c = require('./doh_wire_codec_v2.js');

test('RFC 8484 appendix example: www.example.com A', () => {
  // dns=AAABAAABAAAAAAAAA3d3dwdleGFtcGxlA2NvbQAAAQAB  (RFC 8484 sec 4.1.1)
  assert.strictEqual(c.b64url(c.buildQuery('www.example.com', 'A', 0)),
                     'AAABAAABAAAAAAAAA3d3dwdleGFtcGxlA2NvbQAAAQAB');
});

test('encode/decode round trip', () => {
  const d = c.decodeQuery(c.buildQuery('a.b.example.org.', 'TXT', 7));
  assert.deepStrictEqual([d.id, d.qname, d.qtype, d.qclass, d.qd], [7, 'a.b.example.org', 16, 1, 1]);
});

test('label limits', () => {
  assert.throws(() => c.encodeName('a'.repeat(64) + '.org'), RangeError);
  assert.throws(() => c.encodeName(Array(5).fill('a'.repeat(60)).join('.')), RangeError);   // > 255 octets
});

test('entropy values match the python reference', () => {
  assert.strictEqual(c.shannonEntropy('aaaa'), 0);
  assert.strictEqual(c.shannonEntropy('abcd'), 2);
  assert.strictEqual(c.shannonEntropy(''), 0);
});

test('subdomain label', () => {
  assert.strictEqual(c.subdomainLabel('a.b.example.org.'), 'a.b');
  assert.strictEqual(c.subdomainLabel('example.org'), '');
});

test('corpus reproducible from seed and high entropy', () => {
  const a = c.tunnelCorpus(20, 't.example.net', 42), b = c.tunnelCorpus(20, 't.example.net', 42);
  assert.deepStrictEqual(a, b);
  const mean = a.reduce((s, q) => s + c.shannonEntropy(c.subdomainLabel(q, 3)), 0) / a.length;
  assert.ok(mean > 4.0, `mean entropy ${mean}`);
});
