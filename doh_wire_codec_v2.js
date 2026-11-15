'use strict';
/*
 * doh_wire_codec_v2.js -- RFC 1035 wire encoding / RFC 8484 GET parameter, and
 * the subdomain-entropy feature of thesis Sec 4.7.1. No dependencies.
 *
 * Purpose in the artifact: deterministic generation of DoH query corpora for the
 * ns-3 / dnscat2-style testbed (Sec 4.25.1, 4.26.1) and an independent
 * implementation of the entropy coordinate, so the Python and JS feature values
 * can be diffed.
 *
 *   node --test js/doh_wire_codec_v2.test.js     (Node >= 18)
 */

const QTYPES = { A: 1, NS: 2, CNAME: 5, MX: 15, TXT: 16, AAAA: 28, NULL: 10 };

function encodeName(qname) {
  const labels = qname.replace(/\.$/, '').split('.').filter(Boolean);
  const parts = [];
  for (const l of labels) {
    const b = Buffer.from(l, 'ascii');
    if (b.length === 0 || b.length > 63) throw new RangeError(`label length ${b.length} out of 1..63`);
    parts.push(Buffer.from([b.length]), b);
  }
  parts.push(Buffer.from([0]));
  const out = Buffer.concat(parts);
  if (out.length > 255) throw new RangeError('encoded name exceeds 255 octets');
  return out;
}

function buildQuery(qname, qtype = 'A', id = 0) {
  // RFC 8484 sec 4.1: use ID 0 for cache friendliness.
  const hdr = Buffer.alloc(12);
  hdr.writeUInt16BE(id, 0);
  hdr.writeUInt16BE(0x0100, 2);          // RD
  hdr.writeUInt16BE(1, 4);               // QDCOUNT
  const q = Buffer.alloc(4);
  q.writeUInt16BE(QTYPES[qtype], 0);
  q.writeUInt16BE(1, 2);                 // IN
  return Buffer.concat([hdr, encodeName(qname), q]);
}

function b64url(buf) {
  return buf.toString('base64').replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

function dohGetPath(qname, qtype = 'A') {
  return '/dns-query?dns=' + b64url(buildQuery(qname, qtype, 0));
}

function decodeQuery(buf) {
  const id = buf.readUInt16BE(0);
  const flags = buf.readUInt16BE(2);
  const qd = buf.readUInt16BE(4);
  let o = 12;
  const labels = [];
  while (buf[o] !== 0) {
    const n = buf[o++];
    labels.push(buf.toString('ascii', o, o + n));
    o += n;
  }
  o += 1;
  const qtype = buf.readUInt16BE(o);
  const qclass = buf.readUInt16BE(o + 2);
  return { id, flags, qd, qname: labels.join('.'), qtype, qclass };
}

function shannonEntropy(s) {
  if (!s.length) return 0;
  const c = new Map();
  for (const ch of s) c.set(ch, (c.get(ch) || 0) + 1);
  let h = 0;
  for (const v of c.values()) { const p = v / s.length; h -= p * Math.log2(p); }
  return h;
}

function subdomainLabel(qname, zoneLabels = 2) {
  const p = qname.replace(/\.$/, '').split('.');
  return p.length > zoneLabels ? p.slice(0, p.length - zoneLabels).join('.') : '';
}

// xorshift32 so that corpora are reproducible from a seed, independent of Math.random
function rng(seed) {
  let x = seed >>> 0 || 1;
  return () => { x ^= x << 13; x >>>= 0; x ^= x >>> 17; x ^= x << 5; x >>>= 0; return x / 4294967296; };
}

function tunnelCorpus(n, zone, seed, labelLen = 40) {
  const r = rng(seed), A = 'abcdefghijklmnopqrstuvwxyz234567';
  const out = [];
  for (let i = 0; i < n; i++) {
    let l = '';
    for (let k = 0; k < labelLen; k++) l += A[Math.floor(r() * A.length)];
    out.push(`${l}.${zone}`);
  }
  return out;
}

module.exports = { buildQuery, dohGetPath, decodeQuery, encodeName, b64url,
                   shannonEntropy, subdomainLabel, tunnelCorpus, rng };

if (require.main === module) {
  const [zone = 't.example.net', n = '5', seed = '1'] = process.argv.slice(2);
  for (const q of tunnelCorpus(+n, zone, +seed)) {
    console.log(`${q}\t${shannonEntropy(subdomainLabel(q, zone.split('.').length)).toFixed(4)}\t${dohGetPath(q, 'TXT')}`);
  }
}
