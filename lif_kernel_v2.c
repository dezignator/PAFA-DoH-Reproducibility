/* lif_kernel_v2.c -- software LIF population (thesis Eqs. 44-45), C99.
 *
 * Euler-Maruyama update, xorshift64* + Box-Muller noise. Used to (a) cross-check
 * the numpy reference in pafa_v2/snn_v2.py and (b) get a CPU baseline for the
 * 128 x 1024 neuron layout of Sec 4.24.1. It is a CPU emulation: the Loihi
 * latency/energy of Table 13 are not obtainable from this program.
 *
 * build:  cc -O2 -std=c99 -Wall -Wextra -o lif_kernel_v2 lif_kernel_v2.c -lm
 * run  :  ./lif_kernel_v2 --selftest
 *         ./lif_kernel_v2 --bench 128 1024 20000
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    float tau_m, v_rest, v_th, v_reset, tau_ref, sigma;
} lif_params;

static uint64_t rng_state = 0x9E3779B97F4A7C15ULL;
static uint64_t xorshift64s(void) {
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return rng_state * 0x2545F4914F6CDD1DULL;
}
static double u01(void) { return ((xorshift64s() >> 11) + 0.5) * (1.0 / 9007199254740992.0); }
static float randn(void) {
    double u1 = u01(), u2 = u01();
    return (float)(sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2));
}

/* one step for n neurons; returns number of spikes */
static size_t lif_step(const lif_params *p, float dt, size_t n, const float *I,
                       float *V, float *ref, uint8_t *spk) {
    size_t count = 0;
    const float a = dt / p->tau_m;
    const float b = p->sigma > 0.f ? p->sigma * sqrtf(a) : 0.f;
    for (size_t i = 0; i < n; i++) {
        spk[i] = 0;
        if (ref[i] > 0.f) {
            ref[i] -= dt;
            V[i] = p->v_reset;
            continue;
        }
        float v = V[i] + a * (-(V[i] - p->v_rest) + I[i]);
        if (b > 0.f) v += b * randn();
        if (v >= p->v_th) {
            spk[i] = 1;
            count++;
            v = p->v_reset;
            ref[i] = p->tau_ref;
        }
        V[i] = v;
    }
    return count;
}

static int selftest(void) {
    lif_params p = {20.f, 0.f, 15.f, 0.f, 0.f, 0.f};
    const float dt = 0.01f, I = 20.f;
    float V = 0.f, ref = 0.f, Iv = I;
    uint8_t s = 0;
    long steps = 50000, last = -1, nisi = 0;
    double sum = 0.0;
    for (long t = 0; t < steps; t++) {
        lif_step(&p, dt, 1, &Iv, &V, &ref, &s);
        if (s) {
            if (last >= 0) { sum += (t - last) * dt; nisi++; }
            last = t;
        }
    }
    double expect = p.tau_m * log((I - (p.v_reset - p.v_rest)) / (I - (p.v_th - p.v_rest)));
    double got = nisi ? sum / nisi : -1.0;
    printf("noiseless ISI: got %.4f ms, analytic %.4f ms, spikes %ld\n", got, expect, nisi + 1);
    if (nisi < 5 || fabs(got - expect) > 0.05) { puts("FAIL"); return 1; }
    float Isub = 10.f; V = 0.f; ref = 0.f; long cnt = 0; p.sigma = 0.f;
    for (long t = 0; t < steps; t++) { lif_step(&p, dt, 1, &Isub, &V, &ref, &s); cnt += s; }
    if (cnt != 0) { puts("FAIL: subthreshold input fired"); return 1; }
    puts("PASS");
    return 0;
}

static int bench(size_t cores, size_t per_core, long steps) {
    size_t n = cores * per_core;
    lif_params p = {20.f, 0.f, 15.f, 0.f, 2.f, 0.5f};
    float *I = malloc(n * sizeof *I), *V = calloc(n, sizeof *V), *ref = calloc(n, sizeof *ref);
    uint8_t *spk = malloc(n);
    if (!I || !V || !ref || !spk) return 2;
    for (size_t i = 0; i < n; i++) I[i] = 12.f + 10.f * (float)u01();
    size_t total = 0;
    clock_t c0 = clock();
    for (long t = 0; t < steps; t++) total += lif_step(&p, 0.1f, n, I, V, ref, spk);
    double sec = (double)(clock() - c0) / CLOCKS_PER_SEC;
    printf("neurons=%zu steps=%ld spikes=%zu time=%.3fs  %.2f ns/neuron-step\n",
           n, steps, total, sec, 1e9 * sec / ((double)n * steps));
    free(I); free(V); free(ref); free(spk);
    return 0;
}

int main(int argc, char **argv) {
    if (argc >= 2 && !strcmp(argv[1], "--selftest")) return selftest();
    if (argc >= 5 && !strcmp(argv[1], "--bench"))
        return bench(strtoul(argv[2], 0, 10), strtoul(argv[3], 0, 10), atol(argv[4]));
    fprintf(stderr, "usage: %s --selftest | --bench CORES NEURONS_PER_CORE STEPS\n", argv[0]);
    return 64;
}
