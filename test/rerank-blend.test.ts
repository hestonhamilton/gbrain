/**
 * Protects opt-in blended ordering while raw relevance remains calibrated.
 * Replacing blend scores with provider scores breaks the downstream assertions;
 * existing reranker coverage exercises only replacement order. Uses the existing
 * rerankerFn seam with synthetic rows and no provider calls.
 */
import { describe, expect, test } from 'bun:test';
import type { BrainEngine } from '../src/core/engine.ts';
import type { SearchResult } from '../src/core/types.ts';
import type { RerankResult } from '../src/core/ai/gateway.ts';
import { applyReranker, type RerankerOpts } from '../src/core/search/rerank.ts';
import { applyRankPriors, REDIRECT_STUB_PRIOR } from '../src/core/search/rank-priors.ts';
import { scoresFollowRank } from '../src/core/search/score-contract.ts';
import { applyFeedbackStage, feedbackMultiplier } from '../src/core/search/feedback-boost.ts';
import { formatResultExplain } from '../src/core/search/explain-formatter.ts';
import { buildModesReport, formatKnobValue } from '../src/core/search/modes-report.ts';
import {
  KNOB_CONFIG_KEY, MODE_BUNDLES, SEARCH_MODE_CONFIG_KEYS, attributeKnob,
  knobsHash, loadOverridesFromConfig, loadSearchModeConfig, resolveSearchMode,
} from '../src/core/search/mode.ts';

const provider: RerankResult[] = [
  { index: 2, relevanceScore: 0.9 },
  { index: 1, relevanceScore: 0.8 },
  { index: 0, relevanceScore: 0.1 },
];

function rows(scores = [10, 9, 0]): SearchResult[] {
  return scores.map((score, index) => ({
    slug: `notes/${String.fromCharCode(97 + index)}`, page_id: index + 1,
    title: `Note ${index}`, type: 'note', chunk_text: `Document ${index}`,
    chunk_source: 'compiled_truth', chunk_id: index + 1, chunk_index: 0,
    score, stale: false,
  }));
}

function rerank(input: SearchResult[] = rows(), opts: Partial<RerankerOpts> = {}) {
  return applyReranker('query', input, {
    enabled: true, topNIn: 3, topNOut: null, rerankerFn: async () => provider, ...opts,
  });
}

const slugs = (input: SearchResult[]) => input.map(r => r.slug);
const configEngine = (config: Record<string, string> = {}) => ({
  getConfig: async (key: string) => config[key] ?? null,
}) as unknown as BrainEngine;

describe('reranker score blend', () => {
  test('off preserves the exact replacement ordering and stamps', async () => {
    const input = rows([10, 9, 0, -1]);
    const expected = [
      { ...input[2], rerank_score: 0.9, reranker_delta: 2 },
      { ...input[1], rerank_score: 0.8, reranker_delta: 0 },
      { ...input[0], rerank_score: 0.1, reranker_delta: -2 },
      input[3],
    ];
    expect(await rerank(input)).toEqual(expected);
    expect(await rerank(rows([10, 9, 0, -1]), { blendAlpha: null })).toEqual(expected);
    for (const blendAlpha of [-0.1, 1.1, NaN, Infinity]) {
      expect(await rerank(rows([10, 9, 0, -1]), { blendAlpha })).toEqual(expected);
    }
  });

  test('alpha 0 keeps replacement order and raw provider relevance', async () => {
    const out = await rerank(rows(), { blendAlpha: 0 });
    expect(slugs(out)).toEqual(['notes/c', 'notes/b', 'notes/a']);
    expect(out.map(r => r.rerank_score)).toEqual([0.9, 0.8, 0.1]);
    expect(out.map(r => r.blend_score)).toEqual([1, 0.875, 0]);
    expect(out.map(r => r.reranker_delta)).toEqual([2, 0, -2]);
  });

  test('alpha 1 keeps first-stage order, including first-stage ties', async () => {
    for (const scores of [[10, 9, 0], [10, 10, 0]]) {
      const out = await rerank(rows(scores), { blendAlpha: 1 });
      expect(slugs(out)).toEqual(['notes/a', 'notes/b', 'notes/c']);
      expect(out.map(r => r.reranker_delta)).toEqual([0, 0, 0]);
      expect(out.map(r => r.rerank_score)).toEqual([0.1, 0.8, 0.9]);
    }
  });

  test('alpha 0.5 differs from both orders; blend ties keep original indices', async () => {
    const out = await rerank(rows(), { blendAlpha: 0.5 });
    expect(slugs(out)).toEqual(['notes/b', 'notes/a', 'notes/c']);
    expect(out.map(r => r.blend_score)).toEqual([0.8875, 0.5, 0.5]);
    expect(out.map(r => r.rerank_score)).toEqual([0.8, 0.1, 0.9]);
    expect(out.map(r => r.score)).toEqual([9, 10, 0]);
    expect(out.map(r => r.reranker_delta)).toEqual([1, -1, 0]);
  });

  test('constant first-stage or relevance terms normalize to zero', async () => {
    const first = await rerank(rows([4, 4, 4]), { blendAlpha: 0.5 });
    expect(first.map(r => r.blend_score)).toEqual([0.5, 0.4375, 0]);
    const tied = async () => [2, 0, 1].map(index => ({ index, relevanceScore: 0.7 }));
    const relevance = await rerank(rows(), { blendAlpha: 0.5, rerankerFn: tied });
    expect(slugs(relevance)).toEqual(['notes/a', 'notes/b', 'notes/c']);
    expect(relevance.map(r => r.blend_score)).toEqual([0.5, 0.45, 0]);
    const both = await rerank(rows([4, 4, 4]), { blendAlpha: 0.5, rerankerFn: tied });
    expect(slugs(both)).toEqual(['notes/a', 'notes/b', 'notes/c']);
    expect(both.map(r => r.blend_score)).toEqual([0, 0, 0]);
  });

  test('normalization excludes dropped rows and tail; truncation follows the final order', async () => {
    const opts: Partial<RerankerOpts> = {
      blendAlpha: 0.5, topNIn: 4,
      rerankerFn: async () => [{ index: 2, relevanceScore: 0.9 }, { index: 1, relevanceScore: 0.1 }],
    };
    const out = await rerank(rows([100, 9, 0, -100, -200]), opts);
    expect(slugs(out)).toEqual(['notes/b', 'notes/c', 'notes/a', 'notes/d', 'notes/e']);
    expect(out.slice(0, 2).map(r => r.blend_score)).toEqual([0.5, 0.5]);
    expect(out.slice(0, 2).map(r => r.reranker_delta)).toEqual([1, 1]);
    expect(out.slice(2).every(r => r.blend_score === undefined && r.rerank_score === undefined)).toBe(true);
    const truncated = await rerank(rows([100, 9, 0, -100, -200]), { ...opts, topNOut: 3 });
    expect(truncated).toEqual(out.slice(0, 3));
  });

  test('a single provider-scored row has zero blend and still preserves dropped rows', async () => {
    const out = await rerank(rows(), {
      blendAlpha: 0.5, rerankerFn: async () => [{ index: 1, relevanceScore: 0.8 }],
    });
    expect(slugs(out)).toEqual(['notes/b', 'notes/a', 'notes/c']);
    expect(out[0]).toMatchObject({ blend_score: 0, rerank_score: 0.8, reranker_delta: 1 });
  });

  test('rubric metadata disables blending for the batch', async () => {
    const rerankerFn: RerankerOpts['rerankerFn'] = async input => {
      input.onMeta?.({ model_resolved: 'test:rubric', score_semantics: 'rubric', input_tokens: 0, output_tokens: 0, latency_ms: 0, batches: 1 });
      return provider;
    };
    const out = await rerank(rows(), { blendAlpha: 1, rerankerFn });
    expect(slugs(out)).toEqual(['notes/c', 'notes/b', 'notes/a']);
    expect(out.every(r => r.rerank_score_kind === 'rubric' && r.blend_score === undefined)).toBe(true);
    expect(out).toEqual(await rerank(rows(), { rerankerFn }));
  });

  test('an existing rubric row disables blending for its batch', async () => {
    const input = rows();
    input[1].rerank_score_kind = 'rubric';
    const out = await rerank(input, { blendAlpha: 1 });
    expect(slugs(out)).toEqual(['notes/c', 'notes/b', 'notes/a']);
    expect(out.every(r => r.blend_score === undefined)).toBe(true);
  });
});

describe('blended ordering consumers', () => {
  test('rank priors multiply the blend and compare it with unchanged blended rows', async () => {
    const out = await rerank(rows(), { blendAlpha: 0.5 });
    out[0].redirect_to = 'notes/target';
    const ranked = applyRankPriors(out, 'query');
    expect(slugs(ranked)).toEqual(['notes/a', 'notes/c', 'notes/b']);
    expect(ranked[2]).toMatchObject({
      rank_score: 0.8875 * REDIRECT_STUB_PRIOR, rank_prior: REDIRECT_STUB_PRIOR,
      rerank_score: 0.8, blend_score: 0.8875,
    });
    expect(scoresFollowRank(ranked).map(r => r.score)).toEqual([0.5, 0.5, 0.44375]);
  });

  test('scoresFollowRank exposes blend scores and preserves the blended order', async () => {
    const out = await rerank(rows(), { blendAlpha: 0.5 });
    const scored = scoresFollowRank(out);
    expect(slugs(scored)).toEqual(['notes/b', 'notes/a', 'notes/c']);
    expect(scored.map(r => r.score)).toEqual([0.8875, 0.5, 0.5]);
    expect(scored.map(r => r.fused_score)).toEqual([9, 10, 0]);
    expect(scored.map(r => r.rerank_score)).toEqual([0.8, 0.1, 0.9]);
  });

  test('feedback boosts the blend, leaving raw relevance and blended stamps intact', async () => {
    const engine = {
      getConfig: async (key: string) => key === 'feedback.enabled' ? 'true' : null,
      executeRaw: async () => [1, 2, 3].map(id => ({ id, content_hash: 'synthetic', weight: id === 1 ? 1 : 0.5 })),
    } as unknown as BrainEngine;
    const out = await rerank(rows(), { blendAlpha: 0.5 });
    const boosted = await applyFeedbackStage(engine, out, { reranked: true });
    expect(slugs(boosted)).toEqual(['notes/b', 'notes/a', 'notes/c']);
    expect(boosted[1]).toMatchObject({ rerank_score: 0.1, blend_score: 0.5, feedback_boost: feedbackMultiplier(1, 0.1) });
  });

  test('explain shows blend beside raw rerank relevance only when present', async () => {
    const [blended] = await rerank(rows(), { blendAlpha: 0.5 });
    expect(formatResultExplain(blended, 1)).toContain('rerank score=0.8 blend score=0.8875');
    const [plain] = await rerank();
    expect(formatResultExplain(plain, 1)).not.toContain('blend score=');
  });
});

describe('search.reranker.blend_alpha configuration', () => {
  test('parses valid endpoints and midpoint', () => {
    for (const raw of ['0', '0.5', '1', ' 0.25 ']) {
      const overrides = loadOverridesFromConfig({ 'search.reranker.blend_alpha': raw });
      expect(overrides.reranker_blend_alpha).toBe(Number(raw));
      expect(resolveSearchMode({ overrides }).reranker_blend_alpha).toBe(Number(raw));
    }
  });

  test('out-of-range, off and invalid inputs disable blending', () => {
    for (const raw of ['-0.1', '1.1', 'off', 'OFF', '', ' ', 'NaN', 'Infinity', '0.5junk']) {
      const overrides = loadOverridesFromConfig({ 'search.reranker.blend_alpha': raw });
      expect(resolveSearchMode({ overrides }).reranker_blend_alpha).toBeNull();
    }
  });

  test('unset is off in every mode and absent from all bundles', () => {
    for (const mode of ['conservative', 'balanced', 'tokenmax'] as const) {
      expect(resolveSearchMode({ mode }).reranker_blend_alpha).toBeNull();
      expect(MODE_BUNDLES[mode]).not.toHaveProperty('reranker_blend_alpha');
    }
  });

  test('per-call alpha and explicit off override brain configuration', () => {
    const overrides = { reranker_blend_alpha: 0.5 };
    expect(resolveSearchMode({ overrides, perCall: { reranker_blend_alpha: 0 } }).reranker_blend_alpha).toBe(0);
    expect(resolveSearchMode({ overrides, perCall: { reranker_blend_alpha: null } }).reranker_blend_alpha).toBeNull();
  });

  test('registry and snapshot carry the config key to mode resolution', async () => {
    expect(KNOB_CONFIG_KEY.reranker_blend_alpha).toBe('search.reranker.blend_alpha');
    expect(SEARCH_MODE_CONFIG_KEYS).toContain('search.reranker.blend_alpha');
    const input = await loadSearchModeConfig(configEngine({ 'search.reranker.blend_alpha': '0.5' }));
    const resolved = resolveSearchMode(input);
    expect(resolved.reranker_blend_alpha).toBe(0.5);
    expect(attributeKnob('reranker_blend_alpha', input, resolved)).toMatchObject({
      value: 0.5, source: 'override', source_detail: 'config: search.reranker.blend_alpha',
    });
  });

  test('modes report exposes the blend knob and labels its off default', async () => {
    const report = await buildModesReport(configEngine({ 'search.reranker.blend_alpha': '0.5' }));
    expect(report.resolved.reranker_blend_alpha).toMatchObject({ value: 0.5, source: 'override' });
    const off = await buildModesReport(configEngine());
    expect(off.resolved.reranker_blend_alpha).toMatchObject({ value: null, source_detail: 'default: off' });
    expect(formatKnobValue('reranker_blend_alpha', null)).toBe('off (null)');
  });

  test('blend-on hashes distinguish alpha values while off retains the legacy hash', () => {
    const off = resolveSearchMode({});
    const legacy = { ...off };
    delete legacy.reranker_blend_alpha;
    expect(knobsHash(off)).toBe(knobsHash(legacy));
    const hashes = [0, 0.5, 1].map(reranker_blend_alpha => knobsHash({ ...off, reranker_blend_alpha }));
    expect(new Set([...hashes, knobsHash(off)]).size).toBe(4);
  });
});
