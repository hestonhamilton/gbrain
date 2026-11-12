import type { SearchResult } from '../types.ts';

/** Ordering signal; raw provider relevance stays available for calibration. */
export function rerankOrderScore(row: Pick<SearchResult, 'blend_score' | 'rerank_score'>): number | undefined {
  return row.blend_score ?? row.rerank_score;
}
