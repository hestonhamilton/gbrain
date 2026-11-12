/**
 * `thinking: 'off'` on the Claude 5.5 models (#7567, #8317; local patch on
 * heston/v0.60.682-claude55).
 *
 * Sonnet 5.5 rejects `thinking: {type:'disabled'}`; its lowest setting is
 * `between_tools`. Haiku 5.5 accepts `disabled` only at effort high or below.
 * Opus 5.5 cannot turn thinking off at all, so its floor is adaptive thinking
 * at effort low, and it keeps reasoning (`disables: false`) so the output cap
 * gets the thinking headroom.
 *
 * The pinned @ai-sdk/anthropic (3.0.125) silently rewrites `disabled` for
 * Sonnet and Opus 5.5 (to `between_tools`, or by dropping the field), so the
 * old row never reached the API as `disabled` on those two. What it did send
 * was no effort at all and a judge-sized cap on a model that still thinks.
 * The wire tests below assert the request body that leaves gbrain.
 */

import { afterAll, beforeAll, beforeEach, describe, expect, test } from 'bun:test';
import { applyThinkingOff, thinkingOffControl, thinkingOffMaxOutputTokens } from '../../src/core/ai/thinking-off.ts';
import { chat, configureGateway, resetGateway } from '../../src/core/ai/gateway.ts';

describe('thinking off on Claude 5.5 (#7567, #8317)', () => {
  test('Sonnet 5.5 uses between_tools at effort low', () => {
    expect(thinkingOffControl('anthropic:claude-sonnet-5-5')).toEqual({
      set: { namespace: 'anthropic', options: { thinking: { type: 'between_tools' }, effort: 'low' } },
      disables: true,
    });
  });

  test('Haiku 5.5 uses disabled at effort low', () => {
    expect(thinkingOffControl('anthropic:claude-haiku-5-5')).toEqual({
      set: { namespace: 'anthropic', options: { thinking: { type: 'disabled' }, effort: 'low' } },
      disables: true,
    });
  });

  test('Opus 5.5 never sends disabled: adaptive at effort low, keeps reasoning', () => {
    expect(thinkingOffControl('anthropic:claude-opus-5-5')).toEqual({
      set: { namespace: 'anthropic', options: { thinking: { type: 'adaptive' }, effort: 'low' } },
      disables: false,
    });
  });

  test('dated 5.5 ids match too', () => {
    expect(thinkingOffControl('anthropic:claude-opus-5-5-20261001')?.disables).toBe(false);
  });

  test('Opus 5.5 gets thinking headroom on its output cap', () => {
    expect(thinkingOffMaxOutputTokens('anthropic:claude-opus-5-5', 1000, false, 32000)).toBe(32000);
  });

  test('older Claude is unchanged', () => {
    expect(thinkingOffControl('anthropic:claude-sonnet-4-6')).toEqual({
      set: { namespace: 'anthropic', options: { thinking: { type: 'disabled' } } },
      disables: true,
    });
  });

  test('DeepSeek is unchanged', () => {
    expect(thinkingOffControl('deepseek:deepseek-v4')).toEqual({
      set: { namespace: 'deepseek', options: { thinking: { type: 'disabled' } } },
      disables: true,
    });
  });

  test('a configured enabled budget is replaced, siblings kept', () => {
    const po: Record<string, any> = { anthropic: { thinking: { type: 'enabled', budgetTokens: 9000 }, cacheControl: { type: 'ephemeral' } } };
    applyThinkingOff(po, 'anthropic:claude-opus-5-5');
    expect(po.anthropic).toEqual({ thinking: { type: 'adaptive' }, effort: 'low', cacheControl: { type: 'ephemeral' } });
  });
});

describe('thinking off on Claude 5.5: the request body that leaves gbrain', () => {
  let server: ReturnType<typeof Bun.serve>;
  let bodies: any[] = [];

  beforeAll(() => {
    server = Bun.serve({
      port: 0,
      hostname: '127.0.0.1',
      async fetch(req) {
        const body = await req.json();
        bodies.push(body);
        return Response.json({
          id: 'msg_stub',
          type: 'message',
          role: 'assistant',
          model: body.model,
          content: [{ type: 'text', text: 'ok' }],
          stop_reason: 'end_turn',
          usage: { input_tokens: 1, output_tokens: 1 },
        });
      },
    });
  });

  afterAll(() => {
    server.stop(true);
    resetGateway();
  });

  beforeEach(() => {
    bodies = [];
    resetGateway();
  });

  async function offBody(model: string): Promise<any> {
    configureGateway({
      chat_model: `anthropic:${model}`,
      expansion_model: `anthropic:${model}`,
      env: { ANTHROPIC_API_KEY: 'stub-key', ANTHROPIC_BASE_URL: `http://127.0.0.1:${server.port}/v1` },
    });
    await chat({ model: `anthropic:${model}`, messages: [{ role: 'user', content: 'hi' }], thinking: 'off', maxTokens: 16 });
    expect(bodies).toHaveLength(1);
    return bodies[0];
  }

  test('Opus 5.5: adaptive at effort low with the thinking headroom', async () => {
    const body = await offBody('claude-opus-5-5');
    expect(body.thinking).toEqual({ type: 'adaptive' });
    expect(body.output_config?.effort).toBe('low');
    expect(body.max_tokens).toBe(32000);
  });

  test('Sonnet 5.5: between_tools at effort low, requested cap', async () => {
    const body = await offBody('claude-sonnet-5-5');
    expect(body.thinking).toEqual({ type: 'between_tools' });
    expect(body.output_config?.effort).toBe('low');
    expect(body.max_tokens).toBe(16);
  });

  test('Haiku 5.5: disabled at effort low, requested cap', async () => {
    const body = await offBody('claude-haiku-5-5');
    expect(body.thinking).toEqual({ type: 'disabled' });
    expect(body.output_config?.effort).toBe('low');
    expect(body.max_tokens).toBe(16);
  });
});
