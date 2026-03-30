import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch


def _load_nanomind_module():
    transformers = types.ModuleType("transformers")

    class _PretrainedConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class _PreTrainedModel(torch.nn.Module):
        def __init__(self, config=None):
            super().__init__()
            self.config = config

    class _CausalLMOutputWithPast:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    transformers.PretrainedConfig = _PretrainedConfig
    transformers.PreTrainedModel = _PreTrainedModel
    transformers.GenerationMixin = type("GenerationMixin", (), {})

    activations = types.ModuleType("transformers.activations")
    activations.ACT2FN = {"silu": torch.nn.functional.silu}

    modeling_outputs = types.ModuleType("transformers.modeling_outputs")
    modeling_outputs.CausalLMOutputWithPast = _CausalLMOutputWithPast

    sys.modules["transformers"] = transformers
    sys.modules["transformers.activations"] = activations
    sys.modules["transformers.modeling_outputs"] = modeling_outputs

    module_path = Path(__file__).resolve().parent / "model" / "NanoMind.py"
    spec = importlib.util.spec_from_file_location("nanomind_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _tiny_config(module, **overrides):
    config = module.NanoMindConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        dropout=0.0,
        flash_attention=True,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


class TestCoreOps(unittest.TestCase):
    def test_rmsnorm_matches_manual_formula(self):
        module = _load_nanomind_module()
        rmsnorm = module.RMSNorm(dim=4, eps=1e-6)
        with torch.no_grad():
            rmsnorm.weight.copy_(torch.tensor([1.0, 2.0, 0.5, 1.5]))

        x = torch.tensor(
            [[[1.0, 2.0, 3.0, 4.0], [2.0, 0.0, 2.0, 0.0]]], dtype=torch.float32
        )
        y = rmsnorm(x)

        expected = x * torch.rsqrt((x**2).mean(-1, keepdim=True) + 1e-6)
        expected = expected * rmsnorm.weight

        self.assertTrue(torch.allclose(y, expected, atol=1e-6, rtol=1e-6))

    def test_repeat_kv_repeats_heads_without_changing_values(self):
        module = _load_nanomind_module()
        x = torch.arange(2 * 3 * 2 * 4, dtype=torch.float32).view(2, 3, 2, 4)

        repeated = module.repeat_kv(x, n_rep=3)

        self.assertEqual(repeated.shape, (2, 3, 6, 4))
        self.assertTrue(torch.equal(repeated[:, :, 0], x[:, :, 0]))
        self.assertTrue(torch.equal(repeated[:, :, 1], x[:, :, 0]))
        self.assertTrue(torch.equal(repeated[:, :, 2], x[:, :, 0]))
        self.assertTrue(torch.equal(repeated[:, :, 3], x[:, :, 1]))
        self.assertTrue(torch.equal(repeated[:, :, 4], x[:, :, 1]))
        self.assertTrue(torch.equal(repeated[:, :, 5], x[:, :, 1]))


class TestFeedForward(unittest.TestCase):
    def test_feedforward_matches_swiglu_formula(self):
        module = _load_nanomind_module()
        config = _tiny_config(module, hidden_size=8, intermediate_size=16)
        ffn = module.FeedForward(config).eval()
        x = torch.randn(2, 3, 8, dtype=torch.float32)

        y = ffn(x)
        expected = ffn.down_proj(
            torch.nn.functional.silu(ffn.gate_proj(x)) * ffn.up_proj(x)
        )

        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.allclose(y, expected, atol=1e-6, rtol=1e-6))

    def test_feedforward_auto_sets_intermediate_size(self):
        module = _load_nanomind_module()
        config = module.NanoMindConfig(hidden_size=512, intermediate_size=None)

        ffn = module.FeedForward(config)

        self.assertEqual(config.intermediate_size, 1408)
        self.assertEqual(ffn.gate_proj.out_features, 1408)
        self.assertEqual(ffn.up_proj.out_features, 1408)
        self.assertEqual(ffn.down_proj.in_features, 1408)


class TestRoPE(unittest.TestCase):
    def test_precompute_freqs_shapes_and_first_position(self):
        module = _load_nanomind_module()

        cos, sin = module.precompute_freqs(dim=4, end=6, rope_base=10000.0)

        self.assertEqual(cos.shape, (6, 4))
        self.assertEqual(sin.shape, (6, 4))
        self.assertTrue(torch.allclose(cos[0], torch.ones(4), atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(sin[0], torch.zeros(4), atol=1e-6, rtol=1e-6))

    def test_apply_rotary_is_identity_when_cos_one_sin_zero(self):
        module = _load_nanomind_module()
        q = torch.randn(2, 3, 4, 6, dtype=torch.float32)
        k = torch.randn(2, 3, 2, 6, dtype=torch.float32)
        cos = torch.ones(3, 6, dtype=torch.float32)
        sin = torch.zeros(3, 6, dtype=torch.float32)

        q_emb, k_emb = module.apply_rotary_pos_emb(q, k, cos, sin)

        self.assertTrue(torch.allclose(q_emb, q, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(k_emb, k, atol=1e-6, rtol=1e-6))

    def test_apply_rotary_preserves_l2_norm(self):
        module = _load_nanomind_module()
        q = torch.randn(2, 3, 4, 6, dtype=torch.float32)
        k = torch.randn(2, 3, 2, 6, dtype=torch.float32)
        cos, sin = module.precompute_freqs(dim=6, end=3, rope_base=10000.0)

        q_emb, k_emb = module.apply_rotary_pos_emb(q, k, cos, sin)

        self.assertTrue(
            torch.allclose(
                q_emb.norm(dim=-1), q.norm(dim=-1), atol=1e-5, rtol=1e-4
            )
        )
        self.assertTrue(
            torch.allclose(
                k_emb.norm(dim=-1), k.norm(dim=-1), atol=1e-5, rtol=1e-4
            )
        )


class TestGroupQueryAttention(unittest.TestCase):
    def test_prefill_forward_preserves_shape(self):
        module = _load_nanomind_module()
        config = _tiny_config(module)
        attn = module.GroupQueryAttention(config).eval()

        x = torch.randn(2, 3, 16, dtype=torch.float32)
        cos = torch.ones(3, 4, dtype=torch.float32)
        sin = torch.zeros(3, 4, dtype=torch.float32)

        y, past_kv = attn(x, (cos, sin))

        self.assertEqual(y.shape, x.shape)
        self.assertEqual(y.dtype, x.dtype)
        self.assertEqual(past_kv[0].shape, (2, 3, 2, 4))
        self.assertEqual(past_kv[1].shape, (2, 3, 2, 4))

    def test_decode_forward_with_kv_cache_preserves_shape(self):
        module = _load_nanomind_module()
        config = _tiny_config(module)
        attn = module.GroupQueryAttention(config).eval()

        cache_k = torch.randn(2, 5, 2, 4, dtype=torch.float32)
        cache_v = torch.randn(2, 5, 2, 4, dtype=torch.float32)
        x = torch.randn(2, 1, 16, dtype=torch.float32)
        cos = torch.ones(1, 4, dtype=torch.float32)
        sin = torch.zeros(1, 4, dtype=torch.float32)

        y, past_kv = attn(x, (cos, sin), kv_cache=(cache_k, cache_v))

        self.assertEqual(y.shape, x.shape)
        self.assertEqual(past_kv[0].shape, (2, 6, 2, 4))
        self.assertEqual(past_kv[1].shape, (2, 6, 2, 4))

    def test_prefill_last_token_matches_decode_with_kv_cache(self):
        module = _load_nanomind_module()
        config = _tiny_config(module)
        attn = module.GroupQueryAttention(config).eval()

        x_full = torch.randn(2, 4, 16, dtype=torch.float32)
        cos_full = torch.ones(4, 4, dtype=torch.float32)
        sin_full = torch.zeros(4, 4, dtype=torch.float32)
        full_out, _ = attn(x_full, (cos_full, sin_full))

        x_prefix = x_full[:, :3, :]
        _, prefix_kv = attn(x_prefix, (cos_full[:3], sin_full[:3]))

        x_last = x_full[:, 3:, :]
        decode_out, decode_kv = attn(
            x_last, (cos_full[3:], sin_full[3:]), kv_cache=prefix_kv
        )

        self.assertEqual(decode_out.shape, (2, 1, 16))
        self.assertEqual(decode_kv[0].shape, (2, 4, 2, 4))
        self.assertEqual(decode_kv[1].shape, (2, 4, 2, 4))
        self.assertTrue(
            torch.allclose(full_out[:, -1:, :], decode_out, atol=1e-5, rtol=1e-4)
        )

    def test_attention_mask_path_runs_in_non_flash_mode(self):
        module = _load_nanomind_module()
        config = _tiny_config(module, flash_attention=False)
        attn = module.GroupQueryAttention(config).eval()

        x = torch.randn(2, 4, 16, dtype=torch.float32)
        cos = torch.ones(4, 4, dtype=torch.float32)
        sin = torch.zeros(4, 4, dtype=torch.float32)
        attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.float32)

        masked_out, _ = attn(x, (cos, sin), attention_mask=attention_mask)
        unmasked_out, _ = attn(x, (cos, sin), attention_mask=torch.ones_like(attention_mask))

        self.assertEqual(masked_out.shape, x.shape)
        self.assertFalse(torch.allclose(masked_out, unmasked_out))


class TestBlockAttentionResidual(unittest.TestCase):
    def test_block_attention_residual_matches_uniform_average_when_logits_zero(self):
        module = _load_nanomind_module()
        config = _tiny_config(module)
        residual = module.BlcokAttentionResidual(config)
        with torch.no_grad():
            residual.proj.weight.zero_()

        block_1 = torch.full((2, 3, 16), 1.0)
        block_2 = torch.full((2, 3, 16), 3.0)
        partial_block = torch.full((2, 3, 16), 5.0)

        out = residual([block_1, block_2], partial_block)

        expected = torch.full((2, 3, 16), 3.0)
        self.assertTrue(torch.allclose(out, expected, atol=1e-6, rtol=1e-6))

    def test_block_attention_residual_supports_single_partial_block(self):
        module = _load_nanomind_module()
        config = _tiny_config(module)
        residual = module.BlcokAttentionResidual(config)
        partial_block = torch.randn(2, 3, 16, dtype=torch.float32)

        out = residual([], partial_block)

        self.assertEqual(out.shape, partial_block.shape)


class TestNanoMindBlock(unittest.TestCase):
    def test_block_forward_runs_with_empty_history(self):
        module = _load_nanomind_module()
        config = _tiny_config(module)
        layer = module.NanoMindBlock(layer_id=0, config=config).eval()

        hidden_states = torch.randn(2, 4, 16, dtype=torch.float32)
        cos = torch.ones(4, 4, dtype=torch.float32)
        sin = torch.zeros(4, 4, dtype=torch.float32)

        blocks, partial_block, present_kv = layer(
            hidden_states,
            blocks=[],
            partial_block=None,
            position_embeddings=(cos, sin),
        )

        self.assertGreaterEqual(len(blocks), 1)
        self.assertEqual(partial_block.shape, hidden_states.shape)
        self.assertEqual(present_kv[0].shape, (2, 4, 2, 4))
        self.assertEqual(present_kv[1].shape, (2, 4, 2, 4))

    def test_blocksize_four_only_flushes_every_two_layers(self):
        module = _load_nanomind_module()
        config = _tiny_config(module, attn_res_block_size=4)
        layer0 = module.NanoMindBlock(layer_id=0, config=config).eval()
        layer1 = module.NanoMindBlock(layer_id=1, config=config).eval()

        hidden_states = torch.randn(2, 4, 16, dtype=torch.float32)
        cos = torch.ones(4, 4, dtype=torch.float32)
        sin = torch.zeros(4, 4, dtype=torch.float32)

        blocks, partial_block, _ = layer0(
            hidden_states,
            blocks=[],
            partial_block=None,
            position_embeddings=(cos, sin),
        )
        self.assertEqual(len(blocks), 0)
        self.assertIsNotNone(partial_block)

        blocks, partial_block, _ = layer1(
            partial_block,
            blocks=blocks,
            partial_block=partial_block,
            position_embeddings=(cos, sin),
        )
        self.assertEqual(len(blocks), 1)
        self.assertIsNotNone(partial_block)


class TestNanoMindModel(unittest.TestCase):
    def test_model_init_weights_applies_explicit_scheme(self):
        module = _load_nanomind_module()
        config = _tiny_config(module)
        model = module.NanoMindModel(config)

        with torch.no_grad():
            model.embed_tokens.weight.zero_()
            model.layers[0].self_attention.o_proj.weight.fill_(1.0)
            model.layers[0].mlp.down_proj.weight.fill_(1.0)
            model.layers[0].attn_res.proj.weight.fill_(1.0)
            model.layers[0].mlp_res.proj.weight.fill_(1.0)

        model.init_weights()

        self.assertFalse(torch.allclose(model.embed_tokens.weight, torch.zeros_like(model.embed_tokens.weight)))
        self.assertTrue(torch.allclose(model.layers[0].self_attention.o_proj.weight, torch.zeros_like(model.layers[0].self_attention.o_proj.weight)))
        self.assertTrue(torch.allclose(model.layers[0].mlp.down_proj.weight, torch.zeros_like(model.layers[0].mlp.down_proj.weight)))
        self.assertTrue(torch.allclose(model.layers[0].attn_res.proj.weight, torch.zeros_like(model.layers[0].attn_res.proj.weight)))
        self.assertTrue(torch.allclose(model.layers[0].mlp_res.proj.weight, torch.zeros_like(model.layers[0].mlp_res.proj.weight)))

    def test_model_forward_returns_hidden_states_and_cache(self):
        module = _load_nanomind_module()
        config = _tiny_config(module)
        model = module.NanoMindModel(config).eval()
        input_ids = torch.randint(0, config.vocab_size, (2, 4))

        hidden_states, presents, aux_loss = model(input_ids=input_ids, use_cache=True)

        self.assertEqual(hidden_states.shape, (2, 4, config.hidden_size))
        self.assertEqual(len(presents), config.num_hidden_layers)
        self.assertEqual(presents[0][0].shape, (2, 4, config.num_key_value_heads, 4))
        self.assertEqual(aux_loss, 0.0)

    def test_model_decode_with_cache_preserves_shapes(self):
        module = _load_nanomind_module()
        config = _tiny_config(module)
        model = module.NanoMindModel(config).eval()

        prefix_ids = torch.randint(0, config.vocab_size, (2, 3))
        _, prefix_cache, _ = model(input_ids=prefix_ids, use_cache=True)

        next_ids = torch.randint(0, config.vocab_size, (2, 1))
        hidden_states, next_cache, _ = model(
            input_ids=next_ids, past_key_values=prefix_cache, use_cache=True
        )

        self.assertEqual(hidden_states.shape, (2, 1, config.hidden_size))
        self.assertEqual(len(next_cache), config.num_hidden_layers)
        self.assertEqual(next_cache[0][0].shape, (2, 4, config.num_key_value_heads, 4))

    def test_model_last_token_matches_decode_with_cache(self):
        module = _load_nanomind_module()
        config = _tiny_config(module)
        model = module.NanoMindModel(config).eval()

        input_ids = torch.randint(0, config.vocab_size, (2, 5))
        full_hidden_states, _, _ = model(input_ids=input_ids, use_cache=True)

        prefix_ids = input_ids[:, :4]
        _, prefix_cache, _ = model(input_ids=prefix_ids, use_cache=True)
        next_ids = input_ids[:, 4:]
        decode_hidden_states, _, _ = model(
            input_ids=next_ids, past_key_values=prefix_cache, use_cache=True
        )

        self.assertTrue(
            torch.allclose(
                full_hidden_states[:, -1:, :],
                decode_hidden_states,
                atol=1e-5,
                rtol=1e-4,
            )
        )

    def test_model_forward_respects_attention_mask_in_non_flash_mode(self):
        module = _load_nanomind_module()
        config = _tiny_config(module, flash_attention=False)
        model = module.NanoMindModel(config).eval()

        input_ids = torch.randint(0, config.vocab_size, (2, 4))
        masked = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.float32)
        unmasked = torch.ones_like(masked)

        masked_hidden_states, _, _ = model(input_ids=input_ids, attention_mask=masked)
        unmasked_hidden_states, _, _ = model(input_ids=input_ids, attention_mask=unmasked)

        self.assertEqual(masked_hidden_states.shape, (2, 4, config.hidden_size))
        self.assertFalse(torch.allclose(masked_hidden_states, unmasked_hidden_states))


class TestNanoMindForCausalLM(unittest.TestCase):
    def test_causallm_init_weights_keeps_tied_embeddings(self):
        module = _load_nanomind_module()
        config = _tiny_config(module)
        model = module.NanoMindForCausalLM(config)

        model.init_weights()

        self.assertIs(model.model.embed_tokens.weight, model.lm_head.weight)

    def test_lm_head_shares_weights_with_embeddings(self):
        module = _load_nanomind_module()
        config = _tiny_config(module)
        model = module.NanoMindForCausalLM(config)

        self.assertIs(model.model.embed_tokens.weight, model.lm_head.weight)

    def test_causallm_forward_returns_logits_hidden_states_and_loss(self):
        module = _load_nanomind_module()
        config = _tiny_config(module)
        model = module.NanoMindForCausalLM(config).eval()
        input_ids = torch.randint(0, config.vocab_size, (2, 5))
        labels = input_ids.clone()

        output = model(input_ids=input_ids, labels=labels, use_cache=True)

        self.assertEqual(output.logits.shape, (2, 5, config.vocab_size))
        self.assertEqual(output.hidden_states.shape, (2, 5, config.hidden_size))
        self.assertEqual(len(output.past_key_values), config.num_hidden_layers)
        self.assertEqual(output.loss.ndim, 0)
        self.assertEqual(output.aux_loss, 0.0)

    def test_causallm_loss_matches_manual_cross_entropy(self):
        module = _load_nanomind_module()
        config = _tiny_config(module)
        model = module.NanoMindForCausalLM(config).eval()
        input_ids = torch.randint(0, config.vocab_size, (2, 5))
        labels = input_ids.clone()

        output = model(input_ids=input_ids, labels=labels)
        shift_logits = output.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        expected_loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        self.assertTrue(torch.allclose(output.loss, expected_loss, atol=1e-6, rtol=1e-6))

    def test_logits_to_keep_limits_output_length(self):
        module = _load_nanomind_module()
        config = _tiny_config(module)
        model = module.NanoMindForCausalLM(config).eval()
        input_ids = torch.randint(0, config.vocab_size, (2, 5))

        output = model(input_ids=input_ids, logits_to_keep=2)

        self.assertEqual(output.logits.shape, (2, 2, config.vocab_size))


if __name__ == "__main__":
    unittest.main()
