"""Structural and gradient checks for the paper-derived prediction model."""

import unittest

import torch

from model import CrossAttention, VisionTransformer


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def small_model(self):
        return VisionTransformer(
            image_size=(64, 64), patch_size=(32, 32), emb_dim=24,
            mlp_dim=48, num_heads=4, num_layers=2, dropout_rate=0.0)

    def test_prediction_and_gradient_flow(self):
        model = self.small_model()
        # Exercise the actual dataset dtype and zero-dropout path.
        clip = torch.randn(1, 4, 3, 64, 64, dtype=torch.float64)
        prediction = model(clip)
        self.assertEqual(prediction.shape, (1, 3, 256, 256))
        self.assertEqual(prediction.dtype, torch.float32)
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertLessEqual(prediction.abs().max().item(), 1.0)
        torch.nn.functional.mse_loss(prediction, torch.randn_like(prediction)).backward()
        for module in (model.spatial_transformer, model.temporal_transformer,
                       model.cross_att, model.decoder):
            gradients = [p.grad for p in module.parameters() if p.requires_grad]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)
        self.assertGreater(model.cls_token.grad.abs().sum().item(), 0)

    def test_shared_spatial_encoder_preserves_frame_order(self):
        model = self.small_model().eval()
        clip = torch.randn(1, 4, 3, 64, 64)
        captured = []
        handle = model.temporal_transformer.register_forward_pre_hook(
            lambda module, args: captured.append(args[0].detach()))
        with torch.no_grad():
            model(clip)
            expected = torch.stack([
                model.spatial_transformer(clip[:, frame]) for frame in range(4)
            ], dim=1)
        handle.remove()
        torch.testing.assert_close(captured[0][:, 1:], expected, atol=1e-6, rtol=1e-5)

    def test_cross_attention_aggregates_frames_and_adds_one_residual(self):
        attention = CrossAttention(4, num_heads=1)
        with torch.no_grad():
            for layer in (attention.align1, attention.wq, attention.wk,
                          attention.wv, attention.proj):
                layer.weight.copy_(torch.eye(4))
                if layer.bias is not None:
                    layer.bias.zero_()
        tokens = torch.tensor([[[1., 0., 0., 0.], [0., 2., 0., 0.],
                                [0., 0., 3., 0.]]], requires_grad=True)
        # CLS dot-products are [1, 0, 0]; head scaling is sqrt(4).
        weights = torch.tensor([0.5, 0., 0.]).softmax(0)
        expected = tokens[:, :1] + (tokens * weights[None, :, None]).sum(1, keepdim=True)
        output = attention(tokens)
        torch.testing.assert_close(output, expected)
        output.sum().backward()
        self.assertGreater(tokens.grad[:, 1:].abs().sum().item(), 0)

    def test_paper_defaults(self):
        # Inspect the full-size structure without allocating ~1GB of weights.
        with torch.device('meta'):
            model = VisionTransformer()
        self.assertEqual(model.patch_size, (32, 32))
        self.assertEqual(len(model.spatial_transformer.transformer.encoder_layers), 12)
        self.assertEqual(len(model.temporal_transformer.encoder_layers), 12)
        self.assertEqual(model.temporal_transformer.encoder_layers[0].attn.num_heads, 8)
        self.assertEqual(model.cls_token.shape[-1], 768)
        self.assertEqual(model.decoder.de_dense[0].out_features, 65536)

    def test_invalid_inputs(self):
        model = self.small_model()
        for shape in ((1, 3, 64, 64), (1, 3, 3, 64, 64),
                      (1, 4, 1, 64, 64), (1, 4, 3, 32, 32)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                model(torch.zeros(shape))
        with self.assertRaises(ValueError):
            VisionTransformer(emb_dim=25, num_heads=4)
        with self.assertRaises(ValueError):
            VisionTransformer(image_size=65, patch_size=32)


if __name__ == '__main__':
    unittest.main()
