"""
测试模块
包含各模块的单元测试
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestMLFSNN(unittest.TestCase):
    """MLF-SNN 模块测试"""

    def test_surrogate_gradient(self):
        """测试替代梯度函数"""
        from src.mlf_snn.surrogate import FastSigmoid, MultiThresholdSurrogate

        # 测试 FastSigmoid
        x = torch.randn(10, requires_grad=True)
        y = FastSigmoid.apply(x, 1.0)
        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)

    def test_multi_threshold_surrogate(self):
        """测试多阈值替代梯度"""
        from src.mlf_snn.surrogate import MultiThresholdSurrogate

        x = torch.tensor([0.3, 0.8, 1.2, 2.0, 3.0], requires_grad=True)
        y = MultiThresholdSurrogate.apply(x, [0.6, 1.6, 2.6])

        expected = torch.tensor([0., 1., 1., 2., 3.])
        self.assertTrue(torch.allclose(y, expected))

    def test_lif_neuron(self):
        """测试 LIF 神经元"""
        from src.mlf_snn.neurons import LIFNeuron

        neuron = LIFNeuron(threshold=1.0, decay=0.9)
        x = torch.ones(5, 10)

        spikes = []
        for t in range(10):
            spike = neuron(x)
            spikes.append(spike)

        self.assertEqual(len(spikes), 10)
        self.assertEqual(spikes[0].shape, x.shape)

    def test_mlf_neuron(self):
        """测试 MLF 多阈值神经元"""
        from src.mlf_snn.neurons import MLFNeuron

        neuron = MLFNeuron(thresholds=[0.6, 1.6, 2.6], decay=0.9)
        x = torch.ones(5, 10) * 0.5

        spike = neuron(x)
        self.assertEqual(spike.shape, x.shape)

        # 测试信息容量
        self.assertAlmostEqual(neuron.info_capacity, 2.0, places=5)

    def test_snn_linear(self):
        """测试 SNN 线性层"""
        from src.mlf_snn.layers import SNNLinear

        layer = SNNLinear(10, 20, neuron_type='lif')
        x = torch.randn(5, 10)
        y = layer(x)

        self.assertEqual(y.shape, (5, 20))

    def test_mlfsnn_network(self):
        """测试 MLFSNN 网络"""
        from src.mlf_snn.network import MLFSNN

        model = MLFSNN(
            input_dim=12,
            hidden_dims=[64, 32],
            output_dim=2,
            time_steps=8
        )

        x = torch.randn(4, 12)
        y = model(x)

        self.assertEqual(y.shape, (4, 2))

    def test_rate_encoder(self):
        """测试速率编码器"""
        from src.mlf_snn.encoding import RateEncoder

        encoder = RateEncoder(time_steps=16)
        x = torch.rand(4, 10)
        spikes = encoder(x)

        self.assertEqual(spikes.shape, (4, 16, 10))


class TestManeuverDetection(unittest.TestCase):
    """变轨检测模块测试"""

    def test_ignition_cnn(self):
        """测试点火检测CNN"""
        from src.maneuver_detection.models.ignition_detector import IgnitionCNN

        model = IgnitionCNN(
            input_channels=3,
            window_size=60,
            num_classes=2
        )

        x = torch.randn(4, 60, 3)
        logits, _ = model(x)

        self.assertEqual(logits.shape, (4, 2))

    def test_ignition_lstm(self):
        """测试点火检测LSTM"""
        from src.maneuver_detection.models.ignition_detector import IgnitionLSTM

        model = IgnitionLSTM(
            input_dim=3,
            hidden_dim=64,
            num_classes=2
        )

        x = torch.randn(4, 60, 3)
        logits, _ = model(x)

        self.assertEqual(logits.shape, (4, 2))

    def test_delta_v_mlp(self):
        """测试Δv回归MLP"""
        from src.maneuver_detection.models.delta_v_regressor import DeltaVMLP

        model = DeltaVMLP(input_dim=20, output_dim=3)
        x = torch.randn(4, 20)
        y, _ = model(x)

        self.assertEqual(y.shape, (4, 3))


class TestDPC(unittest.TestCase):
    """DPC聚类测试"""

    def test_dpc_clustering(self):
        """测试DPC聚类"""
        from src.dpc_clustering import ImprovedDPC

        np.random.seed(42)
        X = np.vstack([
            np.random.randn(50, 2) + [0, 0],
            np.random.randn(50, 2) + [5, 5]
        ])

        dpc = ImprovedDPC()
        labels = dpc.fit_predict(X, n_clusters=2)

        self.assertEqual(len(labels), 100)
        self.assertTrue(len(np.unique(labels)) >= 2)


class TestGAN(unittest.TestCase):
    """GAN测试"""

    def test_generator(self):
        """测试生成器"""
        from src.gan import Generator

        gen = Generator(latent_dim=100, output_dim=3)
        z = torch.randn(4, 100)
        out = gen(z)

        self.assertEqual(out.shape, (4, 3))

    def test_discriminator(self):
        """测试判别器"""
        from src.gan import Discriminator

        disc = Discriminator(input_dim=3)
        x = torch.randn(4, 3)
        out = disc(x)

        self.assertEqual(out.shape, (4, 1))


if __name__ == '__main__':
    unittest.main()
