"""Точная реализация зафиксированного инференса PP-LCNet x0.25.

Модель обрабатывает исходный текстовый фрагмент и его точный поворот на 180°.
Перед температурной калибровкой логиты приводятся к антисимметричному виду.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

EXPECTED_WEIGHTS_SHA256 = "08ee1ce4bcdb30dd4e784862334d1df49d80a600294d75daae4ca4c70bc860e9"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1 - 1e-7)
    return np.log(probability / (1 - probability))


def _sigmoid(logit: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(np.asarray(logit, dtype=np.float64), -40, 40)))


class FinalOrientationRuntime:
    """Среда Paddle Inference FP32 для неизменяемого финального конвейера."""

    def __init__(self, package_root: Path, config: dict):
        from paddle.inference import Config, create_predictor

        required = {
            "architecture": "PP-LCNet_x0_25_textline_ori",
            "backend": "paddle_infer_fp32",
            "input_shape": [3, 80, 160],
            "symmetry": "logit_antisymmetrization",
            "ocr_enabled": False,
            "int8_enabled": False,
            "weights_sha256": EXPECTED_WEIGHTS_SHA256,
        }
        for key, expected in required.items():
            if config.get(key) != expected:
                raise RuntimeError(f"Неожиданное значение зафиксированного параметра {key}: {config.get(key)!r}")

        self.temperature = float(config["temperature"])
        self.mean = np.asarray(config["normalization"]["mean"], dtype=np.float32)
        self.std = np.asarray(config["normalization"]["std"], dtype=np.float32)
        weights_dir = package_root / "weights"
        weights_path = weights_dir / "inference.pdiparams"
        if sha256_file(weights_path) != EXPECTED_WEIGHTS_SHA256:
            raise RuntimeError("SHA-256 финальных весов не совпадает с ожидаемым")

        paddle_config = Config(str(weights_dir / "inference.json"), str(weights_path))
        paddle_config.disable_gpu()
        paddle_config.set_cpu_math_library_num_threads(int(config["cpu_threads"]))
        paddle_config.disable_glog_info()
        self.predictor = create_predictor(paddle_config)
        self.input_handle = self.predictor.get_input_handle("x")
        self.output_handle = self.predictor.get_output_handle("fetch_name_0")

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        # Штатная предобработка модели: фиксированный размер 160x80, RGB,
        # нормализация ImageNet и преобразование HWC -> CHW. Пропорции не сохраняются.
        array = np.asarray(image.convert("RGB"))
        array = cv2.resize(array, (160, 80), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        array /= 255.0
        return ((array - self.mean) / self.std).transpose(2, 0, 1)

    def _predict_batch(self, batch: np.ndarray) -> np.ndarray:
        self.input_handle.reshape(batch.shape)
        self.input_handle.copy_from_cpu(batch)
        self.predictor.run()
        return self.output_handle.copy_to_cpu()[:, 1].astype(np.float64)

    def predict_paths(self, paths: list[Path], batch_size: int = 64) -> np.ndarray:
        probabilities: list[float] = []
        for start in range(0, len(paths), batch_size):
            tensors: list[np.ndarray] = []
            for path in paths[start : start + batch_size]:
                with Image.open(path) as opened:
                    image = opened.convert("RGB")
                tensors.append(self._preprocess(image))
                tensors.append(self._preprocess(image.transpose(Image.Transpose.ROTATE_180)))

            raw = self._predict_batch(np.stack(tensors))
            probability_x, probability_rot180 = raw[0::2], raw[1::2]
            symmetric_logit = (_logit(probability_x) - _logit(probability_rot180)) / 2
            probabilities.extend(_sigmoid(symmetric_logit / self.temperature).tolist())

        result = np.asarray(probabilities, dtype=np.float64)
        if len(result) != len(paths):
            raise RuntimeError("Длина результата инференса не совпадает с числом входов")
        if not np.isfinite(result).all() or ((result < 0) | (result > 1)).any():
            raise RuntimeError("Инференс вернул некорректные вероятности")
        return result
