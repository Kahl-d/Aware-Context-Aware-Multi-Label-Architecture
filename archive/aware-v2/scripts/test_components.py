"""
test_components.py — Verify all AWARE v2 components locally before HPC.

Tests:
1. Config loading
2. Data loading & format
3. Dataset creation & batching
4. Model forward pass & shapes
5. Loss computation & ordering
6. Metrics computation
7. Trainer mini-run (2 epochs)
8. Threshold optimization
9. Data: train_data.pkl real-split format
10. Overfit: model can memorize 5 essays (30 epochs)

Usage:
    cd cModels && python3 scripts/test_components.py --data_dir data/
"""

import argparse
import json
import logging
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
results = []


def test(name):
    """Decorator to run and report test."""
    def decorator(fn):
        def wrapper(*args, **kwargs):
            try:
                t0 = time.time()
                fn(*args, **kwargs)
                dt = time.time() - t0
                results.append((name, True, f"{dt:.1f}s"))
                print(f"  [{PASS}] {name} ({dt:.1f}s)")
            except Exception as e:
                results.append((name, False, str(e)))
                print(f"  [{FAIL}] {name}: {e}")
                import traceback
                traceback.print_exc()
        return wrapper
    return decorator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    print("\n" + "=" * 60)
    print("AWARE v2 — Component Tests")
    print("=" * 60 + "\n")

    # ── 1. Config ──
    @test("Config: load toy.yaml")
    def _():
        from config import AWAREConfig, THEMES, NUM_THEMES
        config = AWAREConfig.from_yaml("configs/toy.yaml")
        assert config.model.encoder_name == "microsoft/deberta-v3-base"
        assert config.model.hidden_size == 768
        assert config.training.phase1_epochs == 2
        assert len(THEMES) == 11
        assert NUM_THEMES == 11
        d = config.to_dict()
        assert "model" in d and "training" in d
    _()

    @test("Config: load full.yaml")
    def _():
        from config import AWAREConfig
        config = AWAREConfig.from_yaml("configs/full.yaml")
        assert config.training.phase1_epochs == 5
        assert config.training.batch_size == 8
    _()

    # ── 2. Data loading ──
    @test("Data: load toy_data.pkl")
    def _():
        with open(data_dir / "toy_data.pkl", "rb") as f:
            toy = pickle.load(f)
        assert "train" in toy and "val" in toy
        train = toy["train"]
        assert "essays" in train and "essay_ids" in train and "weights" in train
        assert len(train["essay_ids"]) > 0
        eid = train["essay_ids"][0]
        essay = train["essays"][eid]
        assert "sentences" in essay and "annotations" in essay
        assert len(essay["sentences"]) == len(essay["annotations"])
        logger.info("Toy train: %d essays, val: %d essays", len(toy["train"]["essay_ids"]), len(toy["val"]["essay_ids"]))
    _()

    @test("Data: load splits_stats.json")
    def _():
        with open(data_dir / "splits_stats.json") as f:
            stats = json.load(f)
        assert "splits" in stats
        assert stats["leakage_check"]["train_val_student_overlap"] == 0
        assert stats["leakage_check"]["train_test_student_overlap"] == 0
        assert "theme_weights" in stats
        logger.info("Theme weights: %s", stats["theme_weights"])
    _()

    # ── 3. Dataset & DataLoader ──
    @test("Dataset: create + iterate batch")
    def _():
        from config import AWAREConfig, THEMES
        from dataset import create_dataloader
        from transformers import AutoTokenizer

        config = AWAREConfig.from_yaml("configs/toy.yaml")
        config.training.batch_size = 2
        config.training.num_workers = 0
        tokenizer = AutoTokenizer.from_pretrained(config.model.encoder_name)

        with open(data_dir / "toy_data.pkl", "rb") as f:
            toy = pickle.load(f)
        loader = create_dataloader(toy["train"], tokenizer, config, shuffle=False, augment=False)
        batch = next(iter(loader))

        assert batch["input_ids"].shape == (2, config.model.max_seq_length)
        assert batch["attention_mask"].shape == (2, config.model.max_seq_length)
        assert batch["sentence_mask"].shape == (2, config.model.max_sentences)
        assert batch["labels"].shape == (2, config.model.max_sentences, 11)
        assert len(batch["sentence_boundaries"]) == 2

        # Check sentence_mask has valid entries
        assert batch["sentence_mask"].sum() > 0
        logger.info("Batch shapes OK: input_ids=%s, labels=%s, mask_sum=%d",
                     batch["input_ids"].shape, batch["labels"].shape, int(batch["sentence_mask"].sum()))
    _()

    @test("Dataset: AEDA augmentation changes text")
    def _():
        from dataset import AWAREDataset
        ds = AWAREDataset.__new__(AWAREDataset)
        ds.aeda_prob = 1.0  # Always augment
        ds.aeda_punctuation = list(".,;:!?")
        text = "This is a test sentence with enough words to augment properly here"
        random.seed(42)
        aug = ds._aeda_augment(text)
        assert aug != text, f"AEDA with prob=1.0 should always augment 13-word text, got: '{aug}'"
    _()

    # ── 4. Model forward pass ──
    @test("Model: build + forward pass shapes")
    def _():
        from config import AWAREConfig
        from model import build_model_from_config

        config = AWAREConfig.from_yaml("configs/toy.yaml")
        model = build_model_from_config(config, freeze_encoder=True)

        bs = 2
        seq_len = config.model.max_seq_length
        max_sent = config.model.max_sentences
        input_ids = torch.randint(0, 1000, (bs, seq_len))
        attention_mask = torch.ones(bs, seq_len, dtype=torch.long)
        sentence_mask = torch.zeros(bs, max_sent)
        sentence_mask[:, :3] = 1.0
        boundaries = [[(1, 10), (10, 20), (20, 30)] + [(0, 0)] * (max_sent - 3)] * bs

        with torch.no_grad():
            out = model(input_ids, attention_mask, boundaries, sentence_mask, return_embeddings=True)

        assert out["logits"].shape == (bs, max_sent, 11)
        assert out["sentence_embeddings"].shape == (bs, max_sent, config.model.hidden_size)
        assert out["context_embeddings"].shape == (bs, max_sent, config.model.hidden_size)
        logger.info("Forward pass: logits=%s", out["logits"].shape)
    _()

    @test("Model: freeze/unfreeze encoder")
    def _():
        from config import AWAREConfig
        from model import build_model_from_config

        config = AWAREConfig.from_yaml("configs/toy.yaml")
        model = build_model_from_config(config)

        model._freeze_encoder()
        enc_frozen = sum(1 for p in model.encoder.parameters() if not p.requires_grad)
        enc_total = sum(1 for p in model.encoder.parameters())
        assert enc_frozen == enc_total

        model.unfreeze_encoder()
        enc_trainable = sum(1 for p in model.encoder.parameters() if p.requires_grad)
        assert enc_trainable == enc_total
    _()

    # ── 5. Loss ──
    @test("Loss: ASL computation + ordering")
    def _():
        from config import AWAREConfig
        from losses import build_loss_from_config

        config = AWAREConfig.from_yaml("configs/toy.yaml")
        config.loss.theme_weights = [1.0] * 11
        criterion = build_loss_from_config(config)

        bs, ms, nl = 2, 5, 11
        mask = torch.ones(bs, ms)
        targets = torch.zeros(bs, ms, nl)
        targets[0, 0, 0] = 1.0  # One positive

        # Perfect prediction
        perfect_logits = torch.full((bs, ms, nl), -5.0)
        perfect_logits[0, 0, 0] = 5.0
        loss_perfect = criterion(perfect_logits, targets, mask)

        # Random prediction
        random_logits = torch.randn(bs, ms, nl)
        loss_random = criterion(random_logits, targets, mask)

        # Wrong prediction
        wrong_logits = torch.full((bs, ms, nl), 5.0)
        wrong_logits[0, 0, 0] = -5.0
        loss_wrong = criterion(wrong_logits, targets, mask)

        assert loss_perfect.item() < loss_random.item(), f"perfect {loss_perfect.item()} should < random {loss_random.item()}"
        assert loss_random.item() < loss_wrong.item(), f"random {loss_random.item()} should < wrong {loss_wrong.item()}"
        logger.info("Loss ordering: perfect=%.4f < random=%.4f < wrong=%.4f",
                     loss_perfect.item(), loss_random.item(), loss_wrong.item())
    _()

    # ── 6. Metrics ──
    @test("Metrics: compute_metrics correctness")
    def _():
        from metrics import compute_metrics

        # Perfect predictions
        preds = np.eye(11, dtype=np.float32)
        labels = np.eye(11, dtype=np.float32)
        m = compute_metrics(preds, labels)
        assert m["f1_macro"] == 1.0, f"Expected 1.0, got {m['f1_macro']}"
        assert m["f1_micro"] == 1.0

        # All zeros
        preds_z = np.zeros((10, 11), dtype=np.float32)
        labels_z = np.ones((10, 11), dtype=np.float32)
        m_z = compute_metrics(preds_z, labels_z)
        assert m_z["f1_macro"] < 0.01
    _()

    @test("Metrics: threshold optimization")
    def _():
        from metrics import optimize_thresholds

        np.random.seed(42)
        probs = np.random.rand(100, 11).astype(np.float32)
        labels = (np.random.rand(100, 11) > 0.7).astype(np.float32)
        thresholds = optimize_thresholds(probs, labels)
        assert len(thresholds) == 11
        for theme, t in thresholds.items():
            assert 0.1 <= t <= 0.7, f"{theme}: threshold {t} out of range"
    _()

    # ── 7. Mini training (most critical test) ──
    @test("Trainer: 1+1 epoch toy training")
    def _():
        from config import AWAREConfig, THEMES
        from model import build_model_from_config
        from dataset import create_dataloader
        from trainer import AWARETrainer
        from transformers import AutoTokenizer

        config = AWAREConfig.from_yaml("configs/toy.yaml")
        config.training.phase1_epochs = 1
        config.training.phase2_epochs = 1
        config.training.batch_size = 2
        config.training.gradient_accumulation = 1
        config.training.num_workers = 0

        # Load theme weights
        stats_path = data_dir / "splits_stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)
            tw = stats.get("theme_weights", {})
            config.loss.theme_weights = [tw.get(t, 1.0) for t in THEMES]

        tokenizer = AutoTokenizer.from_pretrained(config.model.encoder_name)

        with open(data_dir / "toy_data.pkl", "rb") as f:
            toy = pickle.load(f)
        train_loader = create_dataloader(toy["train"], tokenizer, config, shuffle=True, augment=False)
        val_loader = create_dataloader(toy["val"], tokenizer, config, shuffle=False, augment=False)

        model = build_model_from_config(config)
        output_dir = Path("results/test_run")
        output_dir.mkdir(parents=True, exist_ok=True)

        trainer = AWARETrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=str(output_dir),
            config=config,
        )
        history = trainer.train()

        assert len(history["epochs"]) == 2  # 1 phase1 + 1 phase2
        assert (output_dir / "best.pt").exists()
        assert (output_dir / "history.json").exists()
        assert (output_dir / "thresholds.json").exists()
        logger.info("Mini training: best F1=%.4f at epoch %d", history["best_f1"], history["best_epoch"])
    _()

    # ── 9. Real split data format ──
    @test("Data: train_data.pkl format")
    def _():
        from dataset import load_split_data
        train_data = load_split_data(str(data_dir / "train_data.pkl"))
        assert "essays" in train_data and "essay_ids" in train_data and "weights" in train_data
        eid = train_data["essay_ids"][0]
        essay = train_data["essays"][eid]
        assert len(essay["sentences"]) == len(essay["annotations"]), \
            f"Sentence/annotation count mismatch: {len(essay['sentences'])} vs {len(essay['annotations'])}"
        assert isinstance(essay["sentences"][0], str), "sentences[0] must be a string"
        assert isinstance(essay["annotations"][0], set), "annotations[0] must be a set"
        logger.info("train_data: %d essays, spot check OK (first essay: %d sentences)",
                    len(train_data["essay_ids"]), len(essay["sentences"]))
    _()

    # ── 10. Overfit test (single most important sanity check) ──
    @test("Overfit: model can memorize 5 essays (30 epochs)")
    def _():
        """
        If the model CANNOT overfit 5 essays in 30 epochs at a high LR, something
        is fundamentally broken in the architecture or data pipeline. This test must
        pass before any HPC training run.
        """
        import torch
        from config import AWAREConfig, THEMES
        from model import build_model_from_config
        from dataset import create_dataloader
        from losses import build_loss_from_config

        config = AWAREConfig.from_yaml("configs/toy.yaml")
        config.training.batch_size = 2
        config.training.num_workers = 0

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.model.encoder_name)

        with open(data_dir / "toy_data.pkl", "rb") as f:
            toy = pickle.load(f)

        # Take only 5 essays that have at least one annotated theme
        all_ids = toy["train"]["essay_ids"]
        themed = [
            eid for eid in all_ids
            if any(ann for ann in toy["train"]["essays"][eid]["annotations"])
        ][:5]
        assert len(themed) >= 3, \
            f"Need at least 3 themed essays in toy_data, found {len(themed)}"

        mini_data = {
            "essays": {eid: toy["train"]["essays"][eid] for eid in themed},
            "essay_ids": themed,
            "weights": {eid: 1.0 for eid in themed},
        }
        loader = create_dataloader(mini_data, tokenizer, config, shuffle=True, augment=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model_from_config(config).to(device)
        criterion = build_loss_from_config(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

        # Train 30 epochs — should overfit badly (that's the point)
        model.train()
        losses = []
        for epoch in range(30):
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                sentence_mask = batch["sentence_mask"].to(device)
                labels = batch["labels"].to(device)
                sentence_boundaries = batch["sentence_boundaries"]
                out = model(input_ids, attention_mask, sentence_boundaries, sentence_mask)
                loss = criterion(out["logits"], labels, sentence_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

        initial_loss = losses[0]
        final_loss = losses[-1]
        assert final_loss < initial_loss * 0.5, (
            f"Model did not learn: initial_loss={initial_loss:.4f}, final_loss={final_loss:.4f}. "
            f"Loss must drop >50% when memorizing 5 essays. "
            f"If this fails, something is wrong with the architecture or data pipeline."
        )
        logger.info(
            "Overfit test PASSED: initial_loss=%.4f → final_loss=%.4f (%.0f%% reduction) on %s",
            initial_loss, final_loss, 100 * (1 - final_loss / initial_loss), device,
        )
    _()

    # ── Summary ──
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    for name, ok, detail in results:
        status = PASS if ok else FAIL
        print(f"  [{status}] {name} — {detail}")
    print(f"\n  {passed}/{total} tests passed")
    if passed < total:
        print(f"\n  WARNING: {total - passed} test(s) FAILED")
        sys.exit(1)
    else:
        print("\n  All tests passed! Ready for HPC.")
    print("=" * 60)


if __name__ == "__main__":
    main()
