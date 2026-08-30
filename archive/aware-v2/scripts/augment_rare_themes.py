"""
augment_rare_themes.py — Generate synthetic training data for rare CCW themes.

Uses the Anthropic Claude API to paraphrase sentences labeled with rare themes
(First Gen, Community Consciousness, Filial Piety) and produces an augmented
training dataset.

Usage:
    # Dry run — see what would be generated
    python scripts/augment_rare_themes.py \
        --data_dir data/ --output_dir data/ --dry-run

    # Full generation
    python scripts/augment_rare_themes.py \
        --data_dir data/ --output_dir data/ \
        --api-key sk-ant-... --target-count 500

    # Resume interrupted run (checks for existing output)
    python scripts/augment_rare_themes.py \
        --data_dir data/ --output_dir data/ \
        --api-key sk-ant-... --target-count 500
"""

import argparse
import json
import logging
import os
import pickle
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Theme definitions — used in the Claude prompt for accurate paraphrasing
# ---------------------------------------------------------------------------

THEMES = [
    "Navigational", "Attainment", "Perseverance", "Aspirational",
    "Social", "Filial Piety", "Spiritual", "Familial",
    "Resistance", "Community Consciousness", "First Gen",
]

THEME_DEFINITIONS = {
    "Navigational": (
        "Knowledge of how to navigate through educational institutions and systems. "
        "Understanding bureaucratic processes, knowing how to find resources, "
        "seek help from advisors, or maneuver through academic requirements."
    ),
    "Attainment": (
        "Value placed on educational achievement and degree attainment. "
        "Emphasis on the importance of getting a degree, graduating, and the "
        "meaning of academic credentials."
    ),
    "Perseverance": (
        "Maintaining hope and resilience despite challenges and barriers. "
        "Pushing through difficulties, setbacks, and obstacles in one's "
        "educational journey."
    ),
    "Aspirational": (
        "Dreams and hopes for the future, especially educational and career goals. "
        "Envisioning a better future through education and professional aspirations."
    ),
    "Social": (
        "Networks of people and community resources that provide support. "
        "Peers, mentors, study groups, and social connections that help "
        "students succeed in education."
    ),
    "Filial Piety": (
        "Sense of duty and responsibility toward family and cultural traditions. "
        "Honoring family expectations, fulfilling obligations to parents and "
        "elders, and respecting cultural heritage in educational pursuits."
    ),
    "Spiritual": (
        "Faith, religion, or spiritual practices that provide strength and "
        "guidance. Drawing on spiritual beliefs, prayer, or religious community "
        "for motivation and resilience in education."
    ),
    "Familial": (
        "Cultural knowledge and values nurtured by family. Family stories, "
        "lessons, cultural traditions, and the role of family in shaping "
        "educational values and identity."
    ),
    "Resistance": (
        "Challenging inequality and oppressive structures through education. "
        "Using education as a tool to fight systemic barriers, prove doubters "
        "wrong, and resist marginalization."
    ),
    "Community Consciousness": (
        "Awareness of community needs and desire to give back. Motivation to "
        "use one's education to improve the lives of others in one's community, "
        "neighborhood, or cultural group."
    ),
    "First Gen": (
        "Experience as the first in one's family to attend college. Navigating "
        "higher education without family guidance, facing unique challenges and "
        "pressures that come with being a first-generation college student."
    ),
}

RARE_THEMES = ["Filial Piety", "Community Consciousness", "First Gen"]

# ---------------------------------------------------------------------------
# Checkpoint file for resumability
# ---------------------------------------------------------------------------
CHECKPOINT_FILENAME = "_augment_checkpoint.json"


def load_train_data(data_dir: str) -> dict:
    """Load the training data pickle file."""
    path = Path(data_dir) / "train_data.pkl"
    logger.info("Loading training data from %s", path)
    with open(path, "rb") as f:
        data = pickle.load(f)
    logger.info(
        "Loaded %d essays, %d essay IDs",
        len(data["essays"]), len(data["essay_ids"]),
    )
    return data


def extract_rare_theme_sentences(
    data: dict,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract all sentences that have at least one rare theme label.

    Returns:
        Dict mapping theme name to list of dicts:
            {
                "sentence": str,
                "essay_id": str,
                "sent_idx": int,
                "all_themes": list[str],  # all themes for this sentence
            }
    """
    theme_sentences: Dict[str, List[Dict[str, Any]]] = {
        t: [] for t in RARE_THEMES
    }

    for eid in data["essay_ids"]:
        essay = data["essays"][eid]
        sentences = essay["sentences"]
        annotations = essay["annotations"]

        for sent_idx, (sent, ann) in enumerate(zip(sentences, annotations)):
            # ann is a set of theme names
            for theme in RARE_THEMES:
                if theme in ann:
                    theme_sentences[theme].append({
                        "sentence": sent,
                        "essay_id": eid,
                        "sent_idx": sent_idx,
                        "all_themes": sorted(ann),
                    })

    for theme, sents in theme_sentences.items():
        logger.info("  %s: %d sentences found", theme, len(sents))

    return theme_sentences


def compute_generation_plan(
    theme_sentences: Dict[str, List[Dict[str, Any]]],
    target_count: int,
    paraphrases_per_sentence: int = 3,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute how many paraphrases to generate per theme.

    Returns:
        Dict mapping theme name to plan dict:
            {
                "existing_count": int,
                "target_count": int,
                "needed": int,
                "sentences_to_augment": int,
                "paraphrases_per_sentence": int,
                "total_to_generate": int,
            }
    """
    plan = {}
    for theme in RARE_THEMES:
        existing = len(theme_sentences[theme])
        needed = max(0, target_count - existing)

        if needed == 0:
            sentences_to_augment = 0
            pps = 0
            total = 0
        else:
            # How many source sentences do we need to paraphrase?
            # Each source produces `paraphrases_per_sentence` new sentences.
            sentences_to_augment = min(
                existing,
                (needed + paraphrases_per_sentence - 1) // paraphrases_per_sentence,
            )
            pps = paraphrases_per_sentence
            total = sentences_to_augment * pps

        plan[theme] = {
            "existing_count": existing,
            "target_count": target_count,
            "needed": needed,
            "sentences_to_augment": sentences_to_augment,
            "paraphrases_per_sentence": pps,
            "total_to_generate": total,
        }

    return plan


def build_paraphrase_prompt(
    sentence: str,
    themes: List[str],
    n_paraphrases: int = 3,
) -> str:
    """
    Build the prompt for Claude to generate paraphrases of a reflective essay sentence.
    """
    theme_defs = "\n".join(
        f"  - {t}: {THEME_DEFINITIONS[t]}" for t in themes if t in THEME_DEFINITIONS
    )

    prompt = f"""You are helping generate training data for a machine learning model that classifies sentences from STEM college student reflective essays into cultural wealth themes.

The following sentence is from a student's reflective essay and has been labeled with these cultural wealth theme(s):
{', '.join(themes)}

Theme definitions:
{theme_defs}

Original sentence:
"{sentence}"

Generate exactly {n_paraphrases} paraphrased versions of this sentence. Each paraphrase must:
1. Preserve the same cultural wealth theme meaning(s) as the original
2. Sound like a realistic sentence from a college student's reflective essay
3. Use different wording, sentence structure, or phrasing than the original
4. Maintain a first-person perspective (as a student writing about their experience)
5. Be a single sentence (no line breaks)
6. Not be a trivial rewording — vary vocabulary and structure meaningfully

Return ONLY a JSON array of {n_paraphrases} strings, one per paraphrase. No explanation, no markdown formatting, just the JSON array.

Example format:
["Paraphrase 1 text here.", "Paraphrase 2 text here.", "Paraphrase 3 text here."]"""

    return prompt


def load_checkpoint(output_dir: str) -> Dict[str, Any]:
    """Load checkpoint if it exists, for resumability."""
    ckpt_path = Path(output_dir) / CHECKPOINT_FILENAME
    if ckpt_path.exists():
        with open(ckpt_path) as f:
            ckpt = json.load(f)
        logger.info(
            "Resuming from checkpoint: %d sentences already generated",
            ckpt.get("total_generated", 0),
        )
        return ckpt
    return {"generated": {}, "total_generated": 0}


def save_checkpoint(output_dir: str, checkpoint: Dict[str, Any]):
    """Save checkpoint for resumability."""
    ckpt_path = Path(output_dir) / CHECKPOINT_FILENAME
    with open(ckpt_path, "w") as f:
        json.dump(checkpoint, f, indent=2)


def generate_paraphrases_batch(
    client,
    sentences_batch: List[Dict[str, Any]],
    model: str,
    n_paraphrases: int = 3,
    max_retries: int = 5,
) -> List[Dict[str, Any]]:
    """
    Generate paraphrases for a batch of sentences using the Claude API.

    Each sentence is processed individually (one API call per sentence)
    with retry logic for rate limits and transient errors.

    Returns:
        List of result dicts:
            {
                "source_sentence": str,
                "source_essay_id": str,
                "source_sent_idx": int,
                "themes": list[str],
                "paraphrases": list[str],
            }
    """
    results = []

    for item in sentences_batch:
        sentence = item["sentence"]
        themes = item["all_themes"]

        prompt = build_paraphrase_prompt(sentence, themes, n_paraphrases)

        paraphrases = None
        for attempt in range(max_retries):
            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )

                # Extract text content
                text = response.content[0].text.strip()

                # Parse JSON array from response
                # Handle cases where model wraps in markdown code block
                if text.startswith("```"):
                    lines = text.split("\n")
                    # Remove first and last lines (```json and ```)
                    text = "\n".join(
                        line for line in lines
                        if not line.strip().startswith("```")
                    ).strip()

                paraphrases = json.loads(text)

                if not isinstance(paraphrases, list):
                    logger.warning(
                        "Response is not a list for sentence '%s...', retrying",
                        sentence[:50],
                    )
                    paraphrases = None
                    continue

                # Filter out empty strings and ensure we have strings
                paraphrases = [
                    p.strip() for p in paraphrases
                    if isinstance(p, str) and p.strip()
                ]

                if len(paraphrases) == 0:
                    logger.warning(
                        "No valid paraphrases parsed for sentence '%s...', retrying",
                        sentence[:50],
                    )
                    paraphrases = None
                    continue

                break  # Success

            except json.JSONDecodeError as e:
                logger.warning(
                    "JSON parse error on attempt %d/%d for '%s...': %s",
                    attempt + 1, max_retries, sentence[:50], e,
                )
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)

            except Exception as e:
                error_str = str(e)
                # Check for rate limit errors
                if "rate" in error_str.lower() or "429" in error_str:
                    wait_time = min(60, 2 ** (attempt + 2))
                    logger.warning(
                        "Rate limited on attempt %d/%d, waiting %ds...",
                        attempt + 1, max_retries, wait_time,
                    )
                    time.sleep(wait_time)
                elif "overloaded" in error_str.lower() or "529" in error_str:
                    wait_time = min(120, 2 ** (attempt + 3))
                    logger.warning(
                        "API overloaded on attempt %d/%d, waiting %ds...",
                        attempt + 1, max_retries, wait_time,
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(
                        "API error on attempt %d/%d for '%s...': %s",
                        attempt + 1, max_retries, sentence[:50], e,
                    )
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)

        if paraphrases is None:
            logger.error(
                "Failed to generate paraphrases for sentence: '%s...' after %d attempts",
                sentence[:80], max_retries,
            )
            continue

        results.append({
            "source_sentence": sentence,
            "source_essay_id": item["essay_id"],
            "source_sent_idx": item["sent_idx"],
            "themes": themes,
            "paraphrases": paraphrases,
        })

    return results


def create_augmented_dataset(
    original_data: dict,
    all_generated: List[Dict[str, Any]],
) -> dict:
    """
    Create augmented training data by adding synthetic single-sentence essays.

    Each synthetic sentence becomes its own essay with essay_id prefix "SYNTH_".
    The original data is preserved unchanged.

    Args:
        original_data: Original train_data.pkl contents.
        all_generated: List of generation result dicts from generate_paraphrases_batch.

    Returns:
        New data dict in the same format as train_data.pkl, with synthetic essays added.
    """
    # Deep copy the original essays dict
    augmented_essays = dict(original_data["essays"])
    augmented_ids = list(original_data["essay_ids"])

    # Preserve weights if present
    augmented_weights = dict(original_data.get("weights", {}))

    synth_count = 0

    for result in all_generated:
        themes = set(result["themes"])
        source_eid = result["source_essay_id"]

        for para_idx, paraphrase in enumerate(result["paraphrases"]):
            # Create unique synthetic essay ID
            synth_id = f"SYNTH_{source_eid}_{result['source_sent_idx']}_{para_idx}"

            # Each synthetic essay has exactly one sentence
            augmented_essays[synth_id] = {
                "sentences": [paraphrase],
                "annotations": [themes],
            }
            augmented_ids.append(synth_id)

            # Give synthetic essays a neutral weight
            augmented_weights[synth_id] = 1.0

            synth_count += 1

    logger.info(
        "Created augmented dataset: %d original + %d synthetic = %d total essays",
        len(original_data["essay_ids"]),
        synth_count,
        len(augmented_ids),
    )

    return {
        "essays": augmented_essays,
        "essay_ids": augmented_ids,
        "weights": augmented_weights,
    }


def save_report(
    output_dir: str,
    plan: Dict[str, Dict[str, Any]],
    actual_counts: Dict[str, int],
    total_api_calls: int,
):
    """Save a JSON report summarizing the augmentation."""
    report = {
        "rare_themes": RARE_THEMES,
        "plan": plan,
        "actual_generated": actual_counts,
        "total_api_calls": total_api_calls,
        "per_theme_summary": {},
    }

    for theme in RARE_THEMES:
        original = plan[theme]["existing_count"]
        generated = actual_counts.get(theme, 0)
        report["per_theme_summary"][theme] = {
            "original_sentences": original,
            "synthetic_sentences": generated,
            "total_after_augmentation": original + generated,
            "target": plan[theme]["target_count"],
        }

    report_path = Path(output_dir) / "augment_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Report saved to %s", report_path)
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data for rare CCW themes using Claude API"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to data directory containing train_data.pkl",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where to save augmented_train_data.pkl and augment_report.json",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-20250514",
        help="Claude model to use for generation (default: claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Just count what would be generated, without calling the API",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=500,
        help="Target total sentences per rare theme after augmentation (default: 500)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of sentences to process before saving checkpoint (default: 10)",
    )
    parser.add_argument(
        "--paraphrases-per-sentence",
        type=int,
        default=3,
        help="Number of paraphrases to generate per source sentence (default: 3)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load training data and extract rare theme sentences ──────────
    logger.info("=" * 60)
    logger.info("STEP 1: Loading training data and extracting rare theme sentences")
    logger.info("=" * 60)

    data = load_train_data(args.data_dir)
    theme_sentences = extract_rare_theme_sentences(data)

    # ── Step 2: Compute generation plan ──────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2: Computing generation plan (target=%d per theme)", args.target_count)
    logger.info("=" * 60)

    plan = compute_generation_plan(
        theme_sentences,
        target_count=args.target_count,
        paraphrases_per_sentence=args.paraphrases_per_sentence,
    )

    total_api_calls = 0
    for theme, p in plan.items():
        logger.info(
            "  %s: %d existing, need %d more, will paraphrase %d sentences "
            "(%d paraphrases each) = %d new sentences",
            theme, p["existing_count"], p["needed"],
            p["sentences_to_augment"], p["paraphrases_per_sentence"],
            p["total_to_generate"],
        )
        total_api_calls += p["sentences_to_augment"]

    logger.info("Total API calls needed: %d", total_api_calls)

    # ── Dry run: stop here ───────────────────────────────────────────────────
    if args.dry_run:
        logger.info("=" * 60)
        logger.info("DRY RUN — no API calls made")
        logger.info("=" * 60)

        # Save dry-run report
        dry_report = save_report(
            str(output_dir), plan, {t: 0 for t in RARE_THEMES}, 0,
        )
        for theme, summary in dry_report["per_theme_summary"].items():
            logger.info(
                "  %s: %d original → would generate %d → total %d (target %d)",
                theme,
                summary["original_sentences"],
                plan[theme]["total_to_generate"],
                summary["original_sentences"] + plan[theme]["total_to_generate"],
                summary["target"],
            )
        return

    # ── Step 3: Initialize API client ────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3: Initializing Anthropic API client")
    logger.info("=" * 60)

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error(
            "No API key provided. Use --api-key or set ANTHROPIC_API_KEY env var."
        )
        sys.exit(1)

    try:
        import anthropic
    except ImportError:
        logger.error(
            "The 'anthropic' package is not installed. "
            "Install it with: pip install anthropic"
        )
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    logger.info("API client initialized (model: %s)", args.model)

    # ── Step 4: Generate paraphrases with checkpointing ──────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4: Generating paraphrases")
    logger.info("=" * 60)

    checkpoint = load_checkpoint(str(output_dir))
    already_done = checkpoint.get("generated", {})
    # Keys in checkpoint["generated"] are "theme::essay_id::sent_idx"

    all_generated: List[Dict[str, Any]] = []
    # Reload previously generated items from checkpoint
    for key, gen_data in already_done.items():
        all_generated.append(gen_data)

    actual_counts: Dict[str, int] = Counter()
    for gen_data in all_generated:
        for theme in gen_data["themes"]:
            if theme in RARE_THEMES:
                actual_counts[theme] += len(gen_data["paraphrases"])

    api_calls_made = checkpoint.get("total_generated", 0)
    batch_buffer: List[Dict[str, Any]] = []

    for theme in RARE_THEMES:
        p = plan[theme]
        if p["sentences_to_augment"] == 0:
            logger.info("  %s: already at target, skipping", theme)
            continue

        sentences_for_theme = theme_sentences[theme][:p["sentences_to_augment"]]

        logger.info(
            "  Processing %s: %d sentences to paraphrase...",
            theme, len(sentences_for_theme),
        )

        for i, item in enumerate(sentences_for_theme):
            # Check if already done (resume support)
            ckpt_key = f"{theme}::{item['essay_id']}::{item['sent_idx']}"
            if ckpt_key in already_done:
                continue

            batch_buffer.append(item)

            # Process batch when full or at the end
            if len(batch_buffer) >= args.batch_size or i == len(sentences_for_theme) - 1:
                results = generate_paraphrases_batch(
                    client=client,
                    sentences_batch=batch_buffer,
                    model=args.model,
                    n_paraphrases=args.paraphrases_per_sentence,
                )

                for result in results:
                    r_key = (
                        f"{theme}::{result['source_essay_id']}"
                        f"::{result['source_sent_idx']}"
                    )
                    checkpoint["generated"][r_key] = result
                    all_generated.append(result)

                    for t in result["themes"]:
                        if t in RARE_THEMES:
                            actual_counts[t] += len(result["paraphrases"])

                api_calls_made += len(batch_buffer)
                checkpoint["total_generated"] = api_calls_made

                # Save checkpoint
                save_checkpoint(str(output_dir), checkpoint)

                logger.info(
                    "    %s: %d/%d done (%d API calls total, %d paraphrases for theme so far)",
                    theme, i + 1, len(sentences_for_theme),
                    api_calls_made, actual_counts.get(theme, 0),
                )

                batch_buffer = []

                # Brief pause between batches to avoid rate limiting
                if i < len(sentences_for_theme) - 1:
                    time.sleep(1.0)

    # ── Step 5: Create augmented dataset ─────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 5: Creating augmented dataset")
    logger.info("=" * 60)

    augmented_data = create_augmented_dataset(data, all_generated)

    # Verify the augmented data structure
    synth_count = sum(1 for eid in augmented_data["essay_ids"] if eid.startswith("SYNTH_"))
    logger.info(
        "Augmented dataset: %d total essays (%d original + %d synthetic)",
        len(augmented_data["essay_ids"]),
        len(augmented_data["essay_ids"]) - synth_count,
        synth_count,
    )

    # Count final theme distribution
    final_theme_counts: Dict[str, int] = Counter()
    for eid in augmented_data["essay_ids"]:
        essay = augmented_data["essays"][eid]
        for ann in essay["annotations"]:
            for t in ann:
                if t in RARE_THEMES:
                    final_theme_counts[t] += 1

    for theme in RARE_THEMES:
        logger.info(
            "  %s: %d original → %d total (target was %d)",
            theme,
            plan[theme]["existing_count"],
            final_theme_counts.get(theme, 0),
            args.target_count,
        )

    # ── Step 6: Save outputs ─────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 6: Saving outputs")
    logger.info("=" * 60)

    # Save augmented training data
    output_path = output_dir / "augmented_train_data.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(augmented_data, f)
    logger.info("Augmented data saved to %s", output_path)

    # Save report
    save_report(str(output_dir), plan, dict(actual_counts), api_calls_made)

    # Clean up checkpoint file (generation complete)
    ckpt_path = output_dir / CHECKPOINT_FILENAME
    if ckpt_path.exists():
        ckpt_path.unlink()
        logger.info("Checkpoint file removed (generation complete)")

    logger.info("=" * 60)
    logger.info("AUGMENTATION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
