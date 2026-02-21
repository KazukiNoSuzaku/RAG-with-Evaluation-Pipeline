"""
Unit tests for the evaluator module.

Tests cover:
- Dataset loading and validation
- RAG inference loop over eval datasets
- Result saving and CSV output
- Aggregate statistic computation

RAGAS metric computation itself is not tested here (it requires an LLM
API call) — that is an integration test concern.  We mock the pipeline
and RAGAS evaluate() to test the evaluator's orchestration logic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.config import EvalConfig, RAGConfig
from src.evaluator import (
    load_eval_dataset,
    run_rag_on_dataset,
    save_evaluation_results,
)
from src.rag_pipeline import RAGResult


# ---------------------------------------------------------------------------
# Tests: load_eval_dataset
# ---------------------------------------------------------------------------


class TestLoadEvalDataset:
    def test_loads_valid_dataset(self, sample_eval_dataset):
        """Should return a list of dicts for a valid JSON file."""
        data = load_eval_dataset(sample_eval_dataset)
        assert isinstance(data, list)
        assert len(data) == 2

    def test_required_fields_present(self, sample_eval_dataset):
        """Each item should have 'question' and 'ground_truth' keys."""
        data = load_eval_dataset(sample_eval_dataset)
        for item in data:
            assert "question" in item
            assert "ground_truth" in item

    def test_raises_on_missing_file(self, tmp_path):
        """Should raise FileNotFoundError for a non-existent path."""
        missing = tmp_path / "does_not_exist.json"
        with pytest.raises(FileNotFoundError):
            load_eval_dataset(missing)

    def test_raises_on_missing_fields(self, tmp_path):
        """Should raise ValueError if a required field is missing."""
        bad_data = [{"question": "Q?"}]  # missing 'ground_truth'
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(bad_data))
        with pytest.raises(ValueError, match="ground_truth"):
            load_eval_dataset(path)

    def test_empty_dataset(self, tmp_path):
        """Empty JSON array should return an empty list without error."""
        path = tmp_path / "empty.json"
        path.write_text("[]")
        data = load_eval_dataset(path)
        assert data == []

    def test_extra_fields_allowed(self, tmp_path):
        """Items with extra fields beyond the required ones are accepted."""
        dataset = [
            {
                "question": "Q?",
                "ground_truth": "A.",
                "category": "factual",
                "difficulty": "easy",
            }
        ]
        path = tmp_path / "extra.json"
        path.write_text(json.dumps(dataset))
        data = load_eval_dataset(path)
        assert len(data) == 1
        assert data[0]["category"] == "factual"


# ---------------------------------------------------------------------------
# Tests: run_rag_on_dataset
# ---------------------------------------------------------------------------


class TestRunRagOnDataset:
    def _make_mock_pipeline(self, answers: List[str]) -> MagicMock:
        """Create a mock pipeline that cycles through *answers*."""
        pipeline = MagicMock()
        results = [
            RAGResult(
                question=f"Q{i}",
                answer=ans,
                contexts=[f"ctx_{i}"],
                source_documents=[],
            )
            for i, ans in enumerate(answers)
        ]
        pipeline.query.side_effect = results
        return pipeline

    def test_returns_correct_number_of_records(self, sample_eval_dataset):
        eval_data = load_eval_dataset(sample_eval_dataset)
        pipeline = self._make_mock_pipeline(["answer 1", "answer 2"])
        records = run_rag_on_dataset(pipeline, eval_data)
        assert len(records) == 2

    def test_record_contains_required_fields(self, sample_eval_dataset):
        eval_data = load_eval_dataset(sample_eval_dataset)
        pipeline = self._make_mock_pipeline(["A", "B"])
        records = run_rag_on_dataset(pipeline, eval_data)
        required = {"question", "answer", "contexts", "ground_truth"}
        for rec in records:
            assert required.issubset(set(rec.keys()))

    def test_ground_truth_preserved(self, sample_eval_dataset):
        eval_data = load_eval_dataset(sample_eval_dataset)
        pipeline = self._make_mock_pipeline(["A", "B"])
        records = run_rag_on_dataset(pipeline, eval_data)
        for i, rec in enumerate(records):
            assert rec["ground_truth"] == eval_data[i]["ground_truth"]

    def test_contexts_is_list(self, sample_eval_dataset):
        eval_data = load_eval_dataset(sample_eval_dataset)
        pipeline = self._make_mock_pipeline(["A", "B"])
        records = run_rag_on_dataset(pipeline, eval_data)
        for rec in records:
            assert isinstance(rec["contexts"], list)

    def test_empty_eval_data_returns_empty(self):
        pipeline = MagicMock()
        records = run_rag_on_dataset(pipeline, [])
        assert records == []
        pipeline.query.assert_not_called()

    def test_pipeline_query_called_once_per_item(self, sample_eval_dataset):
        eval_data = load_eval_dataset(sample_eval_dataset)
        pipeline = self._make_mock_pipeline(["A", "B"])
        run_rag_on_dataset(pipeline, eval_data)
        assert pipeline.query.call_count == len(eval_data)


# ---------------------------------------------------------------------------
# Tests: save_evaluation_results
# ---------------------------------------------------------------------------


class TestSaveEvaluationResults:
    def _make_results(self) -> tuple:
        df = pd.DataFrame(
            {
                "user_input": ["Q1", "Q2"],
                "response": ["A1", "A2"],
                "faithfulness": [0.8, 0.9],
                "answer_relevancy": [0.7, 0.85],
            }
        )
        aggregates = {
            "faithfulness_mean": 0.85,
            "faithfulness_std": 0.07,
            "faithfulness_min": 0.8,
            "faithfulness_max": 0.9,
            "answer_relevancy_mean": 0.775,
            "answer_relevancy_std": 0.07,
            "answer_relevancy_min": 0.7,
            "answer_relevancy_max": 0.85,
        }
        return df, aggregates

    def test_creates_csv_file(self, tmp_path):
        config = EvalConfig(
            dataset_path=str(tmp_path / "eval.json"),
            results_dir=str(tmp_path / "results"),
            plots_dir=str(tmp_path / "plots"),
        )
        df, agg = self._make_results()
        output_path = save_evaluation_results(df, agg, config, "test_run")
        assert output_path.exists()
        assert output_path.suffix == ".csv"

    def test_csv_has_correct_rows(self, tmp_path):
        config = EvalConfig(
            dataset_path=str(tmp_path / "eval.json"),
            results_dir=str(tmp_path / "results"),
            plots_dir=str(tmp_path / "plots"),
        )
        df, agg = self._make_results()
        output_path = save_evaluation_results(df, agg, config, "test_run")
        loaded = pd.read_csv(output_path)
        assert len(loaded) == 2

    def test_creates_results_dir_if_missing(self, tmp_path):
        results_dir = tmp_path / "nested" / "results"
        config = EvalConfig(
            dataset_path=str(tmp_path / "eval.json"),
            results_dir=str(results_dir),
            plots_dir=str(tmp_path / "plots"),
        )
        df, agg = self._make_results()
        save_evaluation_results(df, agg, config, "test_run")
        assert results_dir.exists()

    def test_filename_contains_experiment_name(self, tmp_path):
        config = EvalConfig(
            dataset_path=str(tmp_path / "eval.json"),
            results_dir=str(tmp_path / "results"),
            plots_dir=str(tmp_path / "plots"),
        )
        df, agg = self._make_results()
        output_path = save_evaluation_results(df, agg, config, "my_experiment")
        assert "my_experiment" in output_path.name
