"""GPQA evaluation implementation for SafeTuneBed.

This module implements the GPQA (Graduate-Level Google-Proof Q&A) benchmark
from Rein et al. for evaluating language models on challenging scientific questions.
"""
# pyright: reportMissingTypeStubs=false, reportUntypedNamedTuple=false, reportAny=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportExplicitAny=false

import json
import os
import random
import re
from collections import namedtuple
from dataclasses import dataclass
from typing import Any, Literal, cast

import pandas as pd
import polars as pl
import torch
from pandera.typing.polars import DataFrame
from tqdm import tqdm
from typing_extensions import override

from safetunebed.whitebox.evals.base import (
    WhiteBoxEvaluation,
    WhiteBoxEvaluationConfig,
)
from safetunebed.whitebox.evals.output_schema import (
    EvaluationSchema,
    InferenceSchema,
    ScoreSchema,
)
from safetunebed.whitebox.utils import (
    EvalName,
    MetricName,
    OptimizationDirection,
    dealloc_model_and_tokenizer,
)

# Define the Example namedtuple for GPQA dataset
Example = namedtuple(
    "Example", ["question", "choice1", "choice2", "choice3", "choice4", "correct_index"]
)


@dataclass
class GPQAConfig(WhiteBoxEvaluationConfig):
    """Configuration for GPQA evaluation.

    Attributes:
        prompt_type: Type of prompt to use. Options: 'zero_shot', '5_shot',
                    'zero_shot_chain_of_thought', 'chain_of_thought'
        dataset_split: Which GPQA dataset split to use. Options: 'main', 'diamond',
                      'experts', 'extended'. Default is 'diamond'.
        seed: Random seed for shuffling answer choices. Default is 0.
        temperature: Temperature for chain-of-thought generation. Default is 0.7.
        self_consistency_samples: Number of samples for self-consistency prompt type. Default is 20.
    """

    prompt_type: str = "zero_shot"
    dataset_split: Literal["main", "diamond", "experts", "extended"] = "diamond"
    seed: int = 0
    temperature: float = 0.7
    self_consistency_samples: int = 20


class GPQA(WhiteBoxEvaluation[GPQAConfig]):
    """GPQA evaluation class for scientific question-answering assessment.

    Implements the GPQA benchmark which tests language models on graduate-level
    scientific questions designed to be Google-proof (not easily answerable
    through simple web searches).
    """

    name: EvalName = EvalName.GPQA
    objective: MetricName = MetricName.GPQA_SCORE
    attacker_direction: OptimizationDirection = OptimizationDirection.MINIMIZE
    defender_direction: OptimizationDirection = OptimizationDirection.MAXIMIZE

    def __init__(self, eval_config: GPQAConfig):
        """Initialize the GPQA evaluation with the given configuration.

        Args:
            eval_config: Configuration object containing evaluation parameters.
        """
        super().__init__(eval_config)
        self.examples: list[Example] = []  # Store examples for later use in scoring

    def get_dataset_path(self) -> str:
        """Get the path to the GPQA dataset CSV file based on the configured split."""
        base_dir = os.path.dirname(__file__)
        dataset_path = os.path.join(
            base_dir, "dataset", f"gpqa_{self.eval_config.dataset_split}.csv"
        )
        return dataset_path

    def load_examples(self, path: str) -> list[Example]:
        """Load questions from csv file and return a list of Example namedtuples."""
        question_df = pd.read_csv(path)
        random.seed(self.eval_config.seed)

        def shuffle_choices_and_create_example(row: pd.Series) -> Example:
            list_choices = [
                row["Incorrect Answer 1"],
                row["Incorrect Answer 2"],
                row["Incorrect Answer 3"],
                row["Correct Answer"],
            ]
            random.shuffle(list_choices)
            example = Example(
                row["Question"],
                list_choices[0],
                list_choices[1],
                list_choices[2],
                list_choices[3],
                list_choices.index(row["Correct Answer"]),
            )
            return example

        return [
            shuffle_choices_and_create_example(row) for _, row in question_df.iterrows()
        ]

    def zero_shot_prompt(self, example: Example) -> str:
        """Create a zero-shot prompt given a single example."""
        prompt = f"What is the correct answer to this question: {example.question}"
        prompt += f"\n\nChoices:\n(A) {example.choice1}\n(B) {example.choice2}\n(C) {example.choice3}\n(D) {example.choice4}"
        prompt += '\n\nFormat your response as follows: "The correct answer is (insert answer here)"'
        return prompt

    def five_shot_prompt(self, example: Example) -> str:
        """Create a 5-shot prompt given a single example."""
        prompt = "Here are some example questions from experts. Answer the final question yourself, following the format of the previous questions exactly.\n"

        # Load chain of thought examples
        cot_examples_path = os.path.join(
            os.path.dirname(__file__), "chain_of_thought_examples.json"
        )
        with open(cot_examples_path) as f:
            json_data = json.load(f)

        # Generate examples without explanations
        for q in json_data["questions"]:
            prompt += f"Question: {q['question']}\nChoices:\n"
            for choice, value in q["choices"].items():
                prompt += f"({choice}) {value}\n"
            prompt += f"The correct answer is ({q['correct_answer']})\n"

        prompt += f"Question: {example.question}"
        prompt += f"\nChoices:\n(A) {example.choice1}\n(B) {example.choice2}\n(C) {example.choice3}\n(D) {example.choice4}"
        prompt += "\nWhen you're ready to answer, please use the format \"The correct answer is (insert answer here)."
        return prompt

    def chain_of_thought_prompt(self, example: Example) -> str:
        """Create a chain-of-thought prompt given a single example."""
        prompt = "Here are some example questions from experts. An explanation is given before the final answer. Answer the final question yourself, giving your reasoning beforehand.\n"

        # Load chain of thought examples
        cot_examples_path = os.path.join(
            os.path.dirname(__file__), "chain_of_thought_examples.json"
        )
        with open(cot_examples_path) as f:
            json_data = json.load(f)

        # Generate examples with explanations
        for q in json_data["questions"]:
            prompt += f"Question: {q['question']}\nChoices:\n"
            for choice, value in q["choices"].items():
                prompt += f"({choice}) {value}\n"
            prompt += f"Let's think step by step: \n{q['explanation']}\n"
            prompt += f"The correct answer is ({q['correct_answer']})\n"

        prompt += f"Question: {example.question}"
        prompt += f"\nChoices:\n(A) {example.choice1}\n(B) {example.choice2}\n(C) {example.choice3}\n(D) {example.choice4}"
        prompt += '\nGive step by step reasoning before you answer, and when you\'re ready to answer, please use the format "The correct answer is (insert answer here)":\n'
        return prompt

    def zero_shot_chain_of_thought_base_prompt(self, example: Example) -> str:
        """Create the base zero-shot chain-of-thought prompt."""
        prompt = f"What is the correct answer to this question: {example.question}"
        prompt += f"\n\nChoices:\n(A) {example.choice1}\n(B) {example.choice2}\n(C) {example.choice3}\n(D) {example.choice4}"
        prompt += "\nLet's think step by step: "
        return prompt

    def create_prompts(self, examples: list[Example]) -> list[str]:
        """Create a list of prompts given a list of examples based on prompt_type."""
        prompts = []

        if self.eval_config.prompt_type == "zero_shot":
            prompts = [self.zero_shot_prompt(example) for example in examples]
        elif self.eval_config.prompt_type == "5_shot":
            prompts = [self.five_shot_prompt(example) for example in examples]
        elif self.eval_config.prompt_type == "chain_of_thought":
            prompts = [self.chain_of_thought_prompt(example) for example in examples]
        elif self.eval_config.prompt_type == "zero_shot_chain_of_thought":
            # For zero-shot CoT, we'll handle the two-step generation in compute_inferences
            # Return base prompts here
            prompts = [
                self.zero_shot_chain_of_thought_base_prompt(example)
                for example in examples
            ]
        else:
            raise ValueError(
                f"Prompt type {self.eval_config.prompt_type} not supported."
            )

        return prompts

    def apply_chat_template(self, prompt: str, tokenizer) -> str:  # pyright: ignore[reportMissingParameterType]
        """Apply chat template to a prompt for instruction-tuned models.

        Args:
            prompt: The raw prompt text
            tokenizer: The model's tokenizer

        Returns:
            Formatted prompt with chat template applied
        """
        if (
            hasattr(tokenizer, "apply_chat_template")
            and tokenizer.chat_template is not None
        ):
            messages = [{"role": "user", "content": prompt}]
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return prompt

    @staticmethod
    def parse_answer(answer: str) -> str | None:
        """Parse the letter answer (A, B, C, or D) from model response."""
        patterns = [
            r"answer is \((.)\)",
            r"Answer: \((.)\)",
            r"answer: \((.)\)",
            r"answer \((.)\)",
            r"\((.)\)",
        ]  # note: misses cases where the answer is not in the format of
        # the pattern and sometimes grabs the repeated examples instead

        for pattern in patterns:
            match = re.search(pattern, answer)
            if match and match.group(1) in ["A", "B", "C", "D"]:
                return match.group(1)
        return None

    @override
    def compute_inferences(self) -> DataFrame[InferenceSchema]:
        """Run model on GPQA dataset to compute inferences."""
        # Load model and tokenizer
        model, tokenizer = self.load_model_and_tokenizer()

        # Load dataset
        dataset_path = self.get_dataset_path()
        self.examples = self.load_examples(dataset_path)

        # Skip question 69 if present (known issue with long choices)
        if len(self.examples) > 69:
            example_69 = self.examples[69]
            if any(
                len(choice) > 1000
                for choice in [
                    example_69.choice1,
                    example_69.choice2,
                    example_69.choice3,
                    example_69.choice4,
                ]
            ):
                self.examples[69] = Example(
                    example_69.question,
                    "Too long",
                    "Too long",
                    "Too long",
                    "Too long",
                    example_69.correct_index,
                )

        # Create prompts
        prompts = self.create_prompts(self.examples)

        # Initialize inferences storage
        inferences: dict[str, list[str]] = {
            "prompt": [],
            "response": [],
        }

        # Handle zero-shot chain-of-thought specially (requires two-step generation)
        if self.eval_config.prompt_type == "zero_shot_chain_of_thought":
            print("Generating zero-shot chain-of-thought reasoning traces...")
            for i in tqdm(range(0, len(prompts), self.eval_config.batch_size)):
                batch_prompts = prompts[i : i + self.eval_config.batch_size]

                # Apply chat template to prompts
                batch_prompts_formatted = [
                    self.apply_chat_template(prompt, tokenizer)
                    for prompt in batch_prompts
                ]

                # First step: generate reasoning
                # Leave room for reasoning generation
                max_input_length = min(512, 1024 - 500)  # 500 tokens for reasoning
                batch_encoding = tokenizer(
                    batch_prompts_formatted,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_input_length,
                )

                cot_input_ids_tensor: torch.Tensor = cast(
                    torch.Tensor, batch_encoding["input_ids"]
                )
                cot_attention_mask_tensor: torch.Tensor = cast(
                    torch.Tensor, batch_encoding["attention_mask"]
                )
                input_ids = cot_input_ids_tensor.to("cuda")
                attention_mask = cot_attention_mask_tensor.to("cuda")

                with torch.no_grad():
                    generate = cast(Any, model.generate)
                    reasoning_outputs = generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=500,
                        temperature=self.eval_config.temperature,
                        do_sample=True,
                    )

                # Extract reasoning and create final prompts
                final_prompts = []
                final_prompts_raw = []
                for j, (prompt, input_id, output_ids) in enumerate(
                    zip(batch_prompts, input_ids, reasoning_outputs, strict=False)
                ):
                    # Skip question 69
                    if i + j == 69:
                        final_prompts.append("Choices too long")
                        final_prompts_raw.append("Choices too long")
                        continue

                    # Decode only the generated reasoning tokens
                    input_length = input_id.shape[0]
                    generated_ids = output_ids[input_length:]
                    reasoning = tokenizer.decode(
                        generated_ids, skip_special_tokens=True
                    ).strip()

                    # Create final prompt with reasoning (raw version for storage)
                    final_prompt_raw = (
                        prompt
                        + "\n"
                        + reasoning
                        + '\n\nBased on the above, what is the single, most likely answer choice? Answer in the format "The correct answer is (insert answer here)".'
                    )
                    final_prompts_raw.append(final_prompt_raw)

                    # Create formatted version for model
                    final_prompts.append(
                        self.apply_chat_template(final_prompt_raw, tokenizer)
                    )

                # Second step: generate final answer
                final_encoding = tokenizer(
                    final_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024,
                )

                cot_final_input_ids_tensor: torch.Tensor = cast(
                    torch.Tensor, final_encoding["input_ids"]
                )
                cot_final_attention_mask_tensor: torch.Tensor = cast(
                    torch.Tensor, final_encoding["attention_mask"]
                )
                final_input_ids = cot_final_input_ids_tensor.to("cuda")
                final_attention_mask = cot_final_attention_mask_tensor.to("cuda")

                with torch.no_grad():
                    generate = cast(Any, model.generate)
                    final_outputs = generate(
                        input_ids=final_input_ids,
                        attention_mask=final_attention_mask,
                        max_new_tokens=100,
                        do_sample=False,  # Greedy decoding for final answer
                    )

                # Store results
                for prompt_raw, input_id, output_ids in zip(
                    final_prompts_raw, final_input_ids, final_outputs, strict=False
                ):
                    # Decode only the generated answer tokens
                    input_length = input_id.shape[0]
                    generated_ids = output_ids[input_length:]
                    response = tokenizer.decode(
                        generated_ids, skip_special_tokens=True
                    ).strip()

                    inferences["prompt"].append(prompt_raw)
                    inferences["response"].append(response)

        else:
            # Standard generation for other prompt types
            for i in tqdm(range(0, len(prompts), self.eval_config.batch_size)):
                batch_prompts = prompts[i : i + self.eval_config.batch_size]

                # Apply chat template to each prompt
                batch_prompts_formatted = [
                    self.apply_chat_template(prompt, tokenizer)
                    for prompt in batch_prompts
                ]

                batch_encoding = tokenizer(
                    batch_prompts_formatted,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024,
                )

                std_input_ids_tensor: torch.Tensor = cast(
                    torch.Tensor, batch_encoding["input_ids"]
                )
                std_attention_mask_tensor: torch.Tensor = cast(
                    torch.Tensor, batch_encoding["attention_mask"]
                )
                input_ids = std_input_ids_tensor.to("cuda")
                attention_mask = std_attention_mask_tensor.to("cuda")

                with torch.no_grad():
                    generate = cast(Any, model.generate)
                    batch_outputs = generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=self.eval_config.max_generation_length,
                        do_sample=self.eval_config.temperature > 0,
                        temperature=self.eval_config.temperature
                        if self.eval_config.temperature > 0
                        else None,
                    )

                # Decode responses
                for j, (prompt, _, input_id, output_ids) in enumerate(
                    zip(
                        batch_prompts,
                        batch_prompts_formatted,
                        input_ids,
                        batch_outputs,
                        strict=False,
                    )
                ):
                    # Decode only the generated tokens (not the input)
                    input_length = input_id.shape[0]
                    generated_ids = output_ids[input_length:]
                    response = tokenizer.decode(
                        generated_ids, skip_special_tokens=True
                    ).strip()

                    inferences["prompt"].append(prompt)
                    inferences["response"].append(response)

        # Deallocate model and tokenizer
        dealloc_model_and_tokenizer(model, tokenizer)

        # Create and validate dataframe
        _inferences_df: pl.DataFrame = pl.from_dict(data=inferences)
        return InferenceSchema.validate(_inferences_df)

    @override
    def compute_scores(
        self, inferences: DataFrame[InferenceSchema]
    ) -> DataFrame[ScoreSchema]:
        """Compute scores by evaluating correctness of model answers.

        Args:
            inferences: DataFrame with prompts and model responses

        Returns:
            DataFrame with prompts, responses, and correctness scores
        """
        inferences_df: DataFrame[InferenceSchema] = InferenceSchema.validate(inferences)

        # Initialize scores dict with inferences (to keep prompt/response alongside)
        scores_dict: dict[str, pl.Series] = inferences_df.to_dict()

        # LETTER_TO_INDEX mapping for answer evaluation
        LETTER_TO_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3}

        # Compute per-example scores
        scores: list[float] = []
        refusals = 0

        for i, response in enumerate(inferences_df["response"]):
            # Skip question 69 if it was marked as "Too long"
            if i == 69 and self.examples[i].choice1 == "Too long":
                scores.append(0.0)
                continue

            # Parse the model's answer
            parsed_answer = self.parse_answer(response)
            print(f"Parsed answer: {parsed_answer}")

            if parsed_answer is None:
                # Model refused to answer or couldn't parse answer
                scores.append(0.0)
                refusals += 1
            else:
                # Check if answer is correct
                correct_index = self.examples[i].correct_index
                model_index = LETTER_TO_INDEX.get(parsed_answer, -1)

                if model_index == correct_index:
                    scores.append(1.0)
                else:
                    scores.append(0.0)

        # Add scores to the dataframe
        scores_dict["score"] = pl.Series(scores)

        # Log refusal rate if there were any
        if refusals > 0:
            refusal_rate = refusals / len(scores)
            print(f"Refusal rate: {refusal_rate:.2%} ({refusals}/{len(scores)})")

        _scores_df: pl.DataFrame = pl.from_dict(data=scores_dict)
        return ScoreSchema.validate(_scores_df)

    @override
    def compute_results(
        self, scores: DataFrame[ScoreSchema]
    ) -> DataFrame[EvaluationSchema]:
        """Aggregate scores into overall evaluation metrics.

        Args:
            scores: DataFrame with per-example scores

        Returns:
            DataFrame with overall evaluation metrics (GPQA accuracy)
        """
        scores_df: DataFrame[ScoreSchema] = ScoreSchema.validate(scores)

        # Calculate overall accuracy (mean of per-example scores)
        scores_values: list[float] = cast(list[float], list(scores_df["score"]))
        accuracy: float = (
            float(sum(scores_values) / len(scores_values)) if scores_values else 0.0
        )

        # Create metrics dataframe
        metrics: dict[str, list[str] | list[float]] = {
            "metric_name": [str(self.objective)],
            "metric_value": [accuracy],
        }

        # Log final accuracy
        print(f"GPQA Accuracy: {accuracy:.2%}")

        _metrics_df: pl.DataFrame = pl.from_dict(data=metrics)
        return EvaluationSchema.validate(_metrics_df)


class GPQASmallEvaluation(GPQA):
    """GPQA Evaluation class using a small subset of the GPQA dataset for testing."""

    name: EvalName = EvalName.GPQA_SMALL

    @override
    def load_examples(self, path: str) -> list[Example]:
        """Load a small subset of questions from csv file for testing."""
        # Load all examples using parent method
        all_examples = super().load_examples(path)

        # Return only first 50 examples
        return all_examples[:50]
