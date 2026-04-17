"""JoLA trainer with gate regularization and schedule support."""

import math

import torch
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer


def make_data_collator(response_template="### Response:\n", tokenizer=None, mlm=False):
    data_collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer, mlm=mlm)
    return data_collator


class LinearSchedule:
    def __init__(self, start_lambda, end_lambda, total_steps):
        self.start_lambda = start_lambda
        self.end_lambda = end_lambda
        self.total_steps = total_steps
        self.step_count = 0

    def get_lambda(self):
        self.step_count += 1
        return self.start_lambda + (self.end_lambda - self.start_lambda) * (self.step_count / self.total_steps)


class CyclicSchedule:
    def __init__(self, cycle_length, total_steps):
        self.cycle_length = cycle_length
        self.total_steps = total_steps
        self.step_count = 0

    def get_lambda(self):
        self.step_count += 1
        return 0.5 + 0.5 * math.sin(2 * math.pi * (self.step_count / self.cycle_length))


class PerformanceBasedSchedule:
    def __init__(self, initial_lambda, adjustment_factor=0.01):
        self.current_lambda = initial_lambda
        self.adjustment_factor = adjustment_factor
        self.step_count = 0

    def get_lambda(self, performance_improvement):
        self.step_count += 1
        if performance_improvement < 0:
            self.current_lambda = min(1.0, self.current_lambda + self.adjustment_factor)
        else:
            self.current_lambda = max(0.0, self.current_lambda - self.adjustment_factor)
        return self.current_lambda


class ExponentialDecaySchedule:
    def __init__(self, start_lambda, decay_rate):
        self.start_lambda = start_lambda
        self.decay_rate = decay_rate
        self.step_count = 0

    def get_lambda(self):
        self.step_count += 1
        return self.start_lambda * math.exp(-self.decay_rate * self.step_count)


class JoLATrainer(SFTTrainer):
    def __init__(
        self,
        model,
        train_dataset,
        eval_dataset,
        tokenizer,
        data_collator,
        args,
        callbacks,
        gate_scheduler,
        dataset_text_field="text",
        max_seq_length=400,
        peft_config=None,
    ):
        if callbacks:
            super().__init__(
                model=model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                dataset_text_field=dataset_text_field,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                data_collator=data_collator,
                args=args,
                peft_config=peft_config,
                callbacks=callbacks,
            )
        else:
            super().__init__(
                model=model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                dataset_text_field=dataset_text_field,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                data_collator=data_collator,
                args=args,
                peft_config=peft_config,
            )
        self.gate_scheduler = gate_scheduler
        self.num_steps = (len(train_dataset) // args.per_device_train_batch_size) * args.num_train_epochs
        if self.gate_scheduler == "linear":
            self.lambda_scheduler = LinearSchedule(0.0, 0.2, self.num_steps)
            self.gated_lambda = 0.0
        elif self.gate_scheduler == "cyclic":
            self.lambda_scheduler = CyclicSchedule(cycle_length=20, total_steps=self.num_steps)
            self.gated_lambda = 0.1
        elif self.gate_scheduler == "perform":
            self.lambda_scheduler = PerformanceBasedSchedule(initial_lambda=0.1)
            self.gated_lambda = 0.1
        elif self.gate_scheduler == "expon":
            self.lambda_scheduler = ExponentialDecaySchedule(start_lambda=0.1, decay_rate=0.01)
            self.gated_lambda = 0.1
        else:
            self.lambda_scheduler = None
            self.gated_lambda = 0.0

        self.g1_prop = []
        self.g2_prop = []
        self.last_loss = 50

    def get_penalty(self, log_alpha, stretch_limits=(-0.1, 1.1), temperature=0.33, eps=1e-6):
        low, high = torch.tensor(stretch_limits)
        assert low < 0.0, "p_gate_closed can be computed only if lower stretch limit is negative"
        p_open = torch.sigmoid(log_alpha - temperature * torch.log(-low / high))
        p_open = torch.clamp(p_open, eps, 1.0 - eps)
        total_reg = torch.sum(p_open)
        return total_reg / p_open.size(0)

    def get_gates(self, log_gate, is_train, stretch_limits=(-0.1, 1.1), temperature=0.33, eps=1e-6):
        low, high = stretch_limits
        if is_train:
            shape = log_gate.size()
            noise = (1 - 2 * eps) * torch.rand(shape).to(log_gate.device) + eps
            concrete = torch.sigmoid((torch.log(noise) - torch.log(1 - noise) + log_gate) / temperature)
        else:
            concrete = torch.sigmoid(log_gate)

        stretched_concrete = concrete * (high - low) + low
        clipped_concrete = torch.clamp(stretched_concrete, 0, 1)
        concrete_list = clipped_concrete.squeeze().tolist()
        return concrete_list

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if not model.training:
            inputs = {**inputs, "use_cache": False}

        outputs = model(**inputs)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        cn_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        loss = cn_loss

        g1_l0_norm = 0.0
        g2_l0_norm = 0.0
        if self.gated_lambda != 0:
            for name, param in model.named_parameters():
                if "log_g1" in name and param.requires_grad:
                    g1_l0_norm += self.get_penalty(param)
                    self.g1_prop.append(self.get_gates(log_gate=param, is_train=True))
                if "log_g2" in name and param.requires_grad:
                    g2_l0_norm += self.get_penalty(param)
                    self.g2_prop.append(self.get_gates(log_gate=param, is_train=True))

            num_heads_total = self.model.config.num_hidden_layers * self.model.config.num_attention_heads
            loss = (
                loss
                + self.gated_lambda * g1_l0_norm / num_heads_total
                + (1 - self.gated_lambda) * g2_l0_norm / num_heads_total
            )

        if self.gate_scheduler == "linear":
            self.gated_lambda = self.lambda_scheduler.get_lambda()
        elif self.gate_scheduler == "cyclic":
            self.gated_lambda = self.lambda_scheduler.get_lambda()
        elif self.gate_scheduler == "perform":
            self.gated_lambda = self.lambda_scheduler.get_lambda(performance_improvement=loss - self.last_loss)
            self.last_loss = loss
        elif self.gate_scheduler == "expon":
            self.gated_lambda = self.lambda_scheduler.get_lambda()

        if return_outputs:
            return (loss, outputs)
        return loss
