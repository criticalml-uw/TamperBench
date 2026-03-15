"""Custom trainers for LoFiT fine-tuning."""

from trl import SFTTrainer


def debug_label_masking(batch, tokenizer, print_first_n=1):
    """Debug utility to verify label masking is correct.

    Prints the full sequence, input IDs, labels, and supervised tokens only.
    Use this to verify that only the assistant response is being supervised.

    Args:
        batch: Batch from dataloader containing 'input_ids' and 'labels'
        tokenizer: Tokenizer to decode token IDs
        print_first_n: Number of examples to print (default: 1)
    """
    print("\n" + "="*80)
    print("LABEL MASKING DEBUG")
    print("="*80)

    for idx in range(min(print_first_n, len(batch['input_ids']))):
        input_ids = batch['input_ids'][idx]
        labels = batch['labels'][idx]

        print(f"\n--- Example {idx+1} ---")
        print("\nFull sequence:")
        print(tokenizer.decode(input_ids, skip_special_tokens=False))

        print("\nInput IDs (first 50):", input_ids.tolist()[:50], "...")
        print("Labels (first 50):   ", labels.tolist()[:50], "...")

        # Extract supervised tokens (labels != -100)
        supervised_mask = labels != -100
        supervised_ids = input_ids[supervised_mask]

        num_supervised = supervised_mask.sum().item()
        num_total = len(input_ids)

        print(f"\nSupervised: {num_supervised}/{num_total} tokens ({100*num_supervised/num_total:.1f}%)")
        print("\nSupervised tokens only:")
        print(tokenizer.decode(supervised_ids, skip_special_tokens=False))

    print("\n" + "="*80 + "\n")


class CustomSFTTrainer(SFTTrainer):
    def compute_loss(self,
                     model,
                     inputs,return_outputs=False,
                     num_items_in_batch: int = None,
                     **kwargs ):

        outputs = model(**inputs)

        ### Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        ### We don't use .loss here since the model may return tuples instead of ModelOutput.
        cn_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        loss = cn_loss

        # --- one-time gradient connectivity check ---
        if not getattr(self, '_grad_check_done', False):
            print(f"[DEBUG grad-check] loss={loss.item():.4f}, loss.requires_grad={loss.requires_grad}, grad_fn={type(loss.grad_fn).__name__ if loss.grad_fn else None}")
            try:
                p = model.model.layers[0].self_attn.attn_A[0]
                print(f"[DEBUG grad-check] attn_A[0][0].requires_grad={p.requires_grad}, norm={p.norm().item():.6f}")
            except Exception as e:
                print(f"[DEBUG grad-check] could not access attn_A: {e}")
            self._grad_check_done = True

        # --- track whether params are actually updating each step ---
        step = getattr(self, '_step_count', 0)
        if step % 5 == 0:
            try:
                p = model.model.layers[0].self_attn.attn_A[0]
                print(f"[DEBUG step {step}] loss={loss.item():.4f}, attn_A[0][0] norm={p.norm().item():.6f}")
            except Exception:
                pass
        self._step_count = step + 1

        ### Add L1 regularization term
        l1norm = 0
        l1_lambda=self.l1_lambda

        for param in model.parameters():
            if param.requires_grad:
                l1norm+=param.abs().sum()
        loss+=l1_lambda*l1norm
        if return_outputs:
            return loss,outputs
        else:
            return loss

    def test(self, fname=None, task=None, eval_dataset=None, model_name=None, **kwargs):
        """No-op: evaluation is handled by the project eval system."""
        pass
