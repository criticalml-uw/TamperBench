# Adapted from T-Vaccine: trainer.py
# (https://github.com/Lslland/T-Vaccine)
#
import copy
import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from tamperbench.whitebox.defenses.t_vaccine.repnoise_loss import rep_noise_loss
from packaging import version
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.cuda import amp
from transformers import Trainer, logging
from transformers.utils import is_sagemaker_mp_enabled

try:
    from transformers.trainer_pt_utils import smp_forward_backward
except ImportError:  # pragma: no cover - fallback when SMP deps absent
    def smp_forward_backward(*args, **kwargs):
        raise RuntimeError("SageMaker model parallel utilities are unavailable in this environment.")

if version.parse(torch.__version__) >= version.parse("1.6"):
    pass

logger = logging.get_logger(__name__)


def get_leaf_modules_with_grad(module):
    module_list = []
    for name, module in module.named_modules():
        if 'LlamaAttention' in str(type(module)) or 'OPTAttention' in str(type(module)) or 'Qwen2Attention' in str(
                type(module)) or 'Gemma2Attention' in str(type(module)) or 'GemmaAttention' in str(
            type(module)) or 'MistralAttention' in str(type(module)):
            module_list += [module]
    return module_list


class BaseTrainer(Trainer):
    def get_dataloader(self, special_dataset) -> DataLoader:
        """Return the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        from torch.utils.data import DataLoader, RandomSampler
        from transformers.trainer_utils import seed_worker
        data_collator = self.data_collator

        sampler = RandomSampler(special_dataset)

        dataloader_params = {
            "batch_size": 1,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(special_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = sampler
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return self.accelerator.prepare(DataLoader(special_dataset, **dataloader_params))

        # def wanda_data_init(self, dangerous_dataset, safe_dataset):

    def specific_data_init(self, dangerous_dataset):
        print("Loading wanda datasets.")
        self.dangerous_dataloader = self.get_dataloader(dangerous_dataset)
        # self.safe_dataloader = self.get_dataloader(safe_dataset)
        self.data_iter_danferous = iter(self.dangerous_dataloader)
        # self.data_iter_safe = iter(self.safe_dataloader)

    def sample_from_alignment(self, data_type):
        # Get a  batch
        if data_type == 'dangerous':
            data_iter = self.data_iter_danferous
            dataloader = self.dangerous_dataloader
        else:
            pass
            # data_iter = self.data_iter_safe
            # dataloader = self.safe_dataloader
        try:
            batch = next(data_iter)
        except (StopIteration):
            # If the iterator is exhausted, create a new iterator
            data_iter = iter(dataloader)
            batch = next(data_iter)
        return batch

    def check_dataset(self, inputs, status):
        if status == 'alignment':
            inputs = inputs
        else:
            inputs = self.sample_from_alignment(status)
        return inputs

    def switch_active_layers(self, n_layers, probability, total_layers):
        # Randomly select n_layers to activate
        #active_layers_indices = np.argsort(np.array(probability))[-n_layers:]

        active_layers_indices = sorted(np.random.choice(range(total_layers), n_layers, replace=False, p=probability))
        #active_layers_indices = sorted(np.random.choice(range(total_layers), n_layers, replace=False))

        print(f"In Activating layers at indices: {active_layers_indices} for the next steps.", flush=True)
        return active_layers_indices

    def training_step(
            self, model: nn.Module, inputs: dict[str, torch.Tensor | Any], num_items: int | None = None
    ) -> torch.Tensor:
        inputs = self.check_dataset(inputs, status='alignment')
        model.train()
        inputs = self._prepare_inputs(inputs)

        self.layers = get_leaf_modules_with_grad(model)
        self.do_grad_scaling = False

        def step(inputs_, model_):
            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model_, inputs_, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model_, inputs_)
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.do_grad_scaling:
                self.scaler.scale(loss).backward()
            elif self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)
                # print("gere2")
            return loss
            # print("calling sam")

        self.sam_state = {}
        self.sam_state["hooks"] = []
        self.sam_state["gradient"] = {}
        self.sam_state["gradient_special"] = {}
        self.sam_state["gradient_list"] = {}
        self.sam_state["gradient_probability"] = {}
        # self.sam_state["gradient_memory"] = []

        if self.state.global_step % self.args.probability_steps == 0:
            # Get gradient magnitude
            for layer in self.layers:
                for name, param in layer.named_parameters():
                    if 'lora' in name:
                        param.requires_grad = True

            inputs_ = self.check_dataset(inputs, status='dangerous')
            inputs_ = self._prepare_inputs(inputs_)
            self.pre_gradient_magnitude_step(model)
            step(inputs_, model)
            self.after_gradient_magnitude_step(model, title='dangerous')
            model.zero_grad()
            self.probability = self.sam_state["gradient_probability"]['dangerous']
            # print(self.sam_state["gradient_list"]['dangerous'])
            # import os
            # if os.path.exists('gradient_norm.csv'):
            #    df = pd.read_csv('gradient_norm.csv')
            #    df['gradient_norm_%s'%self.state.global_step]=self.sam_state["gradient_list"]['dangerous']
            # else:
            #    df = pd.DataFrame({'gradient_norm_%s'%self.state.global_step:self.sam_state["gradient_list"]['dangerous']})
            # df.to_csv('gradient_norm.csv', mode='w', index=False)

        if self.state.global_step % self.args.lisa_interval_steps == 0:
            self.active_layers_indices = self.switch_active_layers(self.args.lisa_activated_layers,
                                                                   probability=self.probability,
                                                                   total_layers=len(self.layers))

        inputs['activate_layers'] = []
        if len(self.layers) > 26:
            for i in self.active_layers_indices:
                if i != 0:
                    inputs['activate_layers'].append(i - 1)
                inputs['activate_layers'].append(i)
        else:
            inputs['activate_layers'] = self.active_layers_indices

        self.unfreeze_activate_layers()

        self.pre_first_step(model)
        step(inputs, model)
        self.after_first_step(model)
        model.zero_grad()
        self.pre_second_step(model)
        loss = step(inputs, model)
        self.after_second_step(model)

        return loss.detach() / self.args.gradient_accumulation_steps

    @torch.no_grad()
    def unfreeze_activate_layers(self):
        for layer in self.layers:
            for name, param in layer.named_parameters():
                param.requires_grad = False

        for idx in self.active_layers_indices:
            layer = self.layers[idx]
            for name, param in layer.named_parameters():
                if 'lora' in name:
                    param.requires_grad = True

    @torch.no_grad()
    def pre_first_step(self, model):
        def track_gradient_hook(module, grad_input, grad_output):
            # Store the gradients for the current layer
            # self.sam_state["gradient_memory"].append(grad_output[0].detach().numel())
            self.sam_state["gradient"][module] = grad_output[0].detach().clone() / self.args.gradient_accumulation_steps
            # print(grad_output[0])

        def apply_backward_hooks_recursive(module, hook_fn, hooks):
            hook = module.register_backward_hook(hook_fn)
            hooks.append(hook)  # Append the hook to the list

        # Call the function with the initial empty hooks list
        # leaf_modules_with_grad = get_leaf_modules_with_grad(model)
        for idx in self.active_layers_indices:
            layer = self.layers[idx]
            self.sam_state["gradient"][layer] = 0
            apply_backward_hooks_recursive(layer, track_gradient_hook, self.sam_state["hooks"])

    @torch.no_grad()
    def pre_second_step(self, model):
        def purturbation_hook(module, input, output):
            # Modify the output, for example, by adding a perturbatio
            perturbation = self.sam_state["gradient"][module]
            # print(perturbation[0,1,:])
            # # print(output.shape)
            # print(output[0,1,:])
            output[0].data = output[0] + perturbation
            # print(perturbation.shape)
            # print(output.shape)
            return output

        # Register forward hooks for adding perturbation
        def apply_purturbation_hooks_recursive(module, hook_fn, hooks):
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)

        # leaf_modules_with_grad = get_leaf_modules_with_grad(model)
        for idx in self.active_layers_indices:
            layer = self.layers[idx]
            # print(layer._get_name())
            # Apply hooks to all layers, including nested Sequential blocks
            # if self.state.global_step % 100 == 0 and idx > 20:
            apply_purturbation_hooks_recursive(layer, purturbation_hook, self.sam_state["hooks"])

    @torch.no_grad()
    def after_first_step(self, model):
        # print('gradient_activate:', sum(self.sam_state["gradient_memory"]) * 2 / (1024 ** 3))
        for hook in self.sam_state["hooks"]:
            hook.remove()
        self.sam_state["hooks"] = []

        # print(self.sam_state["gradient"].items())
        grad_norm = self._grad_norm(self.sam_state["gradient"])
        # logging.info(grad_norm)
        # logging.info("norm{}".format(grad_norm))
        for module in self.sam_state["gradient"]:
            # grad_norm = self._grad_norm(self.sam_state["gradient"][module])
            grad = self.sam_state["gradient"][module]
            scale = self.args.rho / (grad_norm + 1e-7)
            e_r = (grad) * scale
            self.sam_state["gradient"][module] = e_r.detach().clone()
            # print(module)
        #     print( torch.norm(self.sam_state["e_r"][module]) )
        # print(len(self.sam_state["e_r"]))

    @torch.no_grad()
    def after_second_step(self, model):
        # disable hook here
        # for module in self.sam_state["e_r"]:
        #     module.weight.data -= self.sam_state["e_r"][module]
        for hook in self.sam_state["hooks"]:
            hook.remove()
        self.sam_state["hooks"] = []
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

    @torch.no_grad()
    def _grad_norm(self, poison_grads_representation):
        norm = torch.norm(
            torch.stack([

                (poison_grads_representation[name]).norm(p=2)

                # ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for name in poison_grads_representation
            ]),
            p=2
        )
        # norm = ( poison_grads_representation ).norm(p=2)
        return norm

    @torch.no_grad()
    def pre_gradient_magnitude_step(self, model):

        def track_gradient_hook(module, grad_input, grad_output):
            # Store the gradients for the current layer
            self.sam_state["gradient_special"][module] = grad_output[
                                                             0].detach().clone() / self.args.gradient_accumulation_steps

        # Register forward hooks for adding perturbation
        def apply_backward_hooks_recursive(module, hook_fn_2, hooks):
            # hook1 = module.register_forward_hook(hook_fn_1)
            hook2 = module.register_backward_hook(hook_fn_2)
            # hooks.append(hook1)
            hooks.append(hook2)

        # leaf_modules_with_grad = get_leaf_modules_with_grad(model)
        for layer in self.layers:
            # apply_wanda_hooks_recursive(layer, wanda_hook, track_gradient_hook, self.sam_state["hooks"])
            apply_backward_hooks_recursive(layer, track_gradient_hook, self.sam_state["hooks"])

    @torch.no_grad()
    def after_gradient_magnitude_step(self, model, title):
        self.sam_state["gradient_list"][title] = []
        for layer in self.layers:
            self.sam_state["gradient_list"][title].append(
                torch.norm(self.sam_state["gradient_special"][layer], 2).item())
            # self.sam_state["gradient_list"][title].append(self.sam_state["gradient_special"][layer].abs().mean().item())

        # calculate the probability of layers being selected.
        total = sum(self.sam_state["gradient_list"][title])
        self.sam_state["gradient_probability"][title] = [i / total for i in self.sam_state["gradient_list"][title]]

        for hook in self.sam_state["hooks"]:
            hook.remove()
        self.sam_state["hooks"] = []
        # self.sam_state["embedding_norm"] = {}
        self.sam_state["gradient_special"] = {}


class Vaccine(Trainer):
    def training_step(
            self, model: nn.Module, inputs: dict[str, torch.Tensor | Any], num_items: int | None = None
    ) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)
        self.do_grad_scaling = False

        def step():
            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.do_grad_scaling:
                self.scaler.scale(loss).backward()
            elif self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)
                # print("gere2")
            return loss
            # print("calling sam")

        # gradient_weight_optimizer_memory = calculate_gradient_weight_optimizer_memory(model)
        self.sam_state = {}
        self.sam_state["hooks"] = []
        self.sam_state["gradient"] = {}
        # self.sam_state["gradient_memory"] = []
        self.pre_first_step(model)
        step()
        self.after_first_step(model)
        model.zero_grad()
        self.pre_second_step(model)
        loss = step()
        self.after_second_step(model)
        # for param in model.parameters():
        #     if param.grad is not None:
        #         param.grad*= 1/2

        # else:
        #     loss = step()
        return loss.detach() / self.args.gradient_accumulation_steps

    @torch.no_grad()
    def pre_first_step(self, model):
        def track_gradient_hook(module, grad_input, grad_output):
            # Store the gradients for the current layer
            self.sam_state["gradient"][module] = grad_output[0].detach().clone() / self.args.gradient_accumulation_steps
            # self.sam_state["gradient_memory"].append(grad_output[0].detach().numel())
            # print(grad_output[0])

        def apply_backward_hooks_recursive(module, hook_fn, hooks):
            hook = module.register_backward_hook(hook_fn)
            hooks.append(hook)  # Append the hook to the list

        # Call the function with the initial empty hooks list
        leaf_modules_with_grad = get_leaf_modules_with_grad(model)
        for layer in leaf_modules_with_grad:
            self.sam_state["gradient"][layer] = 0
            apply_backward_hooks_recursive(layer, track_gradient_hook, self.sam_state["hooks"])

    @torch.no_grad()
    def pre_second_step(self, model):
        def purturbation_hook(module, input, output):
            # Modify the output, for example, by adding a perturbatio
            perturbation = self.sam_state["gradient"][module]
            # print(perturbation[0,1,:])
            # # print(output.shape)
            # print(output[0,1,:])
            output[0].data = output[0] + perturbation
            # print(perturbation.shape)
            # print(output.shape)
            return output

        # Register forward hooks for adding perturbation
        def apply_purturbation_hooks_recursive(module, hook_fn, hooks):
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)

        leaf_modules_with_grad = get_leaf_modules_with_grad(model)
        for layer in leaf_modules_with_grad:
            # print(layer._get_name())
            # Apply hooks to all layers, including nested Sequential blocks
            apply_purturbation_hooks_recursive(layer, purturbation_hook, self.sam_state["hooks"])

    @torch.no_grad()
    def after_first_step(self, model):
        # print(sum(self.sam_state["gradient_memory"]) * 2 / (1024 ** 3))
        for hook in self.sam_state["hooks"]:
            hook.remove()
        self.sam_state["hooks"] = []
        # self.sam_state["gradient_memory"] = []

        # print(self.sam_state["gradient"].items())
        # print(self.sam_state["gradient"])
        grad_norm = self._grad_norm(self.sam_state["gradient"])
        # logging.info(grad_norm)
        # logging.info("norm{}".format(grad_norm))
        for module in self.sam_state["gradient"]:
            # grad_norm = self._grad_norm(self.sam_state["gradient"][module])
            grad = self.sam_state["gradient"][module]
            scale = self.args.rho / (grad_norm + 1e-7)
            e_r = (grad) * scale
            self.sam_state["gradient"][module] = e_r.detach().clone()
            # print(module)
        #     print( torch.norm(self.sam_state["e_r"][module]) )
        # print(len(self.sam_state["e_r"]))

    @torch.no_grad()
    def after_second_step(self, model):
        # disable hook here
        # for module in self.sam_state["e_r"]:
        #     module.weight.data -= self.sam_state["e_r"][module]
        for hook in self.sam_state["hooks"]:
            hook.remove()
        self.sam_state["hooks"] = []
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

    @torch.no_grad()
    def _grad_norm(self, poison_grads_representation):
        norm = torch.norm(
            torch.stack([

                (poison_grads_representation[name]).norm(p=2)

                # ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for name in poison_grads_representation
            ]),
            p=2
        )
        # norm = ( poison_grads_representation ).norm(p=2)
        return norm


class RandomVaccineTrainer(Trainer):
    def training_step(
            self, model: nn.Module, inputs: dict[str, torch.Tensor | Any], num_items: int | None = None
    ) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        def step():
            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.do_grad_scaling:
                self.scaler.scale(loss).backward()
            elif self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)
                # print("gere2")
            return loss

        self.sam_state = {}
        self.sam_state["hooks"] = []
        self.sam_state["gradient"] = {}
        self.pre_second_step(model)
        loss = step()
        self.after_second_step(model)
        # for param in model.parameters():
        #     if param.grad is not None:
        #         param.grad*= 1/2

        # else:
        #     loss = step()
        return loss.detach() / self.args.gradient_accumulation_steps

    @torch.no_grad()
    def pre_second_step(self, model):
        def purturbation_hook(module, input, output):
            # Modify the output, for example, by adding a perturbatio
            # print(perturbation[0,1,:])
            # # print(output.shape)
            # print(output[0,1,:])
            variance = self.args.rho
            # Generate samples from a Gaussian distribution
            gaussian_samples = variance ** (1 / 2) * torch.randn_like(output[0])
            output[0].data = output[0] + gaussian_samples
            # print(output.shape)
            return output

        # Register forward hooks for adding perturbation
        def apply_purturbation_hooks_recursive(module, hook_fn, hooks):
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)

        leaf_modules_with_grad = get_leaf_modules_with_grad(model)
        for layer in leaf_modules_with_grad:
            # print(layer._get_name())
            # Apply hooks to all layers, including nested Sequential blocks
            apply_purturbation_hooks_recursive(layer, purturbation_hook, self.sam_state["hooks"])

    @torch.no_grad()
    def after_second_step(self, model):
        # disable hook here
        # for module in self.sam_state["e_r"]:
        #     module.weight.data -= self.sam_state["e_r"][module]
        for hook in self.sam_state["hooks"]:
            hook.remove()
        self.sam_state["hooks"] = []
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

    @torch.no_grad()
    def _grad_norm(self, poison_grads_representation):
        norm = torch.norm(
            torch.stack([
                (poison_grads_representation[name]).norm(p=2)
                for name in poison_grads_representation
            ]),
            p=2
        )
        # norm = ( poison_grads_representation ).norm(p=2)
        return norm


class FITrainer(Trainer):

    def init(self, model):
        self.initial_weights = {}
        for name, module in model.named_modules():
            if "lora" in name and len(list(module.children())) == 0 and isinstance(module, torch.nn.Linear):
                self.initial_weights[module] = module.weight.data.detach().clone()
        self.round = 0

    def training_step(
            self, model: nn.Module, inputs: dict[str, torch.Tensor | Any], num_items: int | None = None
    ) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        def step():
            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)

            reg = 0
            for name, module in model.named_modules():
                if "lora" in name and len(list(module.children())) == 0 and isinstance(module, torch.nn.Linear):
                    reg += self.args.lamb * torch.sum(
                        self.fisher_vector[module] * torch.square(module.weight - self.initial_weights[module]))
                    # reg += self.args.lamb * torch.sum(torch.square(module.weight -self.initial_weights[module] ))
            # print(reg)
            loss += reg
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.do_grad_scaling:
                self.scaler.scale(loss).backward()
            elif self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)
            return loss

        if self.round == 0:
            self.fisher_vector = {module: 0 for name, module in model.named_modules() if
                                  "lora" in name and len(list(module.children())) == 0 and isinstance(module,
                                                                                                      torch.nn.Linear)}
            eval_dataloader = self.get_eval_dataloader(self.eval_dataset)
            for stepsize, old_inputs in enumerate(eval_dataloader):
                # Update the observed num examples
                # print(inputs)
                model.zero_grad()
                old_inputs = self._prepare_inputs(old_inputs)
                with self.compute_loss_context_manager():
                    loss = self.compute_loss(model, old_inputs)
                self.accelerator.backward(loss)
                for name, module in model.named_modules():
                    if "lora" in name and len(list(module.children())) == 0 and isinstance(module, torch.nn.Linear):
                        self.fisher_vector[module] += torch.square(module.weight.grad.data.detach().clone())
                        # print(self.fisher_vector[module])
                print(loss)

        loss = step()
        # print( sum([torch.norm(self.sam_state ["gradient"][module]) for module in self.sam_state ["gradient"]  ]))
        # leaf_modules_with_grad = get_leaf_modules_with_grad(model)
        # for module in leaf_modules_with_grad:
        #     # print(module.q_proj.lora_A["default"])
        #     module.weight.grad*= (1-self.masks[index])
        #     index+=1
        self.round += 1
        return loss.detach() / self.args.gradient_accumulation_steps


class KLTrainer(Trainer):

    def init(self, model):
        import copy
        self.teacher_model_w = copy.deepcopy(model.state_dict())
        self.round = 0

    def training_step(
            self, model: nn.Module, inputs: dict[str, torch.Tensor | Any], num_items: int | None = None
    ) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        def step():
            temp = {name: copy.deepcopy(param) for name, param in model.named_parameters() if param.requires_grad}
            with torch.no_grad():
                model.load_state_dict(self.teacher_model_w)
                teacher_outputs = self.model(**inputs,
                                             return_dict=True,
                                             use_cache=False,
                                             )
                model.load_state_dict(temp, strict=False)
            student_ouput = model(**inputs,
                                  return_dict=True,
                                  use_cache=False,
                                  )
            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)

            import torch.nn.functional as F
            # Compute KL divergence
            kl_loss = self.args.lamb * torch.nn.KLDivLoss(reduction="batchmean")(F.log_softmax(student_ouput[1], dim=1),
                                                                                 F.softmax(teacher_outputs[1].detach(),
                                                                                           dim=1))
            # reg += self.args.lamb * torch.sum(torch.square(module.weight -self.initial_weights[module] ))
            # kl_loss = torch.mean(student_ouput[1])
            print(kl_loss)
            loss += kl_loss
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.do_grad_scaling:
                self.scaler.scale(loss).backward()
            elif self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)
            return loss

        loss = step()
        self.round += 1
        return loss.detach() / self.args.gradient_accumulation_steps


class TarTrainer(Trainer):
    def get_dataloader(self, special_dataset) -> DataLoader:
        """Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        from torch.utils.data import DataLoader, RandomSampler
        from transformers.trainer_utils import seed_worker
        data_collator = self.data_collator

        sampler = RandomSampler(special_dataset)

        dataloader_params = {
            "batch_size": 10,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(special_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = sampler
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return self.accelerator.prepare(DataLoader(special_dataset, **dataloader_params))

        # def wanda_data_init(self, dangerous_dataset, safe_dataset):

    def specific_data_init(self, dangerous_dataset, model):
        print("Loading wanda datasets.")
        self.dangerous_dataloader = self.get_dataloader(dangerous_dataset)
        # self.safe_dataloader = self.get_dataloader(safe_dataset)
        self.data_iter_danferous = iter(self.dangerous_dataloader)
        # self.data_iter_safe = iter(self.safe_dataloader)
        self.retain_model = copy.deepcopy(model)

    def sample_from_alignment(self, data_type):
        # Get a  batch
        if data_type == 'dangerous':
            data_iter = self.data_iter_danferous
            dataloader = self.dangerous_dataloader
        else:
            pass
            # data_iter = self.data_iter_safe
            # dataloader = self.safe_dataloader
        try:
            batch = next(data_iter)
        except (StopIteration):
            # If the iterator is exhausted, create a new iterator
            data_iter = iter(dataloader)
            batch = next(data_iter)
        return batch

    def check_dataset(self, inputs, status):
        if status == 'alignment':
            inputs = inputs
        else:
            inputs = self.sample_from_alignment(status)
        return inputs

    def log_p_loss(self, logits: torch.Tensor, labels: torch.Tensor, vocab_size: int
                   ) -> torch.Tensor:
        """Compute the log probability loss for a language model.

        This function calculates the cross-entropy loss between the predicted logits
        and the true labels, typically used in language modeling tasks.

        Args:
            logits (torch.Tensor): The predicted logits from the model, typically of shape
                                   (batch_size, sequence_length, vocab_size).
            labels (torch.Tensor): The true labels, typically of shape
                                   (batch_size, sequence_length).
            vocab_size (int): The size of the vocabulary.

        Returns:
            torch.Tensor: The computed loss as a scalar tensor.
        """
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        return loss

    def _filter_inputs(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Filter the input dictionary to keep only specific keys.

        This function takes a dictionary of input tensors and returns a new dictionary
        containing only the keys 'input_ids', 'attention_mask', and 'labels' if they exist
        in the original dictionary.

        Args:
            inputs (Dict[str, torch.Tensor]): A dictionary containing input tensors.

        Returns:
            Dict[str, torch.Tensor]: A filtered dictionary containing only the specified keys.
        """
        return {
            k: v
            for k, v in inputs.items()
            if k in ["input_ids", "attention_mask", "labels"]
        }

    def get_distributed_random_number(self, accelerator: Accelerator):
        random_number = torch.rand(1).to(accelerator.device)
        accelerator.wait_for_everyone()
        return random_number.item()
    def distributed_sample_adversary_lr(self, adversary_lr_samples, accelerator):
        rand_num = self.get_distributed_random_number(accelerator)
        adversary_lr = adversary_lr_samples[
            math.floor(rand_num * len(adversary_lr_samples))
        ]
        return adversary_lr

    def training_step(
            self, model: nn.Module, inputs: dict[str, torch.Tensor | Any], num_items: int | None = None
    ) -> torch.Tensor:
        inputs = self.check_dataset(inputs, status='alignment')
        model.train()
        inputs = self._prepare_inputs(inputs)
        harmful_inputs = self.check_dataset(inputs, status='dangerous')
        harmful_inputs = self._prepare_inputs(harmful_inputs)

        self.layers = get_leaf_modules_with_grad(model)
        self.do_grad_scaling = False

        # adversary_lr_samples = [2e-6, 2e-5, 4e-5]
        #
        # adversary_lr = self.distributed_sample_adversary_lr(
        #     adversary_lr_samples, self.accelerator
        # )
        # print(adversary_lr)
        #
        # inner_optimizer = torch.optim.AdamW(model.parameters(), lr=adversary_lr)
        # inner_optimizer = self.accelerator.prepare_optimizer(inner_optimizer)
        # inner_scheduler = None

        def step():

            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, harmful_inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.do_grad_scaling:
                self.scaler.scale(loss).backward()
            elif self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)

            stored_grads = {name: param.grad.data.clone() for name, param in model.named_parameters() if
                            param.requires_grad}
            # inner_optimizer.step()
            # model.zero_grad(set_to_none=True)

            for name, param in model.named_parameters():
                if param.requires_grad:
                    # param.data -= self.args.rho*stored_grads[name]/grad_norm
                    param.data -= 0.01 * stored_grads[name]

            with self.compute_loss_context_manager():
                loss2 = self.compute_loss(model, inputs)
            if self.use_apex:
                with amp.scale_loss(loss2, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss2)
            stored_grads_tr = {name: param.grad.data.clone() for name, param in model.named_parameters() if
                               param.requires_grad}

            model.zero_grad()

            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         param.data += 0.1 * stored_grads[name]
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # param.data -= self.args.rho*stored_grads[name]/grad_norm
                    param.data += 0.01 * stored_grads[name]

            with self.compute_loss_context_manager():
                # loss3 = self.compute_loss(model, inputs)
                # _x_r = self._filter_inputs(inputs)
                # model_outputs = model(**_x_r, output_hidden_states=True)
                # with torch.no_grad():
                #     base_model_outputs = self.retain_model(**_x_r, output_hidden_states=True)
                # loss3 = self.log_p_loss(model_outputs.logits, _x_r.get("labels"), model.vocab_size)
                loss3 = self.compute_loss(model, inputs)

                # loss4 = loss3 + torch.mean(torch.stack([
                #     (torch.norm(base_hidden - model_hidden, dim=-1)).mean()
                #     for base_hidden, model_hidden in zip(
                #         base_model_outputs.hidden_states, model_outputs.hidden_states)]))
            if self.use_apex:
                with amp.scale_loss(loss3, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss3)

            # tr_gradient + loss4_gradient
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # param.grad += stored_grads[name]
                    param.data.grad = param.grad.data + 2 * stored_grads_tr[name]

            return loss3

        loss = step()
        return loss.detach() / self.args.gradient_accumulation_steps


class RepNoiseTrainer(Trainer):
    def init(self, harmful_dataset):
        # reploss needs standard dataset, load alpaca here
        from torch.utils.data import DataLoader, RandomSampler
        from transformers.trainer_utils import seed_worker
        data_collator = self.data_collator
        sampler = RandomSampler(harmful_dataset)
        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }
        if not isinstance(harmful_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = sampler
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
        self.harmful_dataloader = self.accelerator.prepare(DataLoader(harmful_dataset, **dataloader_params))

    def training_step(
            self, model: nn.Module, inputs: dict[str, torch.Tensor | Any], num_items: int | None = None
    ) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)
        # Get an iterator from the DataLoader
        data_iter = iter(self.harmful_dataloader)
        # Get the next batch
        harmful_inputs = next(data_iter)
        harmful_inputs = self._prepare_inputs(harmful_inputs)

        def step():
            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                # loss = self.compute_loss(model, inputs)
                loss = rep_noise_loss(model, harmful_inputs, inputs, beta=self.args.lamb, alpha=self.args.rho)
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)
                # print("gere2")
            return loss

        loss = step()
        # with torch.no_grad():
        #     if self.round>=self.warm_up_round:
        #         for name, param in model.named_parameters():
        #             if param.requires_grad:
        #                 param.grad *= self.mask[name]

        return loss.detach() / self.args.gradient_accumulation_steps
