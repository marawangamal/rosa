import copy
import math
import time
import torch

import torch.nn as nn
import pandas as pd

from factorizednet.factorized_module.rosa_linear import RosaLinear


class RosaNet(nn.Module):
    def __init__(self, model, ignore=None,
                 delta=None, target=None, steps=None, warmup_steps=0,
                 rank=None, sample_method=None, collapse_fixed=None, make_copy=False, **kwargs):
        super(RosaNet, self).__init__()
        assert (target is None) or (delta is None), "target and delta cannot be specified together"
        assert (target is not None) == (steps is not None), "target and steps must be specified"

        self.factorized_modules = {
            "Conv1D": RosaLinear  # Name of the linear module found in GPT2 models
        }
        self.ignore = ignore if ignore is not None else list()
        self.run_stats = dict()
        self.make_copy = make_copy

        # PEFT parameters
        self.rank = rank
        self.sample_method = sample_method
        self.collapse_fixed = collapse_fixed

        if self.make_copy:
            self._factorized_model = self._factorize_model(copy.deepcopy(model))
        else:
            self._factorized_model = self._factorize_model(model)

    def load_state_dict(self, state_dict):
        for name, layer in self._factorized_model.named_modules():
            if any([isinstance(layer, v) for k, v in self.factorized_modules.items()]):
                prefix = ".".join(["_factorized_model", name]) + "."
                layer_state_dict = {k.replace(prefix, ""): v for k, v in state_dict.items() if k.startswith(prefix)}
                factorized_module = layer.from_state_dict(
                    layer_state_dict
                )
                replacement_address = self._parse_model_addr(name)
                self._set_module(self._factorized_model, replacement_address, factorized_module)

    @property
    def factorized_model(self):
        return self._factorized_model

    @staticmethod
    def _get_module_params(module):
        return sum([p.numel() for p in module.parameters()])

    def sample_trainable(self, rank=None, sample_method=None, collapse_fixed=None):
        """ Mask gradients of all factorized modules in the model

        Args:
            rank: Ratio of gradients to be masked
            sample_method: Method to mask gradients. 'random' or 'top'
            collapse_fixed: Whether to collapse fixed and trainable matrices

        Returns:
            RosaNet object with low rank trainable matrices
        """

        assert sample_method != None or self.sample_method != None, "sample_method must be specified"
        assert rank != None or self.rank != None, "rank must be specified"
        assert collapse_fixed != None or self.collapse_fixed != None, "collapse_fixed must be specified"

        rank = rank if rank is not None else self.rank
        sample_method = sample_method if sample_method is not None else self.sample_method
        collapse_fixed = collapse_fixed if collapse_fixed is not None else self.collapse_fixed
        for name, layer in self._factorized_model.named_modules():
            if any([isinstance(layer, v) for k, v in self.factorized_modules.items()]):
                resampled_module = layer.sample_trainable(
                    rank=rank, method=sample_method, collapse_fixed=collapse_fixed
                )
                replacement_address = self._parse_model_addr(name)
                self._set_module(self._factorized_model, replacement_address, resampled_module)
        return self

    @staticmethod
    def get_num_params(module, trainable=True):
        return int(sum(p.numel() for p in module.parameters() if p.requires_grad == trainable) / 1e6)

    def get_report(self, verbose=False):
        for name, layer in self._factorized_model.named_modules():
            trainable = self.get_num_params(layer, trainable=True)
            fixed = self.get_num_params(layer, trainable=False)

            # self._report.at[name, 'new name'] = name
            self._report.at[name, 'type_'] = type(layer).__name__
            self._report.at[name, 'train_'] = trainable
            self._report.at[name, 'fix_'] = fixed
            self._report.at[name, 'tot_'] = trainable + fixed

        if verbose:
            self._report['train'] = pd.to_numeric(self._report['train'], errors='coerce')
            self._report['train+'] = pd.to_numeric(self._report['train+'], errors='coerce')
            self._report['tcr'] = self._report['train'] / self._report['train+']
            self._report['fix'] = pd.to_numeric(self._report['fix'], errors='coerce')
            self._report['fix+'] = pd.to_numeric(self._report['fix+'], errors='coerce')
            self._report['fcr'] = self._report['fix'] / self._report['fix+']
            cols = ['type', 'train', 'train+', 'new train', 'tcr', 'fix', 'fix+', 'new fix', 'fcr']
        else:
            self._report['train'] = pd.to_numeric(self._report['train'], errors='coerce')
            self._report['train+'] = pd.to_numeric(self._report['train_'], errors='coerce')
            self._report['tcr'] = self._report['train'] / self._report['train_']
            self._report['fix'] = pd.to_numeric(self._report['fix'], errors='coerce')
            self._report['fix+'] = pd.to_numeric(self._report['fix_'], errors='coerce')
            self._report['fcr'] = self._report['fix'] / self._report['fix+']
            cols = ['type', 'train', 'train_', 'tcr', 'fix', 'fix_', 'fcr']

        return self._report[cols]

    @staticmethod
    def _build_report(model):

        report_dict = {
            "name": list(),
            "type": list(),
            "train": list(),
            "fix": list(),
            "type+": list(),
            "train+": list(),
            "fix+": list(),
        }

        for name, layer in model.named_modules():
            ltype = type(layer).__name__

            report_dict['name'].append(name)
            report_dict['type'].append(ltype)
            report_dict['train'].append("N/A")
            report_dict['fix'].append("N/A")
            report_dict["type+"].append("N/A")
            report_dict["train+"].append("N/A")
            report_dict["fix+"].append("N/A")

        # Create DataFrame from the dictionary
        df = pd.DataFrame(report_dict)

        # Set the 'name' column as the index
        df.set_index('name', inplace=True)

        return df

    def _factorize_model(self, model):
        # start_time = time.time()
        self._report = self._build_report(model)
        for name, layer in model.named_modules():
            ltype = type(layer).__name__

            original_params_trainable = self.get_num_params(layer, trainable=True)
            original_params_fixed = self.get_num_params(layer, trainable=False)

            # self._report.at[name, 'name'] = name
            self._report.at[name, 'type'] = type(layer).__name__
            self._report.at[name, 'train'] = original_params_trainable
            self._report.at[name, 'fix'] = original_params_fixed

            if ltype in self.factorized_modules.keys() and name not in self.ignore:
                factorized_module = self.factorized_modules[ltype].from_module(layer)
                replacement_address = self._parse_model_addr(name)

                factorized_trainable = self.get_num_params(factorized_module, trainable=True)
                facrroized_fixed = self.get_num_params(factorized_module, trainable=False)

                self._report.at[name, 'type+'] = type(factorized_module).__name__
                self._report.at[name, 'train+'] = factorized_trainable
                self._report.at[name, 'fix+'] = facrroized_fixed

                self._set_module(model, replacement_address, factorized_module)
        # end_time = time.time()
        # self.run_stats['factorize_time'] = end_time - start_time
        return model

    def _get_module(self, parent: nn.Module, replacement_addr_list: list) -> nn.Module:
        """ Recursive function used to access child modules from a parent nn.Module object

        Args:
            replacement_addr_list: specifies how to access target object from ancestor object.
                ['layer1', 0, 'conv2']
        Returns:
            target object/layer to be replaced.
        """

        if len(replacement_addr_list) == 0:
            return parent
        else:
            attr = replacement_addr_list.pop(0)

            # attr can be accessible in two ways
            child = parent[attr] if isinstance(attr, int) else getattr(parent, attr)
            return self._get_module(child, replacement_addr_list)

    def _set_module(self, model: nn.Module, replacement_addr_list: list,
                    replacement_layer: nn.Module) -> None:
        """ Sets attribute of `model` accessed via `replacement_addr_list` to `replacement_layer` """

        # import pdb; pdb.set_trace()

        if isinstance(replacement_addr_list[-1], int):
            # print("Mem Alloc Before: {} Mem Reserved Before: {}".format(torch.cuda.memory_allocated(), torch.cuda.memory_reserved()))
            self._get_module(model, replacement_addr_list[:-1])[replacement_addr_list[-1]] = replacement_layer
        # print("**")
        else:
            # print("Mem Alloc Before: {} Mem Reserved Before: {}".format(
            # 	torch.cuda.memory_allocated(), torch.cuda.memory_reserved())
            # )
            # delattr(self._get_module(model, replacement_addr_list[:-1]), replacement_addr_list[-1])
            # print("Mem Alloc After: {} Mem Reserved After: {}".format(
            # 	torch.cuda.memory_allocated(), torch.cuda.memory_reserved())
            # )
            setattr(self._get_module(model, replacement_addr_list[:-1]), replacement_addr_list[-1], replacement_layer)

    @staticmethod
    def _parse_model_addr(access_str: str) -> list:
        """ Parses path to child from a parent. E.g., layer1.0.conv2 ==> ['layer1', 0, 'conv2'] """
        parsed = access_str.split('.')
        for i in range(len(parsed)):
            try:
                parsed[i] = int(parsed[i])
            except ValueError:
                pass
        return parsed

    def forward(self, *args, **kwargs):
        return self._factorized_model(*args, **kwargs)
