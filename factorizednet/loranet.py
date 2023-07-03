import copy
import time

import torch.nn as nn
import pandas as pd

from factorizednet.factorized_module.lora_linear import LoraLinear


class LoraNet(nn.Module):
    def __init__(self, model, ignore=None, rank=1, verbose=False, **kwargs):
        super(LoraNet, self).__init__()
        self.replacements = {
            "Conv1D": LoraLinear
        }
        self.rank = rank
        self.ignore = ignore if ignore is not None else list()
        self._report = pd.DataFrame()
        self._factorized_model = self._make_lora_model(copy.deepcopy(model), rank=rank)
        self.run_stats = dict()

    def load_state_dict(self, state_dict):
        state_dict_fixed = {
            k.replace("_factorized_model.", ""): v for k, v in state_dict.items()
        }
        self._factorized_model.load_state_dict(state_dict_fixed)

    @property
    def factorized_model(self):
        return self._factorized_model

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
    def get_num_params(module, trainable=True):
        return int(sum(p.numel() for p in module.parameters() if p.requires_grad == trainable) / 1e6)

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

    def _make_lora_model(self, model, rank=1):
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

            if ltype in self.replacements.keys() and name not in self.ignore:
                lora_module = self.replacements[ltype].from_module(layer, rank=rank)
                replacement_address = self._parse_model_addr(name)

                lora_trainable = self.get_num_params(lora_module, trainable=True)
                lora_fixed = self.get_num_params(lora_module, trainable=False)

                self._report.at[name, 'type+'] = type(lora_module).__name__
                self._report.at[name, 'train+'] = lora_trainable
                self._report.at[name, 'fix+'] = lora_fixed

                self._set_module(model, replacement_address, lora_module)

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

        if isinstance(replacement_addr_list[-1], int):
            self._get_module(model, replacement_addr_list[:-1])[replacement_addr_list[-1]] = replacement_layer
        else:
            # delattr(self._get_module(model, replacement_addr_list[:-1]), replacement_addr_list[-1])
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
