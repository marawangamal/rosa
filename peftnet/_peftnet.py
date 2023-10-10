import re

import torch.nn as nn
import pandas as pd


class PEFTNet(nn.Module):
    def __init__(
            self,
            model: nn.Module,
            peft_map: dict,
            peft_kwargs: dict = None,
            ignore_regex: str = None,
            *args, **kwargs
    ):
        """ Abstract class for PEFT models. PEFT models are models that can be factorized and merged.

        Args:
            model: model to be factorized
            peft_map: dictionary mapping module types to replacement modules
            peft_kwargs: dictionary mapping module types to kwargs for replacement modules
            ignore_regex: names of layers to ignore

        Notes:
            - Modules in `peft_map` dict will be replaced
            - kwargs are passed to replacement module `from_module` method

        """
        super().__init__(*args, **kwargs)

        # Set default values
        self.ignore_regex = ignore_regex
        self.peft_map = peft_map
        self.peft_kwargs = peft_kwargs if peft_kwargs is not None else {k: dict() for k in peft_map.keys()}

        # Make modules non-trainable
        for param in model.parameters():
            param.requires_grad = False

        self.peft_model = model
        self.apply_peft()


    def apply_peft(self) -> 'PEFTNet':
        """Replace linear modules with peft modules"""
        if self.ignore_regex is not None:
            condition = lambda lyr, name: type(lyr).__name__ in self.peft_map.keys() and \
                                          not bool(re.match(self.ignore_regex, name))
        else:
            condition = lambda lyr, name: type(lyr).__name__ in self.peft_map.keys()
        replacement_function = lambda lyr: self.peft_map[type(lyr).__name__].from_module(
            lyr, **self.peft_kwargs[type(lyr).__name__]
        )
        return self._replace(condition, replacement_function)

    def merge(self) -> 'PEFTNet':
        """Apply `merge` on peft modules"""
        condition = lambda lyr, name: any([isinstance(lyr, m) for m in self.peft_map.values()])
        replacement_function = lambda lyr: lyr.merge()
        return self._replace(condition, replacement_function)

    def factorize(self) -> 'PEFTNet':
        """Apply `factorize` on peft modules. If a module is already factorized, it will be merged and re-factorized"""
        condition = lambda lyr, name: any([isinstance(lyr, m) for m in self.peft_map.values()])
        replacement_function = lambda lyr: lyr.factorize()
        return self._replace(condition, replacement_function)

    def _replace(self, condition: callable, replacement_function: callable) -> 'PEFTNet':
        """Replace modules that satisfy a condition using a replacement function"""
        for name, layer in self.peft_model.named_modules():
            if condition(layer, name):
                replacement_module = replacement_function(layer)
                replacement_address = self._parse_model_addr(name)
                self._set_module(self.peft_model, replacement_address, replacement_module)

        return self

    @classmethod
    def get_report(cls, model) -> str:
        """Get report on factorized model"""

        safe_div = lambda x, y: x / y if y != 0 else 0

        # Trainable params table
        df = pd.DataFrame()
        for name, layer in model.named_modules():
            params_dict = cls.get_num_params(layer)

            df.at[name, 'name'] = name
            df.at[name, 'type'] = type(layer).__name__
            df.at[name, '# train'] = params_dict['trainable']
            df.at[name, '# fixed'] = params_dict['fixed']
            df.at[name, 'total'] = params_dict['total']
            df.at[name, '% train'] = round(safe_div(params_dict['trainable'], params_dict['total']) * 100, 2)

        # Set the 'name' column as the index
        df.set_index('name', inplace=True)

        # Return string
        return df.to_string()

    @classmethod
    def get_num_params(cls, model: nn.Module) -> dict:
        params_dict = {k: 0 for k in ["trainable", "fixed", "total"]}
        for p in model.parameters():
            params_dict['total'] += p.numel()
            if p.requires_grad:
                params_dict['trainable'] += p.numel()
            else:
                params_dict['fixed'] += p.numel()

        params_dict = {k: v / 1e6 for k, v in params_dict.items()}
        return params_dict

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

    def _set_module(self, model: nn.Module, replacement_addr_list: list, replacement_layer: nn.Module) -> None:
        """ Sets attribute of `model` accessed via `replacement_addr_list` to `replacement_layer` """
        if isinstance(replacement_addr_list[-1], int):
            self._get_module(model, replacement_addr_list[:-1])[replacement_addr_list[-1]] = replacement_layer
        else:
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

    def __getattr__(self, name):
        try:
            # Try to get attribute from PEFTNet itself first
            return super(PEFTNet, self).__getattr__(name)
        except AttributeError:
            # If not found, try to get it from the internal peft_model
            return getattr(self.peft_model, name)

    def __hasattr__(self, name):
        # Check if PEFTNet has the attribute
        if name in self.__dict__:
            return True
        # If not, check the internal peft_model
        return hasattr(self.peft_model, name)

    def forward(self, *args, **kwargs):
        return self.peft_model(*args, **kwargs)
