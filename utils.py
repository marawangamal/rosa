import pickle
import time
from csv import writer

import torch
import yaml
from enum import Enum

import evaluate

import numpy as np


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def test_latency(model, device='cpu', inp=torch.randn(32, 3, 224, 224), iters=100):
    """Test latency of the argument model."""

    if not isinstance(device, torch.device):
        assert device in ['cpu', 'cuda:0']

    torch_device = torch.device(device)
    model = model.to(torch_device)
    dummy_input = inp.to(torch_device)
    latency = np.zeros((iters, 1))

    with torch.no_grad():
        if device == 'cpu':
            # Warm up
            for _ in range(10):
                _ = model(dummy_input)

            # Measure latency.
            for rep in range(iters):
                start = time.time()
                _ = model(dummy_input)
                elapsed = time.time() - start
                latency[rep] = elapsed

        else:
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

            # Warm up
            for _ in range(10):
                _ = model(dummy_input)

            # Measure latency.
            with torch.no_grad():
                for rep in range(iters):
                    starter.record()
                    _ = model(dummy_input)
                    ender.record()
                    torch.cuda.synchronize()
                    elapsed = starter.elapsed_time(ender)
                    latency[rep] = elapsed

    return np.mean(latency), np.std(latency)


def get_params(rmodel):
    n_params = 0
    for name, param in rmodel.named_parameters():
        n_params += param.numel()
    return n_params


def save_object(obj, filename):
    with open(filename, 'wb') as out_file:
        pickle.dump(obj, out_file, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    with open(filename, 'rb') as inp_file:  # Overwrites any existing file.
        out = pickle.load(inp_file)
    return out


def get_num_params(model):
    return sum(p.numel() for p in model.parameters())


def saveckpt(model, epoch, optimizer):
    pass


def get_yaml_dict(yaml_path="configs.yaml"):
    with open(yaml_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def get_experiment_name(configs, abbrevs=None):
    if abbrevs is None:
        abbrevs = {}

    for key, value in configs.items():
        if isinstance(value, dict):
            get_experiment_name(value, abbrevs)
        else:
            i = 1
            while i <= len(key):
                if key[:i] not in abbrevs:
                    abbrevs[key[:i]] = str(value).replace(" ", "").replace(",", "_").replace("[", "").replace("]", "")
                    break
                i += 1

                if i == len(key) + 1:
                    raise ValueError("Could not find a suitable abbreviation for key: {}".format(key))

    return abbrevs


def get_latency(model, device='cpu', inp=torch.randn(32, 3, 224, 224), iters=2):
    """Test latency of the argument model."""

    if not isinstance(device, torch.device):
        assert device in ['cpu', 'cuda:0']

    torch_device = torch.device(device)
    model = model.to(torch_device)
    dummy_input = inp.to(torch_device)
    latency = np.zeros((iters, 1))

    with torch.no_grad():
        if device == 'cpu':
            # Warm up
            for _ in range(10):
                _ = model(dummy_input)

            # Measure latency.
            for rep in range(iters):
                start = time.time()
                _ = model(dummy_input)
                elapsed = time.time() - start
                latency[rep] = elapsed

        else:
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

            # Warm up
            for _ in range(10):
                _ = model(dummy_input)

            # Measure latency.
            with torch.no_grad():
                for rep in range(iters):
                    starter.record()
                    _ = model(dummy_input)
                    ender.record()
                    torch.cuda.synchronize()
                    elapsed = starter.elapsed_time(ender)
                    latency[rep] = elapsed

    return np.mean(latency), np.std(latency)


class AverageMeter:
    def __init__(self):
        self.sum = 0
        self.pts = 0

    def add(self, val):
        self.sum += val
        self.pts += 1

    @property
    def value(self):
        return self.sum / self.pts


def write2csv(row: list, output_path: str, write_mode='a'):
    with open(output_path, write_mode) as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(row)
        f_object.close()


class BLEU:
    def __init__(self, output_path: str, *args, **kwargs):
        # self.bleu = evaluate.load("bleu")
        self.bleu = evaluate.load("sacrebleu")
        self.output_path = output_path
        write2csv(["predictions", "references", "BLEU"], self.output_path, write_mode='w')

    def __call__(self, predictions: list, references: list, *args, **kwargs):
        results = self.bleu.compute(predictions=predictions, references=references)
        results['bleu'] = results['score']
        write2csv([predictions, references, results['bleu']], self.output_path)

        return results


class LatencyReport:
    def __init__(self):
        self.starter = torch.cuda.Event(enable_timing=True)
        self.latencies = dict()
        self.stoppers = list()

    def report(self):
        # str = "Latency report:\n"
        strs = ["{} {:4d} ms".format(name, int(latency)) for name, latency in self.latencies.items()]
        strs = " | ".join(strs)
        return strs

    def start(self):
        self.starter.record()
        self.stoppers.append(torch.cuda.Event(enable_timing=True))

    def stop(self, name="Unk"):
        self.stoppers[-1].record()
        torch.cuda.synchronize()
        self.latencies[name] = self.starter.elapsed_time(self.stoppers[-1])
        self.stoppers.append(torch.cuda.Event(enable_timing=True))


class CudaMemoryTracker:
    def __init__(self):
        self.memory_allocated = {
            # "start": torch.cuda.memory_allocated(),
        }

        self.memory_reserved = {
            # "start": torch.cuda.memory_reserved(),
        }

    def track(self, name="Unk"):
        self.memory_allocated[name] = torch.cuda.memory_allocated()
        self.memory_reserved[name] = torch.cuda.memory_reserved()

    def report(self):
        strs = ["{} {:4d} MB".format(name, int(mem / (1024 * 1024))) for name, mem in self.memory_allocated.items()]
        strs = " | ".join(strs)
        return strs
