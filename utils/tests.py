def transform_dict(input_dict):
    output_dict = {}

    for key, value in input_dict.items():
        keys_split = key.split('-')
        temp_dict = output_dict

        for sub_key in keys_split[:-1]:
            temp_dict = temp_dict.setdefault(sub_key, {})

        temp_dict[keys_split[-1]] = value

    return output_dict


# Test the function
input_dict = {"train-lr": 0.001, "pmodel-params-rank": 4, "pmodel-params-name": "lora"}

test_cases_inp = [
    {"train-lr": 0.001, "pmodel-params-rank": 4, "pmodel-params-name": "lora"},
    {"train-lr": 0.001, "pmodel-params-rank": 4, "pmodel-params-name": "lora", "train-bs": 10},
    {"train-lr": 0.001, "pmodel-params-rank": 4, "pmodel-name": "gpt2", "pmodel-params-method": "lora"},
    {"train": {"lr": 0.001}, "pmodel": {"name": "gpt2", "params": {"method": "lora", "rank": 4}}},
    {"train": {"lr": 0.1}, "train-lr": 0.001, "pmodel-params-rank": 4, "pmodel-name": "gpt2", "pmodel-params-method": "lora"},
    ]

test_case_gt = [
    {"train": {"lr": 0.001}, "pmodel": {"params": {"rank": 4, "name": "lora"}}},
    {"train": {"lr": 0.001, "bs": 10}, "pmodel": {"params": {"rank": 4, "name": "lora"}}},
    {"train": {"lr": 0.001}, "pmodel": {"name": "gpt2", "params": {"method": "lora", "rank": 4}}},
    {"train": {"lr": 0.001}, "pmodel": {"name": "gpt2", "params": {"method": "lora", "rank": 4}}},
    {"train": {"lr": 0.001}, "pmodel": {"name": "gpt2", "params": {"method": "lora", "rank": 4}}},
]

for inp, gt in zip(test_cases_inp, test_case_gt):
    assert transform_dict(inp) == gt, f"Failed for {inp}"

    # Convert to yaml:
    import yaml
    print(yaml.dump(transform_dict(inp)))
    print('---')
print("All tests passed!")
# print(transform_dict(input_dict))
