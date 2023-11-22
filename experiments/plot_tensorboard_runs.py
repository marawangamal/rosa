import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
import os

rte = {
    "root": "/home/mila/m/marawan.gamal/scratch/rosa/runs-2023-09-28",
    "exps": {
        "FT": "glue/rte/e10_l2e-05_b32_f1.0_s512_nadamw_be0.9_0.98_ep1e-08_w0.1_nalinear_wa0.06_namroberta-base_namenone_fa1_facepoch_iTrue_r1_leepoch_factrandom_factoequal_uFalse_t0",
        "ROSA": "glue/rte/e10_l0.002_b16_f1.0_s512_nadamw_be0.9_0.98_ep1e-08_w0.1_nalinear_wa0.06_namroberta-base_namerosa_fa1_facepoch_iTrue_r2_leepoch_factrandom_factoequal_uFalse_biFalse_t0",
        "LoRA": "glue/rte/e10_l0.0002_b16_f1.0_s512_nadamw_be0.9_0.98_ep1e-08_w0.1_nalinear_wa0.06_namroberta-base_namelora_fa1_facepoch_iTrue_r2_leepoch_factrandom_factoequal_uFalse_t0"
    },
    "title": "RTE Accuracy",
    "tag_name": "valid/accuracy",
    "xlabel": "Steps",
    "ylabel": "Accuracy",
}

cola = {
    "root": "/home/mila/m/marawan.gamal/scratch/rosa/runs/glue/cola",
    "exps": {
        "FT": "e10_l2e-05_b16_f1.0_s512_nadamw_be0.9_0.98_ep1e-06_w0.1_nalinear_wa0.06_namroberta-base_namenone_fa1_facepoch_iTrue_r1_leepoch_aab_sarandom_factequal_uFalse_biFalse_fasFalse_t0",
        "ROSA": "e10_l0.002_b16_f1.0_s512_nadamw_be0.9_0.98_ep1e-06_w0.1_nalinear_wa0.06_namroberta-base_namerosa_fa2_facepoch_iTrue_r2_leepoch_aab_sarandom_factequal_uFalse_biFalse_fasFalse_t0",
        "LoRA": "e10_l0.0002_b16_f1.0_s512_nadamw_be0.9_0.98_ep1e-06_w0.1_nalinear_wa0.06_namroberta-base_namelora_fa1_facepoch_iTrue_r2_leepoch_aab_sarandom_factequal_uFalse_biFalse_fasFalse_t0"
    },
    "title": "CoLA MCC",
    "tag_name": "valid/matthews_correlation",
    "xlabel": "Steps",
    "ylabel": "MCC",
}

cola_loss = {
    "root": "/home/mila/m/marawan.gamal/scratch/rosa/runs",
    "exps": {
        "FT": "glue/cola/e10_l2e-05_b16_f1.0_s512_nadamw_be0.9_0.98_ep1e-06_w0.1_nalinear_wa0.06_namroberta-base_namenone_fa1_facepoch_iTrue_r1_leepoch_aab_sarandom_factequal_uFalse_biFalse_fasFalse_t0",
        # "ROSA F2 R8": "glue/cola/e10_l0.002_b16_f1.0_s512_nadamw_be0.9_0.98_ep1e-06_w0.1_nalinear_wa0.06_namroberta-base_namerosa_fa2_facepoch_iTrue_r8_leepoch_aab_sarandom_factequal_uFalse_biFalse_fasFalse_t0",
        "ROSA (R=8) ": "glue/cola/e10_l0.002_b16_f1.0_s512_nadamw_be0.9_0.98_ep1e-06_w0.1_nalinear_wa0.06_namroberta-base_namerosa_fa2_facepoch_iTrue_r8_leepoch_aab_sarandom_factequal_uFalse_biFalse_fasFalse_t0",
        "ROSA (R=2) ": "glue/cola/e10_l0.002_b16_f1.0_s512_nadamw_be0.9_0.98_ep1e-06_w0.1_nalinear_wa0.06_namroberta-base_namerosa_fa2_facepoch_iTrue_r2_leepoch_aab_sarandom_factequal_uFalse_biFalse_fasFalse_t0",
        "LoRA (R=2)": "glue/cola/e10_l0.0002_b16_f1.0_s512_nadamw_be0.9_0.98_ep1e-06_w0.1_nalinear_wa0.06_namroberta-base_namelora_fa1_facepoch_iTrue_r2_leepoch_aab_sarandom_factequal_uFalse_biFalse_fasFalse_t0",
        "LoRA (R=8)": "glue/cola/e10_l2e-05_b16_f1.0_s512_nadamw_be0.9_0.98_ep1e-06_w0.1_nalinear_wa0.06_namroberta-base_namelora_fa1_facepoch_iTrue_r8_leepoch_aab_sarandom_factequal_uFalse_biFalse_fasFalse_t0"
    },
    # "title": "Train Loss",
    "tag_name": "train/loss",
    "xlabel": "Steps",
    "ylabel": "Loss",
}

cola_mcc = {
    "root": "/home/mila/m/marawan.gamal/scratch/rosa/runs",
    "exps": {
        "FT": "glue/cola/e10_l2e-05_b16_f1.0_s512_nadamw_be0.9_0.98_ep1e-06_w0.1_nalinear_wa0.06_namroberta-base_namenone_fa1_facepoch_iTrue_r1_leepoch_aab_sarandom_factequal_uFalse_biFalse_fasFalse_t0",
        # "ROSA F2 R8": "glue/cola/e10_l0.002_b16_f1.0_s512_nadamw_be0.9_0.98_ep1e-06_w0.1_nalinear_wa0.06_namroberta-base_namerosa_fa2_facepoch_iTrue_r8_leepoch_aab_sarandom_factequal_uFalse_biFalse_fasFalse_t0",
        "ROSA (R=8)": "glue/cola/e10_l0.002_b16_f1.0_s512_nadamw_be0.9_0.98_ep1e-06_w0.1_nalinear_wa0.06_namroberta-base_namerosa_fa1_facepoch_iTrue_r8_leepoch_aab_sarandom_factequal_uFalse_biFalse_fasFalse_t0",
        "ROSA (R=2)": "glue/cola/e10_l0.002_b16_f1.0_s512_nadamw_be0.9_0.98_ep1e-06_w0.1_nalinear_wa0.06_namroberta-base_namerosa_fa1_facepoch_iTrue_r2_leepoch_aab_sarandom_factequal_uFalse_biFalse_fasFalse_t0",
        "LoRA (R=2)": "glue/cola/e10_l0.0002_b16_f1.0_s512_nadamw_be0.9_0.98_ep1e-06_w0.1_nalinear_wa0.06_namroberta-base_namelora_fa1_facepoch_iTrue_r2_leepoch_aab_sarandom_factequal_uFalse_biFalse_fasFalse_t0",
        "LoRA (R=8)": "glue/cola/e10_l2e-05_b16_f1.0_s512_nadamw_be0.9_0.98_ep1e-06_w0.1_nalinear_wa0.06_namroberta-base_namelora_fa1_facepoch_iTrue_r8_leepoch_aab_sarandom_factequal_uFalse_biFalse_fasFalse_t0"
    },
    # "title": "MCC",
    "tag_name": "valid/matthews_correlation",
    "xlabel": "Steps",
    "ylabel": "MCC",
}

def extract_scalar_data_from_event_file(event_file_path, tag_name):
    scalar_data = []
    for e in tf.compat.v1.train.summary_iterator(event_file_path):
        for v in e.summary.value:
            if v.tag == tag_name:
                scalar_data.append((e.step, v.simple_value))
    return np.array(scalar_data)


def exp_name_to_color(exp_name):
    if "FT" in exp_name:
        return "black"
    if "ROSA" in exp_name:
        return "red"
    if "LoRA" in exp_name:
        return "blue"
    return np.random.rand(3,)


def color_and_marker_func(name, peft_rank_max=8, peft_rank_step=2):
    rank_colors = {
        max(r, 1): plt.cm.get_cmap('tab10')(i) for i, r in
        enumerate(range(0, peft_rank_max + 1, peft_rank_step))
    }
    # import pdb; pdb.set_trace()
    rank = int(name.split('=')[-1].strip()[:-1]) if "FT" not in name else 0
    color = rank_colors[rank] if "FT" not in name else "black"
    marker = {"LoRA": "--", "ROSA": "-", "FT": "-"}[name.split(" ")[0]]
    return color, marker


def main():

    # Config
    exp = cola_mcc  # Choose experiment [cola, rte]
    save_path = "cola_mcc.pdf"
    print(f"Results will be saved to {save_path}")

    # Plot
    plt.figure(figsize=(10, 6))
    for label, exp_name in exp['exps'].items():
        log_dir = os.path.join(exp['root'], exp_name)
        event_file = next(filter(lambda f: f.startswith("events.out.tfevents"), os.listdir(log_dir)))
        event_file_path = os.path.join(log_dir, event_file)
        data = extract_scalar_data_from_event_file(event_file_path, exp['tag_name'])
        color, marker = color_and_marker_func(label)
        plt.plot(data[:, 0], data[:, 1], label=label, color=color, linestyle=marker)

    plt.xlabel(exp['xlabel'])
    plt.ylabel(exp['ylabel'])
    if 'title' in exp:
        plt.title(exp['title'])
    plt.legend()
    plt.tight_layout()  # Adjust the layout
    plt.savefig(save_path, bbox_inches='tight')  # Save the figure

if __name__ == "__main__":
    main()
