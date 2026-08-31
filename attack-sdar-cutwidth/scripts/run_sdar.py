import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import time
import argparse
import sys
sys.path.append('../src')
import data
from sdar.sdar import SDARAttacker
from sdar.sdar_hetero import SDARHeteroAttacker
from util.util import plot_attack_results, load_config

parser = argparse.ArgumentParser(description='Run SDAR experiment with specified configurations.')
parser.add_argument("setting", type=str, choices=["vsl", "usl"])
parser.add_argument("model", type=str, choices=["resnet", "plainnet"])
parser.add_argument("dataset", type=str, choices=["cifar10", "cifar100", "tinyimagenet", "stl10"])
parser.add_argument("level", type=int, choices=[4,5,6,7,8,9])
parser.add_argument("run", type=int, help="Run number.")

parser.add_argument("--width", type=str, default="standard", choices=["standard", "wide", "narrow", "narrow8", "cut8", "cut16", "cut32", "cut64", "cut128"])
parser.add_argument("--attacker_width", type=str, default=None, choices=["standard", "wide", "narrow", "narrow8", "cut8", "cut16", "cut32", "cut64", "cut128"],
                    help="Width of the reconstruction attacker (decoder + sim-discriminator), decoupled "
                         "from the client model --width. Defaults to --width. Set 'standard' with a "
                         "narrow client cut (e.g. --width cut8) so the narrow cut faces a "
                         "full-strength attacker and its SSIM is not inflated by a handicapped decoder.")
parser.add_argument("--server_width", type=str, default=None, choices=["standard", "wide", "narrow", "narrow8", "cut8", "cut16", "cut32", "cut64", "cut128"],
                    help="Width of the server backbone g, decoupled from the client model --width. "
                         "Defaults to --width. Set 'standard' with a narrow client cut (--width cut8) so "
                         "the victim keeps full server capacity (via the 1x1 cut adapter) and actually "
                         "learns the task instead of collapsing to chance.")
parser.add_argument("--aux_data_frac", type=float, default=1.0)
parser.add_argument("--num_class_to_remove", type=int, default=0)
parser.add_argument("--diff_simulator", action="store_true")

parser.add_argument("--ablation", type=str, choices=["no_e_dis", "no_d_dis", "no_dis", "no_cond", "no_label_flip", "naive_sda"], default=None)

# number of heterogenous distributions of the client data
parser.add_argument("--num_hetero_client", type=int, default=1)

# the following arguments are used to specify the defense method
parser.add_argument("--l1", type=float, default=0.0)
parser.add_argument("--l2", type=float, default=0.0)
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--alpha", type=float, default=0.0)
parser.add_argument("--bottleneck_mode", type=str, default="none", choices=["none", "fixed_random", "topk_sparse", "learned_1x1", "block_random"])
parser.add_argument("--compression_ratio", type=float, default=1.0)
parser.add_argument("--bottleneck_k", type=int, default=0)
parser.add_argument("--block_size", type=int, default=2,
                    help="block_random tile size: 1=per-position (max utility/min privacy), "
                         "cut_hw=global dense==fixed_random (max privacy). Must divide the cut H,W.")
parser.add_argument("--wcc_lambda", type=float, default=0.0)
parser.add_argument("--center_loss_lambda", type=float, default=0.0,
                    help="Center loss (Wen et al. 2016) weight on the payload (0=off). Same "
                         "class-compaction objective as WCC but with persistent EMA centers. "
                         "Use with --bottleneck_mode fixed_random for the center-loss comparison.")
parser.add_argument("--center_alpha", type=float, default=0.5,
                    help="EMA update rate for center-loss class centers.")
parser.add_argument("--dp_clip_norm", type=float, default=0.0,
                    help="DP: per-sample L2 clip norm on the transmitted payload (0=off). "
                         "Applied inside the bottleneck before lift-back, so the attacker's "
                         "reconstruction and the eval SSIM both see the clip+noise. May stack "
                         "on --bottleneck_mode fixed_random.")
parser.add_argument("--dp_sigma", type=float, default=0.0,
                    help="DP: Gaussian noise multiplier; noise std = dp_sigma * dp_clip_norm "
                         "(requires --dp_clip_norm > 0).")
parser.add_argument("--num_iter_override", type=int, default=None)

parser.add_argument("--force_cut_adapter", action="store_true",
                    help="Keep the server-side 1x1 cut adapter even when the client cut already has "
                         "the channel count the server stage expects. Used by the cut-width (d) sweep "
                         "so the 4096-d 'cut64' arm has the same server graph as its siblings.")

parser.add_argument("--print_to_stdout", action="store_true")

args = parser.parse_args()

compression_defense = args.bottleneck_mode != "none" or args.wcc_lambda != 0.0 or args.center_loss_lambda != 0.0
dp_defense = args.dp_clip_norm > 0.0

if args.compression_ratio <= 0.0:
    raise ValueError("compression_ratio must be positive.")
if args.bottleneck_k < 0:
    raise ValueError("bottleneck_k must be non-negative.")
if args.wcc_lambda < 0.0:
    raise ValueError("wcc_lambda must be non-negative.")
if args.center_loss_lambda < 0.0:
    raise ValueError("center_loss_lambda must be non-negative.")
if args.num_iter_override is not None and args.num_iter_override <= 0:
    raise ValueError("num_iter_override must be positive.")

if sum([args.level > 7, args.l1 != 0.0, args.l2 != 0.0, args.dropout != 0.0, args.alpha != 0.0, compression_defense]) > 1:
        raise ValueError("At most one of l1, l2, dropout, alpha, bottleneck defense can be not default.")

defense = sum([args.level > 7, args.l1 != 0.0, args.l2 != 0.0, args.dropout != 0.0, args.alpha != 0.0, compression_defense]) > 0 or dp_defense

# at most one of width and aux_data_frac can be not default -- EXCEPT that the cut-width (d)
# sweep deliberately pairs a non-standard client width with the compression/DP defense, which is
# the whole point of that experiment (vary d at a fixed cut depth, then sweep CR and WCC on top).
_width_x_defense = args.width != "standard" and defense
if not _width_x_defense and sum([args.width != "standard", args.aux_data_frac != 1.0, args.diff_simulator, args.num_class_to_remove != 0, args.ablation != None, args.num_hetero_client > 1, defense]) > 1:
    raise ValueError("Only one of width, aux_data_frac, diff_simulator, num_class_to_remove, ablation, num_hetero_client, defense can be not default.")
if _width_x_defense and sum([args.aux_data_frac != 1.0, args.diff_simulator, args.num_class_to_remove != 0, args.ablation != None, args.num_hetero_client > 1]) > 0:
    raise ValueError("The width x defense combination does not compose with aux_data_frac, diff_simulator, num_class_to_remove, ablation or num_hetero_client.")

# get the directory of this file
curr_dir = os.path.join(os.path.dirname(__file__))

if _width_x_defense:
    # cut-width (d) sweep: keep the defense parameter tag AND the width tag, so every
    # (width, CR, wcc) cell gets its own file.
    rela_dir = f"cut_width_results/{args.bottleneck_mode}/{args.setting}/{args.dataset}"
    _wtag = f"cr{args.compression_ratio:g}" if args.bottleneck_k <= 0 else f"k{args.bottleneck_k}"
    if args.wcc_lambda != 0.0:
        _wtag += f"_wcc{args.wcc_lambda:g}"
    if args.center_loss_lambda != 0.0:
        _wtag += f"_center{args.center_loss_lambda:g}"
    if args.dp_clip_norm > 0.0:
        _wtag += f"_dps{args.dp_sigma:g}"
    file_name = f"{_wtag}_{args.width}_{args.model}_l{args.level}_run{args.run}"
elif args.width != "standard":
    rela_dir = f"model_width_results/{args.setting}/{args.dataset}"
    file_name = f"{args.model}_{args.width}_l{args.level}_run{args.run}"
elif args.aux_data_frac != 1.0:
    rela_dir = f"data_frac_results/{args.setting}/{args.dataset}"
    file_name = f"{args.model}_{args.aux_data_frac}_l{args.level}_run{args.run}"
elif args.diff_simulator:
    rela_dir = f"diff_simulator_results/{args.setting}/{args.dataset}"
    file_name = f"{args.model}_l{args.level}_run{args.run}"
elif args.num_class_to_remove != 0:
    rela_dir = f"remove_aux_class_results/{args.setting}/{args.dataset}"
    file_name = f"{args.model}_aux_wo_{args.num_class_to_remove}_l{args.level}_run{args.run}"
elif args.ablation != None:
    rela_dir = f"ablation_results/{args.ablation}/{args.setting}/{args.dataset}"
    file_name = f"{args.model}_l{args.level}_run{args.run}"
elif args.num_hetero_client > 1:
    rela_dir = f"hetero_clients_results/{args.setting}/{args.dataset}"
    file_name = f"{args.num_hetero_client}_{args.model}_l{args.level}_run{args.run}"
elif defense:
    if args.l1 != 0.0:
        rela_dir = f"defense_results/l1/{args.setting}/{args.dataset}"
        file_name = f"{args.l1}_{args.model}_l{args.level}_run{args.run}"
    elif args.l2 != 0.0:
        rela_dir = f"defense_results/l2/{args.setting}/{args.dataset}"
        file_name = f"{args.l2}_{args.model}_l{args.level}_run{args.run}"
    elif args.dropout != 0.0:
        rela_dir = f"defense_results/dropout/{args.setting}/{args.dataset}"
        file_name = f"{args.dropout}_{args.model}_l{args.level}_run{args.run}"
    elif args.alpha != 0.0:
        rela_dir = f"defense_results/decorrelation/{args.setting}/{args.dataset}"
        file_name = f"{args.alpha}_{args.model}_l{args.level}_run{args.run}"
    elif compression_defense:
        bottleneck_name = args.bottleneck_mode
        if args.bottleneck_mode == "none":
            # pure loss-based defense (no projection): name by whichever loss is active
            if args.center_loss_lambda != 0.0:
                bottleneck_name = "center"
                param_tag = f"center{args.center_loss_lambda:g}"
            else:
                bottleneck_name = "wcc"
                param_tag = f"wcc{args.wcc_lambda:g}"
        else:
            param_tag = f"k{args.bottleneck_k}" if args.bottleneck_k > 0 else f"cr{args.compression_ratio:g}"
        if args.bottleneck_mode == "block_random":
            param_tag = f"{param_tag}_b{args.block_size}"
        if args.wcc_lambda != 0.0 and args.bottleneck_mode != "none":
            param_tag = f"{param_tag}_wcc{args.wcc_lambda:g}"
        if args.center_loss_lambda != 0.0 and args.bottleneck_mode != "none":
            param_tag = f"{param_tag}_center{args.center_loss_lambda:g}"
        if dp_defense:
            param_tag = f"{param_tag}_dps{args.dp_sigma:g}"
        rela_dir = f"defense_results/{bottleneck_name}/{args.setting}/{args.dataset}"
        file_name = f"{param_tag}_{args.model}_l{args.level}_run{args.run}"
    elif dp_defense:
        # DP on the raw (undefended) model: no projection bottleneck, DP is the only defense.
        rela_dir = f"defense_results/dp/{args.setting}/{args.dataset}"
        file_name = f"raw_dps{args.dp_sigma:g}_{args.model}_l{args.level}_run{args.run}"
    elif args.level > 7:
        rela_dir = f"defense_results/deeper_f/{args.setting}/{args.dataset}"
        file_name = f"{args.model}_l{args.level}_run{args.run}"
else:
    rela_dir = f"sdar_results/{args.setting}/{args.dataset}"
    file_name = f"{args.model}_l{args.level}_run{args.run}"

base_dir = os.path.join(curr_dir, rela_dir)
fig_dir = os.path.join(base_dir, "fig")
output_dir = os.path.join(base_dir, "output")

if not os.path.exists(base_dir):
    os.makedirs(base_dir)
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if not args.print_to_stdout:
    sys.stdout = open(os.path.join(base_dir, file_name + ".txt"),'wt')

u_shape = (args.setting == "usl")
conditional = not u_shape
num_class = 100 if args.dataset == "cifar100" else 200 if args.dataset == "tinyimagenet" else 10
batch_size = 32 if args.dataset == "stl10" else 128 # since stl10 contains only 13000 images
num_iter = 40000 if args.level > 7 else 20000 if args.dataset in ["cifar10", "cifar100", "tinyimagenet", "stl10"] else 10000
if args.num_iter_override is not None:
    num_iter = args.num_iter_override
config = load_config("sdar", args.model, args.dataset, u_shape)

if args.ablation == "no_e_dis":
    config["lambda1"] = 0.0
elif args.ablation == "no_d_dis":
    config["lambda2"] = 0.0
elif args.ablation == "no_dis":
    config["lambda1"] = 0.0
    config["lambda2"] = 0.0
elif args.ablation == "no_cond":
    conditional = False
elif args.ablation == "no_label_flip":
    config["flip_rate"] = 0.0
elif args.ablation == "naive_sda":
    config["lambda1"] = 0.0
    config["lambda2"] = 0.0
    config["flip_rate"] = 0.0
    conditional = False

if args.num_hetero_client == 1:
    client_ds, server_ds = data.load_dataset(args.dataset, frac=args.aux_data_frac, num_class_to_remove=args.num_class_to_remove)
else:
    client_ds_list, server_ds_list = data.load_dataset_with_hetero_clients(args.dataset, args.num_hetero_client)

print(f"Start {args.setting}/{args.model} experiments on {args.dataset} at l{args.level} with {config}")
if compression_defense:
    block_str = f", block_size={args.block_size}" if args.bottleneck_mode == "block_random" else ""
    print(f"Compression defense: mode={args.bottleneck_mode}, compression_ratio={args.compression_ratio}, bottleneck_k={args.bottleneck_k}{block_str}, wcc_lambda={args.wcc_lambda}, center_loss_lambda={args.center_loss_lambda}")
if dp_defense:
    print(f"DP defense: dp_clip_norm={args.dp_clip_norm}, dp_sigma={args.dp_sigma} (noise std = {args.dp_sigma * args.dp_clip_norm:g})")

if args.num_hetero_client == 1:
    sdar_attacker = SDARAttacker(client_ds, server_ds, num_class=num_class, batch_size=batch_size)
else:
    sdar_attacker = SDARHeteroAttacker(client_ds_list, server_ds_list, num_class=num_class, batch_size=batch_size)

start_time = time.time()
history = sdar_attacker.run(level=args.level, num_iter=num_iter, config=config, u_shape=u_shape, conditional=conditional, model_type=args.model, width=args.width, attacker_width=args.attacker_width, server_width=args.server_width, diff_simulator=args.diff_simulator, l1=args.l1, l2=args.l2, dropout=args.dropout, alpha=args.alpha, bottleneck_mode=args.bottleneck_mode, compression_ratio=args.compression_ratio, bottleneck_k=args.bottleneck_k, block_size=args.block_size, wcc_lambda=args.wcc_lambda, center_lambda=args.center_loss_lambda, center_alpha=args.center_alpha, dp_clip_norm=args.dp_clip_norm, dp_sigma=args.dp_sigma, force_cut_adapter=args.force_cut_adapter, verbose_freq=100)
end_time = time.time()

print(f"Experiment took {end_time - start_time} seconds, on average {(end_time - start_time)/num_iter} seconds/iteration.")
print("Experiment done. Saving history...")

if args.num_hetero_client == 1:
    _ = sdar_attacker.evaluate(verbose=True)
else:
    eval_ds, _ = data.load_dataset(args.dataset, frac=args.aux_data_frac, num_class_to_remove=args.num_class_to_remove)
    _ = sdar_attacker.evaluate(eval_ds, verbose=True)

eval_batch = data.load_single_batch(args.dataset, batch_size=batch_size)
x,y = [(x,y) for (x,y) in iter(eval_batch)][0]

x_recon, mse = sdar_attacker.attack(x, y)
t = 10
plot_attack_results(x_recon[t:t+20], os.path.join(fig_dir, file_name + ".png"))
example_mse = np.mean((x[t:t+20] - x_recon[t:t+20])**2)
print(f"Attack MSE on batch: {mse}, on these examples: {example_mse}")

np.save(os.path.join(output_dir, file_name + ".npy"), history)
print(f"History saved to {os.path.join(rela_dir, file_name + '.npy')}")
