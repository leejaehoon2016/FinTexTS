import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import yaml
from forecasting_task.models.DLinear import Model as DLinear
from forecasting_task.models.PatchTST import Model as PatchTST
from forecasting_task.models.Informer import Model as Informer
from forecasting_task.models.Autoformer import Model as Autoformer
from forecasting_task.models.iTransformer import Model as iTransformer

from forecasting_task.models.Reformer import Model as Reformer
from forecasting_task.models.Crossformer import Model as Crossformer
from forecasting_task.models.Transformer import Model as Transformer
from forecasting_task.models.FiLM import Model as FiLM
from forecasting_task.models.Nonstationary_Transformer import Model as Nonstationary_Transformer
from forecasting_task.models.TSMixer import Model as TSMixer
from forecasting_task.models.TiDE import Model as TiDE

from forecasting_task.data_provider.dataset import get_dataset_dataloader

def _parse_used_col_prefixes(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [v.strip() for v in value.split(",") if v.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="forecasting_task runner")
    parser.add_argument("--task_name", default="short_term_forecast")
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--pred_len", type=int, default=3)
    parser.add_argument("--label_len", type=int, default=16)
    
    
    parser.add_argument(
        "--used_col_prefixes",
        type=_parse_used_col_prefixes,
        default=[
            "macro",
            "sector",
            "targetCompany",
            "relatedCompany",
            "filing",
        ],
        help="Comma-separated list, e.g. macro,sector,targetCompany,relatedCompany",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--root_path", type=str, default="data/")
    parser.add_argument("--data_path", default="fintexts/NVDA.parquet")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model_type", type=str, default="PatchTST")
    parser.add_argument("--moving_avg", type=int, default=25)
    parser.add_argument("--patch_len", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--dropout", type=int, default=0.1)
    parser.add_argument("--e_layers", type=int, default=2)
    parser.add_argument("--d_ff", type=int, default=64)
    
    parser.add_argument("--text_hidden_size", type=int, default=64)
    parser.add_argument("--text_num_layers", type=int, default=1)
    parser.add_argument("--text_weight", type=float, default=0.1)
    
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument("--output_attention", type=bool, default=False)
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--freq", type=str, default="d")
    parser.add_argument("--distil", type=bool, default=True)
    
    parser.add_argument("--num_epoch", type=int, default=10)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--logdir",
        type=str,
        default="logs/tmp_test",
    )
    
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    
    args = parser.parse_args()
    args.enc_in = 4
    args.dec_in = 4
    args.c_out = 4
    args.d_layers = args.e_layers
    args.p_hidden_dims = [128,128]
    args.p_hidden_layers = 2
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "args.yaml"), "w") as f:
        yaml.safe_dump(vars(args), f, sort_keys=False)
    
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class TextModel(nn.Module):
    def __init__(self, hidden_size, pred_len, num_layers):
        super(TextModel, self).__init__()
        self.relu = nn.ReLU()
        self.num_layers = num_layers

        if num_layers <= 1:
            self.layers = nn.ModuleList([nn.Linear(384, pred_len)])
        else:
            layers = [nn.Linear(384, hidden_size)]
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Linear(hidden_size, pred_len))
            self.layers = nn.ModuleList(layers)

    def forward(self, price_x, x):
        ori_x = x.clone()
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.relu(x)
        out = x.unsqueeze(-1).repeat(1, 1, price_x.shape[1]) + price_x.unsqueeze(1).repeat(1, x.shape[1], 1)
        valid_mask = (ori_x.abs().sum(1) >= 1e-6).view(-1, 1, 1).float()
        return out * valid_mask

def _prepare_batch(batch, device, pred_len, label_len):
    seq_price_x, seq_price_y, seq_text_x, seq_x_mark, seq_y_mark, _ = batch
    seq_price_x = seq_price_x.float().to(device)
    seq_price_y = seq_price_y.float().to(device)
    seq_text_x = seq_text_x.float().to(device)
    seq_x_mark = seq_x_mark.float().to(device)
    seq_y_mark = seq_y_mark.float().to(device)

    dec_inp = torch.zeros_like(seq_price_y[:, -pred_len:, :]).float().to(device)
    dec_inp = torch.cat([seq_price_y[:, :label_len, :], dec_inp], dim=1).float().to(device)
    ground_truth = seq_price_y[:, -pred_len:, :]
    return seq_price_x, seq_text_x, seq_x_mark, seq_y_mark, dec_inp, ground_truth


def _inverse_transform_batch(dataset, tensor):
    array = tensor.detach().cpu().numpy()
    batch, length, channels = array.shape
    flat = array.reshape(-1, channels)
    inv = dataset.inverse_transform(flat)
    return inv.reshape(batch, length, channels)


def _run_epoch(
    dataloader, dataset, model, text_model, args, device, optimizer=None, collect_io=False
):
    is_train = optimizer is not None
    if is_train:
        model.train()
        text_model.train()
    else:
        model.eval()
        text_model.eval()

    mse_sum = 0.0
    mae_sum = 0.0
    count = 0
    input_chunks = []
    output_chunks = []
    gt_chunks = []

    for batch in dataloader:
        seq_price_x, seq_text_x, seq_x_mark, seq_y_mark, dec_inp, ground_truth = _prepare_batch(
            batch, device, args.pred_len, args.label_len
        )

        if is_train:
            optimizer.zero_grad()

        output = model(seq_price_x, seq_x_mark, dec_inp, seq_y_mark)
        if args.text_weight != 0:
            text_output = text_model(seq_price_x[:, -1, :], seq_text_x[:, -1, :])
            output = (1 - args.text_weight) * output + args.text_weight * text_output

        mse_loss = torch.mean((output - ground_truth) ** 2)
        mae_loss = torch.mean(torch.abs(output - ground_truth))

        if is_train:
            mse_loss.backward()
            optimizer.step()

        if collect_io:
            input_chunks.append(seq_price_x.detach().cpu())
            output_chunks.append(output.detach().cpu())
            gt_chunks.append(ground_truth.detach().cpu())

        batch_size = output.shape[0]
        mse_sum += mse_loss.item() * batch_size
        mae_sum += mae_loss.item() * batch_size
        count += batch_size

    mse = mse_sum / max(count, 1)
    mae = mae_sum / max(count, 1)
    if collect_io:
        inputs = torch.cat(input_chunks, dim=0) if input_chunks else None
        outputs = torch.cat(output_chunks, dim=0) if output_chunks else None
        gts = torch.cat(gt_chunks, dim=0) if gt_chunks else None
        return mse, mae, inputs, outputs, gts
    return mse, mae


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, train_dataloader = get_dataset_dataloader(
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=args.pred_len,
        flag="train",
        used_col_prefixes=args.used_col_prefixes,
        data_path=args.data_path,
        root_path=args.root_path,
        batch_size=args.batch_size,
    )
    val_dataset, val_dataloader = get_dataset_dataloader(
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=args.pred_len,
        flag="val",
        used_col_prefixes=args.used_col_prefixes,
        data_path=args.data_path,
        root_path=args.root_path,
        batch_size=args.batch_size,
    )
    test_dataset, test_dataloader = get_dataset_dataloader(
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=args.pred_len,
        flag="test",
        used_col_prefixes=args.used_col_prefixes,
        data_path=args.data_path,
        root_path=args.root_path,
        batch_size=args.batch_size,
    )
    if args.model_type.lower() == "dlinear":
        model = DLinear(args)
    elif args.model_type.lower() == "patchtst":
        model = PatchTST(args)
    elif args.model_type.lower() == "informer":
        model = Informer(args)
    elif args.model_type.lower() == "itransformer":
        model = iTransformer(args)
    elif args.model_type.lower() == "autoformer":
        model = Autoformer(args)
    elif args.model_type.lower() == "reformer":
        model = Reformer(args)
    elif args.model_type.lower() == "pyraformer":
        model = Pyraformer(args)
    elif args.model_type.lower() == "crossformer":
        model = Crossformer(args)
    elif args.model_type.lower() == "transformer":
        model = Transformer(args)
    elif args.model_type.lower() == "film":
        model = FiLM(args)
    elif args.model_type.lower() == "nonstationary_transformer":
        model = Nonstationary_Transformer(args)
    elif args.model_type.lower() == "fedformer":
        model = FEDformer(args)
    elif args.model_type.lower() == "timemixer":
        model = TimeMixer(args)
    elif args.model_type.lower() == "tsmixer":
        model = TSMixer(args)
    elif args.model_type.lower() == "tide":
        model = TiDE(args)
    elif args.model_type.lower() == "micn":
        model = MICN(args)
    else:
        raise ValueError(f"Invalid model type: {args.model_type}")
    model = model.to(device)
    
    text_model = TextModel(
        hidden_size=args.text_hidden_size,
        pred_len=args.pred_len,
        num_layers=args.text_num_layers,
    ).to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(text_model.parameters()), lr=args.lr
    )

    best_val = float("inf")
    best_test_mse = float("inf")
    best_test_mae = float("inf")
    best_test_inputs = None
    best_test_outputs = None
    patience_counter = 0

    for epoch in range(1, args.num_epoch + 1):
        train_mse, _ = _run_epoch(
            train_dataloader,
            train_dataset,
            model,
            text_model,
            args,
            device,
            optimizer=optimizer,
        )
        with torch.no_grad():
            val_mse, _ = _run_epoch(val_dataloader, val_dataset, model, text_model, args, device)
            test_mse, test_mae, test_inputs, test_outputs, test_gts = _run_epoch(
                test_dataloader,
                test_dataset,
                model,
                text_model,
                args,
                device,
                collect_io=True,
            )

        print(
            f"[Epoch {epoch}] "
            f"train_mse={train_mse:.6f} "
            f"val_mse={val_mse:.6f} "
            f"test_mse={test_mse:.6f} "
            f"test_mae={test_mae:.6f}"
        )

        if val_mse < best_val:
            best_val = val_mse
            best_test_mse = test_mse
            best_test_mae = test_mae
            patience_counter = 0
            best_test_inputs = test_inputs
            best_test_outputs = test_outputs
            best_test_gts = test_gts
            best_test_inputs_inv = (
                _inverse_transform_batch(test_dataset, test_inputs)
                if test_inputs is not None
                else None
            )
            best_test_outputs_inv = (
                _inverse_transform_batch(test_dataset, test_outputs)
                if test_outputs is not None
                else None
            )
            best_test_gts_inv = (
                _inverse_transform_batch(test_dataset, test_gts)
                if test_gts is not None
                else None
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (best val_mse={best_val:.6f})")
                break

    print(
        f"[Best] val_mse={best_val:.6f} test_mse={best_test_mse:.6f} test_mae={best_test_mae:.6f}"
    )
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "result.txt"), "w") as f:
        f.write(
            f"best_val_mse={best_val:.6f}\n"
            f"best_test_mse={best_test_mse:.6f}\n"
            f"best_test_mae={best_test_mae:.6f}\n"
        )
    if best_test_inputs is not None and best_test_outputs is not None:
        torch.save(
            {
                "inputs": best_test_inputs,
                "outputs": best_test_outputs,
                "ground_truths": best_test_gts,
                "inputs_inv": best_test_inputs_inv,
                "outputs_inv": best_test_outputs_inv,
                "ground_truths_inv": best_test_gts_inv,
            },
            os.path.join(args.logdir, "test_io.pt"),
        )
    