import os
import sys
import time
import torch
import wandb
import logging
import numpy as np
from pathlib import Path
from copy import deepcopy

from experiment_launcher import single_experiment, run_experiment

from ldm.systems.handler import get_dynamics_model
from ldm.systems.state_weights import get_state_weights
from ldm.utils.df_save import create_df
from ldm.utils.read_datasets import read_datasets
from ldm.utils.jacobian_reg import hutchinson_jacobian_frobenius_sq
from ldm.utils.rollout_model import RolloutModel
from ldm.utils.rollout_model_with_history import RolloutModelWithHistory

#os.environ["WANDB_API_KEY"] = "a9819ac569197dbd24b580d854c3041ad75efafd"

log = logging.getLogger(__name__)
DEBUG = len([k for k in os.environ.keys() if "DEBUG" in k.upper()]) > 0
#DEBUG = True
print(f"Debug mode: {DEBUG}")

if DEBUG:
    torch.autograd.set_detect_anomaly(True)
    def custom_repr(self):
        return f"{{Tensor:{tuple(self.shape)}}} {original_repr(self)}"
    original_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = custom_repr

max_torch_num_threads = 8
torch.backends.mkldnn.enabled = True
torch.set_num_threads(max_torch_num_threads)
torch.set_num_interop_threads(max_torch_num_threads)
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('highest')



@single_experiment
def experiment(
    dataset: str = None,
    #system: str = "vector_field",
    #system: str = "f1tenth",
    #system: str = "acrobot",
    system: str = "vw_golf",
    dynamics_model_type: str = "pacejka_single_track_fd",
    state_extender: str = None,
    backward: bool = False,
    n_epochs: int = 1000,
    batch_size: int = 256,
    observation_window_len: int = 100,
    val_rollout_len: int = 100,
    train_rollout_len: int = 100,
    lr: float = 5e-4,
    #grad_clip: float = 0.0005,
    grad_clip: float = 0.01,
    train_chunk_size: int = 2,
    val_chunk_size: int = 2,
    Tp: float = 0.01, # TODO this need to be consistent with dataset
    stride: int = 3,
    tire_forces_reg: float = 0.0,
    input_grad_reg: float = 0.,
    integration_method: str = "euler",
    compile: bool = False,
    #compile: bool = True,
    randomize_init: float = 0.0,
    n: int = 10,
    n_up: int = 6,
    nn: int = 32,
    history_len: int = 10,  # history window for mlp_history model
    wandb_artifact_save_interval: int = 20,
    results_dir: str = "./results",
    seed: int = 444,
):
    config = {**locals()}
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"

    log.info(f"Process ID {os.getpid()}  seed {seed}")
    log.info(f"Output directory  : {results_dir}")

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    group_name = f"{system}_{dynamics_model_type}_{state_extender}_adam_gc{grad_clip}_lr{lr}_bs{batch_size}_cs{train_chunk_size}_{val_chunk_size}_rol{val_rollout_len}_{train_rollout_len}_hist{observation_window_len}_stride{stride}_{integration_method}"
    if "kicajka" in dynamics_model_type: 
        group_name += f"_n{n}_nup{n_up}"
    project_name = f"ldm_{system}"

    if dataset is not None:
        project_name = f"ldm_{dataset}_{system}"

    run = wandb.init(project=project_name,
                     group=group_name,
                     name=group_name + f"_seed{seed}",
                     config=config,
                     mode="disabled" if DEBUG else "online")

    train_dataset, val_dataset, _ = read_datasets(system, batch_size, observation_window_len,
                                                  val_rollout_len, stride=stride, device=device, dataset=dataset)

    # print batch per epoch
    log.info(f"Train dataset size in batch{len(train_dataset)}")
    log.info(f"Val dataset size in batch{len(val_dataset)}")

    # Model
    dynamics_model, state_extender = get_dynamics_model(system=system,
                                                        model_type=dynamics_model_type,
                                                        state_extender_type=state_extender,
                                                        n=n, n_up=n_up, nn=nn, randomize_init=randomize_init)
    #a = torch.load(os.path.join(os.path.dirname(__file__), f"./results/444/models/stellar-cosmos-50/dyn_model.pt"))
    #dynamics_model.load_state_dict(a)
    #a = torch.load(os.path.join(os.path.dirname(__file__), f"./results/444/models/f1tenth_mlp_gc0.0001_lr0.0005_bs256_rollout100_seed444/best_dyn_model_ep61.pt"))
    #a = torch.load(os.path.join(os.path.dirname(__file__), f"./results/444/models/f1tenth_mlp_gc1.0_lr0.0005_bs256_rollout100_seed444/best_dyn_model_ep61.pt"))
    #dynamics_model.load_state_dict(a)
    dynamics_model = dynamics_model.to(device)
    log.info(dynamics_model)
    #parameters = dynamics_model.get_parameters_vector().detach().numpy()
    if "history" in dynamics_model_type:
        model_rollout = RolloutModelWithHistory(dyn_model=dynamics_model,
                                               integration_method=integration_method,
                                               Tp=Tp,
                                               compile=compile,
                                               history_len=history_len,)
    else:
        model_rollout = RolloutModel(dyn_model=dynamics_model,
                                     integration_method=integration_method,
                                     Tp=Tp,
                                     compile=compile,
                                     state_extender=state_extender,)
    model_rollout = model_rollout.to(device)

    if backward:
        model_reverse_rollout = RolloutModel(dyn_model=dynamics_model,
                                            integration_method=integration_method,
                                            Tp=-Tp,
                                            compile=compile,
                                            state_extender=state_extender,)
        model_reverse_rollout = model_reverse_rollout.to(device)

    if not DEBUG:
        wandb.watch(dynamics_model, log_freq=1000)

    

    # Collect parameters from both dynamics_model and state_extender
    all_params = list(dynamics_model.parameters())
    if state_extender is not None:
        all_params.extend(list(state_extender.parameters()))
    
    decay_params = [p for p in all_params if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for p in all_params if p.requires_grad and p.dim() < 2]
    optim_groups = [
        #{'params': decay_params, 'weight_decay': 0.01},
        {'params': decay_params, 'weight_decay': 0.0},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    # Optimization
    #optimizer = torch.optim.AdamW(optim_groups, lr=lr)
    optimizer = torch.optim.Adam(all_params, lr=lr)

    log.info(optimizer)

    loss_fn = torch.nn.MSELoss(reduction="none")

    best_val_loss = float("inf")
    best_dmodel = None
    best_model_df = None

    state_weights = get_state_weights(system).unsqueeze(0).unsqueeze(0).to(device)


    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}/{n_epochs} ")

        epoch_start_time = time.time()

        grad_norm_sum = 0
        train_loss_list = []
        input_grad_norm_list = []

        dynamics_model.train()
        t0 = time.perf_counter()
        # Training loop
        for sample_i, data in enumerate(train_dataset):
            #times = []
            #times.append(time.perf_counter())
            print(f"Train batch {sample_i+1}/{len(train_dataset)}", end='\r')
            H, U, X0, X, *_ = data

            optimizer.zero_grad()

            X_pred_long, tire_forces = model_rollout(H, X0, U, train_rollout_len, chunk_size=train_chunk_size)
            n_states = state_weights.shape[-1]
            loss_long_truncated = loss_fn(X_pred_long[:, :train_rollout_len, :n_states],
                                          X[:, :train_rollout_len, :n_states]) * state_weights

            loss_long = torch.sum(loss_long_truncated) / loss_long_truncated.numel()
            loss = loss_long

            #import numpy as np
            #t = np.linspace(Tp, Tp * train_rollout_len, train_rollout_len)
            #import matplotlib.pyplot as plt
            #for i in range(3):
            #  for j in range(3):
            #      plt.subplot(3, 3, i*3+j+1)
            #      plt.plot(X[i, :, j].detach().numpy(), label="true")
            #      plt.plot(X_sim[i, :, j].detach().numpy(), label="pred")
            #      plt.plot(X_sim_reverse[i, :, j].detach().numpy()[::-1], label="backward pred")
            #      plt.legend()
            #plt.show()


            if tire_forces_reg:
                #tire_forces_loss = tire_forces_reg * torch.sum(tire_forces ** 2) / tire_forces.numel()
                tire_forces_loss = tire_forces_reg * torch.mean((tire_forces[..., -2] - tire_forces[..., -1]) ** 2)
                loss = loss + tire_forces_loss

            # Input gradient regularization: Hutchinson estimator of ||J||_F^2
            input_grad_norm_val = 0.0
            if input_grad_reg > 0.0:
                X_ = X.reshape(-1, X.shape[-1])
                U_ = U.reshape(-1, U.shape[-1])
                jac_frob_sq = hutchinson_jacobian_frobenius_sq(
                    dynamics_model, X_, U_, X.shape[-1], device
                )
                loss = loss + input_grad_reg * jac_frob_sq
                input_grad_norm_val = jac_frob_sq.sqrt().detach().item()

            loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    all_params, grad_clip
                )
            #times.append(time.perf_counter())

            optimizer.step()
            #times.append(time.perf_counter())

            # Logging
            train_loss_list.append(loss.detach())
            input_grad_norm_list.append(input_grad_norm_val)
        t1 = time.perf_counter()

        # Validation
        val_loss_list = []
        dynamics_model.eval()
        with torch.no_grad():
            for sample_i, data in enumerate(val_dataset):
                H, U, X0, X, *_ = data

                X_pred_long, _ = model_rollout(
                    H, X0, U, val_rollout_len, chunk_size=val_chunk_size)

                n_states = state_weights.shape[-1]
                loss_long = loss_fn(X_pred_long[:, :val_rollout_len, :n_states],
                                    X[:, :val_rollout_len, :n_states]) * state_weights

                val_loss_list.append(loss_long)

        t2 = time.perf_counter()

        # Logging
        mean_epoch_train_loss = torch.stack(train_loss_list, dim=0).mean().item()
        mean_epoch_val_loss = torch.stack(val_loss_list, dim=0).mean().item()
        mean_input_grad_norm = np.mean(input_grad_norm_list) if input_grad_norm_list else 0.0

        log_dict = {
            "train_long_loss": mean_epoch_train_loss,
            "val_long_loss": mean_epoch_val_loss,
            "best_long_val_loss": best_val_loss,
            "norm_grad": grad_norm_sum,
            "input_grad_norm": mean_input_grad_norm,
            "train_rollout_len": train_rollout_len,
            "val_rollout_len": val_rollout_len,
            "epoch_time": time.time() - epoch_start_time,
        }

        wandb.log(log_dict, step=epoch)
        print(log_dict)

        print_str = f"epoch {epoch}, "
        for key, value in log_dict.items():
            print_str += f" {key} {value:.4f}, "
        log.info(print_str)

        # Save best model
        if mean_epoch_val_loss < best_val_loss:
            best_val_loss = mean_epoch_val_loss
            best_dmodel = deepcopy(dynamics_model.state_dict())
            #best_model_df = deepcopy(df)
            #outpath = Path(f"{results_dir}/models") / run.name
            outpath = os.path.join(os.path.dirname(__file__), "models", run.name)
            os.makedirs(outpath, exist_ok=True)
            torch.save(best_dmodel, os.path.join(outpath, f"best_dyn_model_ep{epoch}.pt"))

        #if epoch % wandb_artifact_save_interval == 0:
        #    wandb_run_name = run.name
        #    outpath = Path(f"{results_dir}/models") / wandb_run_name / f"epoch_{epoch}"
        #    outpath.mkdir(parents=True, exist_ok=True)
        #    torch.save(best_dmodel, outpath / "dyn_model.pt")
        #    artifact = wandb.Artifact(f"model", type="model")
        #    artifact.add_dir(outpath)
        #    #artifact.add_dir(cfg_dir)
        #    artifact.save()

        #if best_model_df is not None:
        #    wandb.log({"val_rollout": wandb.Table(dataframe=best_model_df)},
        #            step=epoch)

        if torch.isnan(torch.tensor([mean_epoch_train_loss,
                                     mean_epoch_val_loss,
                                    ])).any():
            log.error("NAN detected")
            break
        t3 = time.perf_counter()
        print(f"Times: train {t1 - t0:.2f}s, val {t2 - t1:.2f}s, log {t3 - t2:.2f}s")

    # save model
    wandb_run_name = run.name
    outpath = Path(f"{results_dir}/models") / wandb_run_name
    outpath.mkdir(parents=True, exist_ok=True)
    torch.save(best_dmodel, outpath / "dyn_model.pt")
    artifact = wandb.Artifact("model", type="model")
    artifact.add_dir(outpath)
    #artifact.add_dir(cfg_dir)
    artifact.save()

    run.finish()

    return best_val_loss


if __name__ == "__main__":
    run_experiment(experiment)
