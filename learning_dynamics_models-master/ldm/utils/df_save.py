import torch
import pandas as pd


def create_df(X, X_sim, u,  wandb_table_len, state_names, control_names):
    
    dict = {}
    for i in range(X.shape[-1]):
        dict[state_names[i] + "_sim"] = X_sim[:, :, i].reshape(-1)[:wandb_table_len]
        dict[state_names[i]] = X[:, :, i].reshape(-1)[:wandb_table_len]
    
    for i in range(u.shape[-1]):
        dict[control_names[i]] = u[:, :, i].reshape(-1)[:wandb_table_len]
        
    df = pd.DataFrame(dict)
    return df


def create_df_p_traj(p, p_const, wandb_table_len : int, param_names : list):
    df_list = []
    param_names  = param_names.copy()

    for i, param_name in enumerate(param_names):
        df = pd.DataFrame({param_name: p[:, :, i].reshape(-1)[:wandb_table_len],
                           param_name + "_const": p_const[i]})
        df_list.append(df)
    
    df = pd.concat(df_list, axis=1)
        
    return df