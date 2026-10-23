import pandas as pd
import scanpy as sc
import muon as mu
import mofax as mofa
# import os
import numpy as np
from paths_and_constants import DATA_DIR, MOFA_DIR

def load_anndata(
    path_file: str, 
) -> sc.AnnData:
    """Loads an AnnData object from a CSV file located in the CLR_PROCESSED_DATA_DIR.

    Arguments:
        file_name -- Name of the CSV file to load.

    Returns:
        An AnnData object containing the loaded data.
    """
    return sc.AnnData(
        pd.read_csv(path_file, index_col=0)
    )


def create_mudata(
    list_of_files: dict | list,
) -> mu.MuData:
    """Creates a MuData object from multiple AnnData objects loaded from specified CSV files.

    Arguments:
        list_of_files -- List or Dict with paths of the CSV files containing data.

    Returns:
        A MuData object containing the loaded AnnData objects. If input is a List, modules of object returned are named as files. Otherwise, modules are named as values of input.
    """
    data_dict = {}
    if type(list_of_files) == list:
        for path_file in list_of_files:
            assert type(path_file) == str
            data_dict[path_file] = load_anndata(path_file)
        mdata = mu.MuData(
            data_dict
        )
        return mdata

    else:
        for path_file, name_module in list_of_files.items():
            assert type(path_file) == str 
            assert type(name_module) == str
            data_dict[name_module] = load_anndata(path_file)
        mdata =mu.MuData(
            data_dict
        )
        return mdata
    
def create_MOFA(
        paths_and_names : dict,
        num_factors : int,
        model_path : str
        ) -> (mu.MuData, mofa.mofa_model):
    """Creates MOFA model, and fits it by given data.
    
    Arguments:
        paths_and_names -- Dict with paths of the CSV files containing data as 
                           keys, and user-given name of that data as values.

    Returns:
        A MuData object containing information of data used, and the MOFA model
        (as mofa_model object) already fitted to data.
    """
    mdata = create_mudata(paths_and_names)
    # if not os.path.exists(MODEL_PATH):
    mu.tl.mofa(
        mdata,
        use_obs="union",
        n_factors=num_factors, 
        likelihoods=["gaussian"]*len(paths_and_names.keys()),
        convergence_mode="slow",
        outfile=model_path,
        )
    model = mofa.mofa_model(model_path)
    return mdata, model

def MOFA_analysis(
        mdata : mu.MuData,
        model : mofa.mofa_model,
        out_dir: str,
        name_file: str = 'MOFA',
        get_weights: bool = False,
        )-> None | dict:
    """Given processed data by MOFA method (using 'create_MOFA', for example),
    saves main factor matrix and weights in specified folder and with a 
    specified name.
    
    
    Arguments:
        mdata -- mu.MuData type, having information about the model.
        model -- mofa.mofa_model type, containing transformation and factor 
                 information.
        out_dir -- Path to folder where the files will be saved.
        name_file -- Name to identify type of data worked. Useful when 
                     comparing diferent sets of data.

    Returns:
        Either nothing, or the weight matrix fiven by MOFA of each set of data 
        that created 'mdata' (useful for extracting features from factors).
    """
    
    #Get and save global factors
    Z = model.get_factors()
    Z_df = pd.DataFrame(Z)
    Z_df.columns = [f"Factor_{i + 1}" for i in range(Z_df.shape[1])]
    Z_df.index = mdata.obs_names
    Z_df.to_csv(f"{out_dir}M_factors_{name_file}.csv")

    #Get and save weights
    W = model.get_weights()
    n_data = {
        name: mod.n_vars for name, mod in mdata.mod.items()
             }
    slice_limits = np.cumulative_sum([*n_data.values()], include_initial=True)
    idx_data = {
        [*n_data.keys()][k]: slice(
            slice_limits[k],
            slice_limits[k+1],
        ) for k in range(mdata.n_mod)
    }
    output = {
        name: W[index,:] for name, index in idx_data.items()
    }
    
    for model_name in mdata.mod.keys():
        pd.DataFrame(output[model_name], 
                     index=mdata.mod[model_name].var_names,
                     columns=[f"Factor_{_}" for _ in range(1,model.nfactors)],
                    ).to_csv(
            f"{out_dir}W_{model_name}_{name_file}.csv"
        )
    if get_weights:    
        return output
    
def MOFA_exp_var(
        model : mofa.mofa_model,
        path_file : str = 'M_variance_explained.csv',
        clean_format: bool = True, 
        show : bool = False,
        )-> pd.DataFrame:
    """Creates tables with explained variance of original data in function of 
    factors given by MOFA mode. Useful to create graphs and evaluate  efficacy.
    
    Arguments:
        model -- mofa.mofa_model object to evaluate.
        name_file -- Name of file to save table of explained variance. Set 
                     as empty string or None object if saving the table is not
                     required.
        clean_format -- True if correct format of columns is required. Useful 
                        for future display.
        show -- Set as True if you wish to visualize a preview of the table.
        

    Returns:
        Either nothing, or the weight matrix fiven by MOFA of each set of data 
        that created 'mdata' (useful for extracting features from factors).
    """
    
    variance_explained = model.get_r2()
    if clean_format:
        variance_explained.drop(columns=["Group"], inplace=True)
        variance_explained["Factor"] = variance_explained["Factor"].str.replace(
            r"Factor(\d{1})$", r"Factor0\1", regex=True
            )
    variance_explained_pivot = variance_explained.pivot(
        index="Factor", columns="View", values="R2"
    )
    if path_file not in [None,'']:
        variance_explained_pivot.to_csv(path_file)
    
    if show:
        import matplotlib.pyplot as plt
        #Visualización mediante gráfico de tabla 
        plt.figure(figsize=(15, 10))
        plt.imshow(variance_explained_pivot, aspect="auto", cmap="Blues")
        plt.xticks(
            ticks=range(len(variance_explained_pivot.columns)),
            labels=variance_explained_pivot.columns,
            rotation=45,
        )
        plt.yticks(
            ticks=range(len(variance_explained_pivot.index)),
            labels=variance_explained_pivot.index,
        )
        plt.colorbar(label="R2 (%)")
        plt.title("Variance Explained per Factor and View")
        plt.tight_layout()
    
    return variance_explained_pivot


def performance_mofa(
    k : int,
    Data: dict,
)-> list:
    """
    (Deprecated) Auxiliar method for choosing hyperparameters by brute force. 
    """
    assert k>=2
    mdata, model = create_MOFA(Data, num_factors=k, name_model_file=f'mofa_model_k{k}.hdf5')
    ve = MOFA_exp_var(model, save_result=False)
    min_ve = np.min([np.sum(ve[col]) for col in ve.columns.array])
    return [k, min_ve]