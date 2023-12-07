import os
import logging
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pigvae.trainer import PLGraphAE
from pigvae.synthetic_graphs.hyperparameter import add_arguments
from pigvae.synthetic_graphs.data import GraphDataModule, GraphDataModule_without_dynamic
from pigvae.ddp import MyDDP
from pigvae.synthetic_graphs.metrics import Critic

import torch
import umap
import random
from sklearn.preprocessing import StandardScaler
import torch.distributed as dist
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data import DenseGraphBatch
import networkx as nx
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from networkx.algorithms.similarity import graph_edit_distance
import pandas as pd

from pigvae.modules import Permuter

SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)

torch.set_default_dtype(torch.double)
logging.getLogger("lightning").setLevel(logging.WARNING)

def plot_probability_matrix(probability_matrix, adjacency, figure_name):
    # np.fill_diagonal(probability_matrix, 0)
    diagonal = torch.diag(probability_matrix)
    probability_matrix.fill_diagonal_(0)
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(15, 6), subplot_kw={"aspect": "equal"})
    sns.heatmap(probability_matrix, vmin=0, vmax=1, cmap="Blues", ax=ax1)
    sns.heatmap(adjacency, vmin=0, vmax=1, cmap="Blues", ax=ax2)
    ax1.set_title("predicted edge prob.")
    ax2.set_title("target adjacency matrix")
    plt.savefig(f"figures/heatmap{figure_name}.pdf")
    plt.close(fig)

def plot_graph(edges, title, subplot=None):
    if subplot is None:
        plt.figure(figsize=(8, 8))
    else:
        plt.subplot(*subplot)

    G = nx.Graph()
    G.add_edges_from(edges)
    nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray')
    plt.title(title)

    if subplot is None:
        plt.savefig(f"{title}.png")
        plt.close()
        
def extract_edges(edge_features_tensor):
    # Check the second channel (index 1) for a direct edge indicated by a '1'
    direct_edge_indices = edge_features_tensor[..., 1] == 1
    edges = [(i, j) for i in range(direct_edge_indices.shape[0]) for j in range(i) if direct_edge_indices[i, j]]  # Avoiding mirrored edges for undirected graphs
    return edges

def is_identity_matrix(matrix):
    identity = torch.eye(matrix.size(0)).to(matrix.device)
    return torch.equal(matrix, identity)

def check_node_order_preservation(perm):
    return is_identity_matrix(perm)

def main(hparams):
    if not os.path.isdir(hparams.save_dir + "/run{}/".format(hparams.id)):
        print("Creating directory")
        os.mkdir(hparams.save_dir + "/run{}/".format(hparams.id))
    print("Starting Run {}".format(hparams.id))
    checkpoint_callback = ModelCheckpoint(
        dirpath=hparams.save_dir + "/run{}/".format(hparams.id),
        save_last=True,
        save_top_k=1,
        monitor="val_loss"
    )
    
    early_stop_callback = EarlyStopping(
    monitor="val_loss",  # Assuming this is the metric you want to monitor
    patience=50,
    verbose=True,
    mode="min"
    )
    
    lr_logger = LearningRateMonitor()
    tb_logger = TensorBoardLogger(hparams.save_dir + "/run{}/".format(hparams.id))
    critic = Critic
    model = PLGraphAE(hparams.__dict__, critic)
    graph_kwargs = {
        "n_min": hparams.n_min,
        "n_max": hparams.n_max,
        "m_min": hparams.m_min,
        "m_max": hparams.m_max,
        "p_min": hparams.p_min,
        "p_max": hparams.p_max
    }
    datamodule = GraphDataModule(
        graph_family=hparams.graph_family,
        graph_kwargs=graph_kwargs,
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers,
        samples_per_epoch=6000
    )
    my_ddp_plugin = MyDDP()
    trainer = pl.Trainer(
        gpus=hparams.gpus,
        progress_bar_refresh_rate=5 if hparams.progress_bar else 0,
        logger=tb_logger,
        checkpoint_callback=True,
        val_check_interval=hparams.eval_freq if not hparams.test else 100,
        accelerator="ddp",
        plugins=[my_ddp_plugin],
        gradient_clip_val=0.1,
        callbacks=[lr_logger, checkpoint_callback, early_stop_callback],
        terminate_on_nan=True,
        replace_sampler_ddp=False,
        precision=hparams.precision,
        max_epochs=hparams.num_epochs,
        reload_dataloaders_every_epoch=True,
        resume_from_checkpoint=hparams.resume_ckpt if hparams.resume_ckpt != "" else None
    )
    trainer.fit(model=model, datamodule=datamodule)
    
    # After training or loading the model
    checkpoint_path = "/home/shuaiqun/Graph-instance-generation/pigvae-main/run4/epoch=55-step=10527.ckpt"
    model = PLGraphAE.load_from_checkpoint(checkpoint_path, critic=critic)  # Load from checkpoint if needed
    
    save_dir = "/home/shuaiqun/Graph-instance-generation/pigvae-main/figures"
    os.makedirs(save_dir, exist_ok=True)
    
    datamodule = GraphDataModule(
    graph_family=hparams.graph_family,
    graph_kwargs=graph_kwargs,
    batch_size=1,
    distributed_sampler=False,
    num_workers=hparams.num_workers,
    samples_per_epoch=200,
    use_saved_graphs=True,
    save_dir="/home/shuaiqun/Graph-instance-generation/pigvae-main/saved_training_samples_mix_6000"
    )

    datamodule.prepare_data()
    datamodule.setup()  # Optionally pass 'stage' if needed
    data_loader1 = datamodule.train_dataloader()  # Use the custom DataLoader for saved graphs
    N = len(data_loader1)
    save_path = "/home/shuaiqun/Graph-instance-generation/pigvae-main/figures/embeddings.npy"
    embeddings1 = model.generate_embeddings(data_loader1, save_path)
    
    graph_kwargs2 = {
        "n_min": hparams.n_min,
        "n_max": hparams.n_max,
        "m_min": hparams.m_min,
        "m_max": hparams.m_max,
        "p_min": 0.2,
        "p_max": 0.2
    }
    # List of values to iterate over
    p_values = [0.2, 0.4, 0.6, 0.8]

    # Dictionary to store the embeddings for each value
    all_embeddings = []
    all_embeddings.append(embeddings1)

    for p_value in p_values:
        # Update p_min and p_max in graph_kwargs2
        graph_kwargs2['p_min'] = p_value
        graph_kwargs2['p_max'] = p_value

        # Create a new GraphDataModule instance with updated kwargs
        datamodule2 = GraphDataModule_without_dynamic(
            graph_family=hparams.graph_family,
            graph_kwargs=graph_kwargs2,
            batch_size=1,
            num_workers=hparams.num_workers,
            samples_per_epoch=1000,
            distributed_sampler=False,
            use_full_dataset=True
        )

        # Prepare data and set up the module
        datamodule2.prepare_data()
        datamodule2.setup()  # Optionally pass 'stage' if needed

        # Get the DataLoader
        data_loader2 = datamodule2.val_dataloader()

        # Generate embeddings
        save_path = f"/home/shuaiqun/Graph-instance-generation/pigvae-main/figures/embeddings_{p_value}.npy"
        embeddings = model.generate_embeddings(data_loader2, save_path)
        all_embeddings.append(embeddings)
    
    print(len(all_embeddings))
    combined_embeddings = np.vstack(all_embeddings)
    
    reducer = umap.UMAP()
    embeddings_scaled = StandardScaler().fit_transform(combined_embeddings)
    embeddings = reducer.fit_transform(embeddings_scaled)
    # Plotting
    fig, ax = plt.subplots()
    colors = ['red', 'orange', 'blue', 'purple', 'black']  # Different colors for each group
    labels = ['p=0.5', 'p=0.2', 'p=0.4', 'p=0.6', 'p=0.8']  # Labels for each group
    group_sizes = [6000, 1000, 1000, 1000, 1000]
    point_size = 10

    start_idx = 0
    for i in range(5):
        end_idx = start_idx + group_sizes[i]
        ax.scatter(embeddings[start_idx:end_idx, 0], embeddings[start_idx:end_idx, 1], c=colors[i], label=labels[i], s=point_size)
        start_idx = end_idx
        
    ax.legend()
    plt.savefig("figures/latent_space_5_groups.pdf")
    
    threshold = 0.5  # Define your threshold
    with torch.no_grad():
        model.eval()
        
        true_labels = []
        predicted_probs = []
        
        # Initialize lists to store metrics for each batch
        mse_losses = []
        mae_losses = []
        edit_distances = []
        individual_roc_auc_scores = []
        
        input_edge_probs = []
        reconstructed_edge_probs = []

        for k, batch in enumerate(data_loader1):
            
            print("Batch structure:", batch)  # Add this line for debugging
            print(batch.edge_features[0, :, :, 0])
            x = batch.node_features.double()
            L = batch.edge_features.double()
            mask = batch.mask.double() if batch.mask is not None else None
        
            graph = DenseGraphBatch(node_features=x, edge_features=L, mask=mask)
            graph_pred_batch, perm, mu, logvar = model(graph, training=False)
            print(graph_pred_batch)
            
            # Check if the node order is preserved
            node_order_preserved = check_node_order_preservation(perm)
            print(f"Batch {k}: Node order preserved: {node_order_preserved}")
            
            plt.figure(figsize=(16, 8))  # Set up the figure for side-by-side subplots

            # Plot the original graph on the left, convert the original edge features to an adjacency matrix
            original_edge_features = batch.edge_features[0].double()
            num_nodes = original_edge_features.shape[0]
            print("Number of nodes: ", num_nodes)
            original_adjacency_matrix = torch.zeros((num_nodes, num_nodes))
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if original_edge_features[i, j, 1] > threshold:
                        original_adjacency_matrix[i, j] = 1
                
            original_edges = extract_edges(original_edge_features)
            plot_graph(original_edges, f"Original Graph {k}", subplot=(1, 2, 1))

            # Plot the reconstructed graph on the right, convert the predicted probabilities to an adjacency matrix
            reconstructed_edge_probabilities = torch.sigmoid(graph_pred_batch.edge_features[0]).double()  # Convert to double for MSE calculation
            reconstructed_adjacency_matrix = torch.zeros((num_nodes, num_nodes))
            predicted_reconstructed_adjacency_matrix = torch.zeros((num_nodes, num_nodes))
            for i in range(num_nodes):
                for j in range(num_nodes):
                    predicted_reconstructed_adjacency_matrix[i, j] = reconstructed_edge_probabilities[i, j, 1]
                    if reconstructed_edge_probabilities[i, j, 1] > threshold:
                        reconstructed_adjacency_matrix[i, j] = 1
                        
            plot_probability_matrix(predicted_reconstructed_adjacency_matrix, original_adjacency_matrix, k)
            
            reconstructed_adjacency_matrix.fill_diagonal_(0)
            input_edge_probs.append(torch.mean(original_adjacency_matrix))
            reconstructed_edge_probs.append(torch.mean(reconstructed_adjacency_matrix))
            
            # Convert the adjacency matrices to NetworkX graphs
            original_graph = nx.from_numpy_array(original_adjacency_matrix.numpy())
            reconstructed_graph = nx.from_numpy_array(reconstructed_adjacency_matrix.numpy())
            
            # Calculate and print MSE loss for each batch
            mse_loss = F.mse_loss(reconstructed_edge_probabilities, original_edge_features).item()
            mse_losses.append(mse_loss)
            
            mae_loss = np.mean(np.abs(reconstructed_edge_probabilities.cpu().numpy() - original_edge_features.cpu().numpy()))
            mae_losses.append(mae_loss)
            
            # Calculate the graph edit distance
            # distance = graph_edit_distance(original_graph, reconstructed_graph)
            # edit_distances.append(distance)
    
            num_nodes = reconstructed_edge_probabilities.shape[1]
    
            reconstructed_edges = [(a, b) for a in range(num_nodes) for b in range(num_nodes) if reconstructed_edge_probabilities[a][b, 1] > threshold]
            plot_graph(reconstructed_edges, f"Reconstructed Graph {k}", subplot=(1, 2, 2))
            
            # ROC_AUC
            # Assuming the original graph's edges are represented as binary (1 for edge, 0 for no edge)
            true_edge_features = batch.edge_features[0].double()
            
            if k ==0:
                print(true_edge_features)
            
            # Compute ROC-AUC for the current batch
            true_labels = true_edge_features[:, :, 1].flatten().tolist()
            predicted_probs = reconstructed_edge_probabilities[:, :, 1].flatten().tolist()
            roc_auc = roc_auc_score(true_labels, predicted_probs)
            
            # Store the ROC-AUC score
            individual_roc_auc_scores.append(roc_auc)
            
            # Modify the metrics text to include only ROC-AUC and Graph Edit Distance
            # metrics_text = f"ROC-AUC: {individual_roc_auc_scores[k]:.4f}\nGraph Edit Distance: {edit_distances[k]:.4f}"
            # plt.figtext(0.5, 0.01, metrics_text, ha="center", fontsize=12, bbox={"facecolor": "orange", "alpha": 0.5, "pad": 5})

            plt.savefig(f"figures/Graph Comparison {k}.png")
            plt.close()
        
        # Create a DataFrame from the scores and distances
        df = pd.DataFrame({
            'ROC_AUC_Score': individual_roc_auc_scores
            # 'Graph_Edit_Distance': edit_distances
        })

        # Specify the CSV file name
        csv_file_name = 'graph_metrics.csv'

        # Save the DataFrame to a CSV file
        df.to_csv(csv_file_name, index=False)

        print(f"Data saved to {csv_file_name}")

        # Calculate ROC-AUC
        mean_roc_auc = np.mean(individual_roc_auc_scores)
        std_error_roc_auc = np.std(individual_roc_auc_scores) / np.sqrt(len(individual_roc_auc_scores))
        
        print(f"Mean ROC-AUC: {mean_roc_auc}")
        print(f"Standard Error of ROC-AUC: {std_error_roc_auc}")

        # # Calculate mean and standard error for Graph Edit Distance
        # mean_ged = np.mean(edit_distances)
        # stderr_ged = np.std(edit_distances) / np.sqrt(len(edit_distances))

        # print(f"Mean Graph Edit Distance: {mean_ged:.4f}, Standard Error: {stderr_ged:.4f}")

        # After processing all batches
        # Prepare DataFrame for plotting
        data = pd.DataFrame({
            "kind": ["input"] * len(input_edge_probs) + ["reconstructed"] * len(reconstructed_edge_probs),
            "density": np.array(input_edge_probs + reconstructed_edge_probs)
        })

        # Plotting
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title("Edge Probability Distribution")
        sns.violinplot(data=data, x="kind", y="density", ax=ax)
        plt.savefig("figures/edge_prob_distribution.pdf")
        plt.close(fig)
        
        # Plotting the histogram
        plt.figure(figsize=(8, 6))  # You can adjust the figure size as needed
        plt.hist(edit_distances, bins='auto', color='blue', alpha=0.7)  # 'auto' lets matplotlib decide the number of bins
        plt.title('Histogram of Graph Edit Distances')
        plt.xlabel('Graph Edit Distance')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig("figures/graph_edit_distance_histogram.png")  # Saving the plot
        plt.close()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser = add_arguments(parser)
    args = parser.parse_args()
    main(args)
