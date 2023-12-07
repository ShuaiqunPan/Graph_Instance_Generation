import torch
import pytorch_lightning as pl
from pigvae.modules import GraphAE
from sklearn.metrics import roc_auc_score
from data import DenseGraphBatch
import numpy as np


class PLGraphAE(pl.LightningModule):

    def __init__(self, hparams, critic):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.graph_ae = GraphAE(hparams)
        self.critic = critic(hparams)

    def forward(self, graph, training):
        graph_pred, perm, mu, logvar = self.graph_ae(graph, training, tau=1.0)
        return graph_pred, perm, mu, logvar
    
    def calculate_roc_auc(self, graph_true, graph_pred):
        with torch.no_grad():
            # Ensure the shapes are correct and consistent
            assert graph_true.edge_features.shape == graph_pred.edge_features.shape

            num_graphs = graph_true.edge_features.shape[0]

            roc_auc_scores = []
            for i in range(num_graphs):
                true_edges = graph_true.edge_features[i, :, :, 1].flatten()
                pred_edges = torch.sigmoid(graph_pred.edge_features[i, :, :, 1]).flatten()

                # Optional: Apply a mask if your graphs are sparse or irregular
                # mask = ... (define your mask based on your graph structure)
                # true_edges = true_edges[mask]
                # pred_edges = pred_edges[mask]

                true_edges_np = true_edges.cpu().numpy()
                pred_edges_np = pred_edges.cpu().numpy()

                # Calculate ROC-AUC per graph and store
                roc_auc = roc_auc_score(true_edges_np, pred_edges_np)
                roc_auc_scores.append(roc_auc)

            # Aggregate ROC-AUC scores
            avg_roc_auc = sum(roc_auc_scores) / len(roc_auc_scores)
            return avg_roc_auc

    def training_step(self, graph, batch_idx):
        graph_pred, perm, mu, logvar = self(
            graph=graph,
            training=True,
        )
        loss = self.critic(
            graph_true=graph,
            graph_pred=graph_pred,
            perm=perm,
            mu=mu,
            logvar=logvar,
        )
        roc_auc = self.calculate_roc_auc(graph, graph_pred)
        loss['train_roc_auc'] = roc_auc
        self.log_dict(loss)
        return loss

    def validation_step(self, graph, batch_idx):
        graph_pred, perm, mu, logvar = self(
            graph=graph,
            training=True,
        )
        metrics_soft = self.critic.evaluate(
            graph_true=graph,
            graph_pred=graph_pred,
            perm=perm,
            mu=mu,
            logvar=logvar,
            prefix="val",
        )
        roc_auc_soft = self.calculate_roc_auc(graph, graph_pred)
        metrics_soft['val_roc_auc'] = roc_auc_soft
        
        graph_pred, perm, mu, logvar = self(
            graph=graph,
            training=False,
        )
        metrics_hard = self.critic.evaluate(
            graph_true=graph,
            graph_pred=graph_pred,
            perm=perm,
            mu=mu,
            logvar=logvar,
            prefix="val_hard",
        )
        roc_auc_hard = self.calculate_roc_auc(graph, graph_pred)
        metrics_hard['val_hard_roc_auc'] = roc_auc_hard
        
        metrics = {**metrics_soft, **metrics_hard}
        self.log_dict(metrics)
        self.log_dict(metrics_soft)

    def generate_embeddings(self, data_loader, save_path):
        self.eval()
        embeddings = []
        with torch.no_grad():
            for batch in data_loader:
                x = batch.node_features.double()  # Convert to double
                L = batch.edge_features.double()  # Convert to double
                mask = batch.mask.double() if batch.mask is not None else None

                graph = DenseGraphBatch(node_features=x, edge_features=L, mask=mask)
                graph_emb, _, _, _ = self.graph_ae.encode(graph)
                embedding = graph_emb.cpu().numpy().flatten()
                embeddings.append(embedding)
        print(len(embeddings))
        
        # Convert list of arrays to a single NumPy array
        embeddings_array = np.array(embeddings)
        # Save to a .npy file
        np.save(save_path, embeddings_array)
        print(f"Embeddings saved to {save_path}")
    
        return embeddings

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.graph_ae.parameters(), lr=self.hparams["lr"], betas=(0.9, 0.98))
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer,
            gamma=0.999,
        )
        if "eval_freq" in self.hparams:
            scheduler = {
                'scheduler': lr_scheduler,
                'interval': 'step',
                'frequency': 2 * (self.hparams["eval_freq"] + 1)
            }
        else:
            scheduler = {
                'scheduler': lr_scheduler,
                'interval': 'epoch'
            }
        return [optimizer], [scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure=None,
                       second_order_closure=None, on_tpu=False, using_native_amp=False, using_lbfgs=False):
        # warm up lr
        if self.trainer.global_step < 10000:
            lr_scale = min(1., float(self.trainer.global_step + 1) / 10000.)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.hparams.lr

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

