"""
Neural Network Model for Destroy Operator Selection
Simpler implementation following NDS paper Algorithm 1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class NDSModel(nn.Module):
    """
    Neural policy network π_θ for selecting customers to remove
    
    Architecture:
    - Encoder: Encodes VRP instance and current solution
    - Decoder: Selects which customers to remove (autoregressive)
    """
    
    def __init__(self, embedding_dim: int = 128, num_heads: int = 8, num_encoder_layers: int = 3, ff_dim: int = 512, dropout: float = 0.1):
        """
        Initialize NDS model
        
        Args:
            embedding_dim: Dimension of node embeddings
            num_heads: Number of attention heads
            num_encoder_layers: Number of encoder layers
            ff_dim: Feedforward hidden dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        
        # Input embedding layers
        self.depot_embedding = nn.Linear(2, embedding_dim)  # x, y
        self.node_embedding = nn.Linear(3, embedding_dim)   # x, y, demand
        
        # Encoder: Self-attention layers to encode instance and solution
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Decoder: Attention-based pointer network for selecting nodes
        self.decoder_start_token = nn.Parameter(torch.randn(embedding_dim))
        self.decoder_query = nn.Linear(embedding_dim, embedding_dim)
        self.decoder_key = nn.Linear(embedding_dim, embedding_dim)
        self.decoder_value = nn.Linear(embedding_dim, embedding_dim)
        
        # Cached encoder output
        self.encoded = None
        self.last_selected = None
        
    def encode(self, depot_xy: torch.Tensor, node_xy: torch.Tensor, node_demand: torch.Tensor) -> torch.Tensor:
        """
        Encode VRP instance
        
        Args:
            depot_xy: Depot coordinates (batch, 2)
            node_xy: Customer coordinates (batch, num_customers, 2)
            node_demand: Customer demands (batch, num_customers)
        
        Returns:
            Encoded representations (batch, num_nodes, embedding_dim)
        """
        batch_size = node_xy.shape[0]
        num_customers = node_xy.shape[1]
        
        # Embed depot and customers
        depot_embedded = self.depot_embedding(depot_xy).unsqueeze(1)  # (batch, 1, embed)
        
        node_features = torch.cat([node_xy, node_demand.unsqueeze(-1)], dim=-1)
        nodes_embedded = self.node_embedding(node_features)  # (batch, num_customers, embed)
        
        # Concatenate depot and customers
        all_nodes = torch.cat([depot_embedded, nodes_embedded], dim=1)  # (batch, num_nodes, embed)
        
        # Apply encoder
        encoded = self.encoder(all_nodes)
        
        return encoded
    
    def decode_step(self, encoded: torch.Tensor, mask: torch.Tensor = None, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select next customer to remove
        
        Args:
            encoded: Encoded node representations (batch, num_nodes, embedding_dim)
            mask: Boolean mask for already selected nodes (batch, num_nodes)
            temperature: Softmax temperature for sampling
        
        Returns:
            Tuple of (selected_indices, log_probabilities)
        """
        batch_size = encoded.shape[0]
        num_nodes = encoded.shape[1]
        
        # Get query from last selected node (or start token)
        if self.last_selected is None:
            query = self.decoder_start_token.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
        else:
            # Gather encoding of last selected node
            idx = self.last_selected.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.embedding_dim)
            query = torch.gather(encoded, 1, idx)  # (batch, 1, embedding_dim)
        
        # Compute attention scores
        Q = self.decoder_query(query)  # (batch, 1, embedding_dim)
        K = self.decoder_key(encoded)  # (batch, num_nodes, embedding_dim)
        
        # Attention logits
        logits = torch.matmul(Q, K.transpose(1, 2)) / (self.embedding_dim ** 0.5)  # (batch, 1, num_nodes)
        logits = logits.squeeze(1)  # (batch, num_nodes)
        
        # Apply mask (depot and already selected nodes)
        if mask is not None:
            logits = logits.masked_fill(mask, float('-inf'))
        
        # Convert to probabilities with temperature
        probs = F.softmax(logits / temperature, dim=-1)
        
        # Sample from distribution
        selected = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (batch,)
        log_probs = torch.log(probs.gather(1, selected.unsqueeze(-1)).squeeze(-1) + 1e-10)
        
        # Update last selected
        self.last_selected = selected
        
        return selected, log_probs
    
    def reset_decoder(self):
        """Reset decoder state for new sequence"""
        self.last_selected = None
    
    def forward(self, depot_xy: torch.Tensor, node_xy: torch.Tensor, node_demand: torch.Tensor, num_remove: int, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode instance and decode removal sequence
        
        Args:
            depot_xy: Depot coordinates (batch, 2)
            node_xy: Customer coordinates (batch, num_customers, 2)
            node_demand: Customer demands (batch, num_customers)
            num_remove: Number of customers to remove
            temperature: Sampling temperature
        
        Returns:
            Tuple of (selected_customers, log_probabilities)
        """
        batch_size = node_xy.shape[0]
        num_customers = node_xy.shape[1]
        
        # Encode instance
        encoded = self.encode(depot_xy, node_xy, node_demand)
        
        # Reset decoder
        self.reset_decoder()
        
        # Decode removal sequence
        selected_customers = []
        log_probs = []
        mask = torch.zeros(batch_size, num_customers + 1, dtype=torch.bool, device=node_xy.device)
        mask[:, 0] = True  # Mask depot (index 0)
        
        for _ in range(num_remove):
            customer_idx, log_prob = self.decode_step(encoded, mask, temperature)
            
            selected_customers.append(customer_idx)
            log_probs.append(log_prob)
            
            # Update mask
            mask.scatter_(1, customer_idx.unsqueeze(-1), True)
        
        # Stack results
        selected_customers = torch.stack(selected_customers, dim=1)  # (batch, num_remove)
        log_probs = torch.stack(log_probs, dim=1)  # (batch, num_remove)
        
        return selected_customers, log_probs


class SimpleNDSModel(nn.Module):
    """
    Even simpler model: Single-step non-autoregressive selection
    Selects all customers to remove in one shot
    """
    
    def __init__(self, embedding_dim: int = 128, num_heads: int = 8, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Input embeddings
        self.depot_embedding = nn.Linear(2, embedding_dim)
        self.node_embedding = nn.Linear(3, embedding_dim)
        
        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output head: probability of removing each customer
        self.removal_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 1)
        )
    
    def forward(self, depot_xy: torch.Tensor, node_xy: torch.Tensor, node_demand: torch.Tensor, num_remove: int, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Forward pass: encode and predict removal probabilities """
        batch_size = node_xy.shape[0]
        
        # Embed
        depot_embedded = self.depot_embedding(depot_xy).unsqueeze(1)
        node_features = torch.cat([node_xy, node_demand.unsqueeze(-1)], dim=-1)
        nodes_embedded = self.node_embedding(node_features)
        
        all_nodes = torch.cat([depot_embedded, nodes_embedded], dim=1)
        
        # Encode
        encoded = self.encoder(all_nodes)
        
        # Get removal scores (exclude depot)
        customer_encoded = encoded[:, 1:, :]  # (batch, num_customers, embedding_dim)
        removal_logits = self.removal_head(customer_encoded).squeeze(-1)  # (batch, num_customers)
        
        # Convert to probabilities
        removal_probs = F.softmax(removal_logits / temperature, dim=-1)
        
        # Sample top-k customers to remove (without replacement)
        selected_customers = torch.multinomial(removal_probs, num_samples=num_remove, replacement=False)
        
        # Get log probabilities
        log_probs = torch.log(removal_probs.gather(1, selected_customers) + 1e-10)
        
        return selected_customers, log_probs


# Helper function to create model
def create_model(model_type: str = 'simple', **kwargs) -> nn.Module:
    """
    Factory function to create NDS model
    
    Args:
        model_type: 'simple' or 'autoregressive'
        **kwargs: Model hyperparameters
    
    Returns:
        NDS model
    """
    if model_type == 'simple':
        return SimpleNDSModel(**kwargs)
    elif model_type == 'autoregressive':
        return NDSModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
