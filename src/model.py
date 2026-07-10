import torch
import torch.nn as nn


class TwoTowerModel(nn.Module):
    """
    Sequential two-tower retrieval model.

    MOVIE TOWER
        movie ID embedding + MLP over content features (genres + scaled TMDB
        numericals). The content path is residual, so well-observed movies
        lean on their ID embedding and rare movies fall back on content.

    USER TOWER
        A causal Transformer over the user's most recent movies, where each
        movie is represented by the movie tower. The hidden state at the last
        position is the user vector. There is no per-user ID embedding, so
        the model works for users it has never seen as long as they have at
        least one interaction.

    Scoring is a plain dot product between user and movie vectors, which
    keeps the model servable from a vector index.

    Text embeddings are intentionally not used here; they are reserved for
    the RAG / explanation layer.
    """

    def __init__(
        self,
        genres: torch.Tensor,
        numerical: torch.Tensor,
        dim: int = 256,
        max_len: int = 50,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        num_movies = genres.shape[0]
        self.num_movies = num_movies
        self.pad_idx = num_movies
        self.dim = dim
        self.max_len = max_len

        features = torch.cat([genres, numerical], dim=1).float()
        self.register_buffer("features", features, persistent=False)

        self.movie_embedding = nn.Embedding(num_movies, dim)
        self.feature_mlp = nn.Sequential(
            nn.Linear(features.shape[1], dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        self.position_embedding = nn.Embedding(max_len, dim)
        self.input_dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.output_norm = nn.LayerNorm(dim)

        causal = torch.triu(
            torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1
        )
        self.register_buffer("causal_mask", causal, persistent=False)
        self.register_buffer(
            "self_mask", torch.eye(max_len, dtype=torch.bool), persistent=False
        )
        self.num_heads = num_heads

        nn.init.normal_(self.movie_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

    # ------------------------------------------------------------------
    # Movie tower
    # ------------------------------------------------------------------

    def item_vectors(self) -> torch.Tensor:
        """Vectors for the whole catalog, shape [num_movies, dim]."""
        return self.movie_embedding.weight + self.feature_mlp(self.features)

    # ------------------------------------------------------------------
    # User tower
    # ------------------------------------------------------------------

    def encode_sequence(
        self,
        history: torch.Tensor,
        item_vectors: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        history: [B, L] movie indices, left-padded with `pad_idx`, L <= max_len.

        Returns hidden states [B, L, dim]; position t summarises movies 0..t.
        """
        if item_vectors is None:
            item_vectors = self.item_vectors()

        length = history.shape[1]
        is_real = history != self.pad_idx
        padded = torch.cat(
            [item_vectors, item_vectors.new_zeros(1, self.dim)], dim=0
        )

        # positions are right-aligned so the newest movie always gets the
        # same position embedding regardless of history length
        positions = torch.arange(
            self.max_len - length, self.max_len, device=history.device
        )
        x = padded[history] * (self.dim ** 0.5) + self.position_embedding(positions)
        x = self.input_dropout(x) * is_real.unsqueeze(-1)

        # Block attention to future positions and to padding. Every position
        # may still attend to itself, so fully-padded prefixes never produce
        # an empty attention row.
        blocked = self.causal_mask[:length, :length].unsqueeze(0) | ~is_real.unsqueeze(1)
        blocked = blocked & ~self.self_mask[:length, :length].unsqueeze(0)
        blocked = blocked.repeat_interleave(self.num_heads, dim=0)

        x = self.encoder(x, mask=blocked)
        return self.output_norm(x) * is_real.unsqueeze(-1)

    def encode_users(
        self,
        history: torch.Tensor,
        item_vectors: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """User vectors [B, dim] from the most recent position."""
        return self.encode_sequence(history, item_vectors)[:, -1]

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """Scores for every movie, shape [B, num_movies]."""
        item_vectors = self.item_vectors()
        return self.encode_users(history, item_vectors) @ item_vectors.T
