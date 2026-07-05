import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoTowerModel(nn.Module):
    """
    Two-tower recommendation model.

    USER TOWER
    -----------
    Combines:
        1. learned user ID embedding
        2. pooled representations of the user's recent
           positive movie history

    MOVIE TOWER
    -----------
    Combines:
        1. learned movie ID embedding
        2. multi-hot genre features
        3. scaled TMDB numerical features

    Text embeddings are intentionally NOT used by the
    retrieval model. They are reserved for the later RAG
    / recommendation-explanation layer.
    """

    def __init__(
        self,
        num_users: int,
        num_movies: int,
        num_genres: int,
        numerical_dim: int,
        embedding_dim: int = 128,
        output_dim: int = 128,
    ):
        super().__init__()

        # ========================================
        # USER TOWER
        # ========================================

        self.user_embedding = nn.Embedding(
            num_embeddings=num_users,
            embedding_dim=embedding_dim,
        )

        # Each history movie is represented by the
        # movie tower in output_dim dimensions.
        history_dim = output_dim

        self.user_tower = nn.Sequential(
            nn.Linear(
                embedding_dim + history_dim,
                256,
            ),
            nn.ReLU(),

            nn.Dropout(0.1),

            nn.Linear(
                256,
                output_dim,
            ),
        )

        # ========================================
        # MOVIE ID EMBEDDING
        # ========================================

        self.movie_embedding = nn.Embedding(
            num_embeddings=num_movies,
            embedding_dim=embedding_dim,
        )

        # ========================================
        # GENRE FEATURES
        # ========================================

        self.genre_projection = nn.Sequential(
            nn.Linear(
                num_genres,
                64,
            ),
            nn.ReLU(),
        )

        # ========================================
        # NUMERICAL TMDB FEATURES
        # ========================================

        self.numeric_projection = nn.Sequential(
            nn.Linear(
                numerical_dim,
                32,
            ),
            nn.ReLU(),
        )

        # ========================================
        # FINAL MOVIE REPRESENTATION
        # ========================================
        #
        # movie ID embedding
        # + genre representation
        # + numerical representation
        #
        # Text is deliberately excluded.
        # ========================================

        movie_input_dim = (
            embedding_dim
            + 64
            + 32
        )

        self.movie_tower = nn.Sequential(
            nn.Linear(
                movie_input_dim,
                256,
            ),
            nn.ReLU(),

            nn.Dropout(0.1),

            nn.Linear(
                256,
                output_dim,
            ),
        )

    # ============================================
    # Encode user from ID only
    # ============================================

    def encode_user_ids(
        self,
        users: torch.Tensor,
    ) -> torch.Tensor:

        user_vector = self.user_embedding(
            users
        )

        return F.normalize(
            user_vector,
            dim=-1,
        )

    # ============================================
    # Encode movies
    # ============================================

    def encode_movies(
        self,
        movies: torch.Tensor,
        genres: torch.Tensor,
        numerical: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode movies into normalized retrieval vectors.

        Parameters
        ----------
        movies:
            Movie indices with shape [batch_size]

        genres:
            Genre feature tensor with shape
            [batch_size, num_genres]

        numerical:
            Scaled numerical TMDB features with shape
            [batch_size, numerical_dim]

        Returns
        -------
        torch.Tensor
            Normalized movie vectors with shape
            [batch_size, output_dim]
        """

        # ----------------------------------------
        # Movie ID embedding
        # ----------------------------------------

        movie_embedding = self.movie_embedding(
            movies
        )

        # ----------------------------------------
        # Genre representation
        # ----------------------------------------

        genre_embedding = self.genre_projection(
            genres
        )

        # ----------------------------------------
        # Numerical representation
        # ----------------------------------------

        numerical_embedding = (
            self.numeric_projection(
                numerical
            )
        )

        # ----------------------------------------
        # Combine features
        # ----------------------------------------

        x = torch.cat(
            [
                movie_embedding,
                genre_embedding,
                numerical_embedding,
            ],
            dim=-1,
        )

        # ----------------------------------------
        # Movie tower
        # ----------------------------------------

        x = self.movie_tower(
            x
        )

        return F.normalize(
            x,
            dim=-1,
        )

    # ============================================
    # Encode user from movie history
    # ============================================

    def encode_users_from_history(
        self,
        users: torch.Tensor,
        history_movie_vectors: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build a user representation from recent positive
        movie history plus the learned user ID embedding.

        Parameters
        ----------
        users:
            Tensor with shape [batch_size]

        history_movie_vectors:
            Tensor with shape
            [batch_size, history_length, output_dim]

        history_mask:
            Boolean tensor with shape
            [batch_size, history_length]

        Returns
        -------
        torch.Tensor
            Normalized user vectors with shape
            [batch_size, output_dim]
        """

        # ----------------------------------------
        # Convert boolean mask to float
        # ----------------------------------------

        mask = history_mask.unsqueeze(
            -1
        ).float()

        # ----------------------------------------
        # Mask padded history entries
        # ----------------------------------------

        masked_vectors = (
            history_movie_vectors *
            mask
        )

        # ----------------------------------------
        # Mean-pool valid history movies
        # ----------------------------------------

        history_sum = masked_vectors.sum(
            dim=1
        )

        history_count = (
            mask.sum(
                dim=1
            ).clamp(
                min=1.0
            )
        )

        history_vector = (
            history_sum /
            history_count
        )

        # ----------------------------------------
        # User ID embedding
        # ----------------------------------------

        user_id_vector = self.user_embedding(
            users
        )

        # ----------------------------------------
        # Combine user ID + history
        # ----------------------------------------

        x = torch.cat(
            [
                user_id_vector,
                history_vector,
            ],
            dim=-1,
        )

        # ----------------------------------------
        # User tower
        # ----------------------------------------

        x = self.user_tower(
            x
        )

        return F.normalize(
            x,
            dim=-1,
        )

    # ============================================
    # Forward
    # ============================================

    def forward(
        self,
        users: torch.Tensor,
        positive_history_vectors: torch.Tensor,
        history_mask: torch.Tensor,
        movies: torch.Tensor,
        genres: torch.Tensor,
        numerical: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute user-movie similarity scores.

        This method is primarily useful for diagnostics.
        Training can also directly use encode_users_from_history()
        and encode_movies().
        """

        # ----------------------------------------
        # User representation
        # ----------------------------------------

        user_vectors = (
            self.encode_users_from_history(
                users,
                positive_history_vectors,
                history_mask,
            )
        )

        # ----------------------------------------
        # Movie representation
        # ----------------------------------------

        movie_vectors = self.encode_movies(
            movies,
            genres,
            numerical,
        )

        # ----------------------------------------
        # Dot-product similarity
        # ----------------------------------------

        scores = (
            user_vectors *
            movie_vectors
        ).sum(
            dim=-1
        )

        return scores