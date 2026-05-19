"""Geometric Manifold Rectification for imbalanced binary classification."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors
from typing import Optional


class GeometricManifoldRectifier(BaseEstimator):
    """Data-gated geometric preprocessing for imbalanced tabular learning.

    The rectifier estimates local geometric pressure from neighborhood overlap,
    entropy, boundary conflict, sparsity, and imbalance. When the training split
    already looks sufficiently well supported, it returns the input unchanged.
    Otherwise it applies bounded majority boundary pruning and safe same-class
    synthesis. In bidirectional mode, low-support majority regions may also be
    lightly densified under a strict global budget.

    Parameters are intentionally conservative by default. The class exposes an
    imbalanced-learn-style ``fit_resample`` method and stores a lightweight audit
    dictionary in ``audit_`` after each call.
    """

    def __init__(
        self,
        k_neighbors: int = 15,
        metric_switch_dim: int = 100,
        activation_threshold: float = 0.62,
        activation_temperature: float = 0.08,
        activation_floor: float = 0.12,
        max_majority_remove: float = 0.006,
        prune_quantile: float = 0.96,
        max_synthetic_ratio: float = 0.06,
        max_prior_shift: float = 0.010,
        safe_anchor_quantile: float = 0.55,
        min_safe_score: float = 0.45,
        min_support_for_intervention: int = 40,
        max_support_for_intervention: int = 100,
        sufficiency_noop_threshold: float = 0.25,
        allow_bidirectional_synthesis: bool = True,
        max_majority_synthetic_ratio: float = 0.012,
        max_total_synthetic_ratio: float = 0.018,
        class_support_target: int = 500,
        min_majority_sparsity_for_synthesis: float = 0.0,
        max_majority_sufficiency_for_synthesis: float = 1.0,
        max_majority_synthetic_per_split: Optional[int] = None,
        use_sample_weight: bool = False,
        apply_weights_in_identity: bool = False,
        min_weight_signal: float = 0.08,
        class_weight_power: float = 0.0,
        majority_boundary_downweight: float = 0.0,
        minority_geometry_boost: float = 0.0,
        max_sample_weight: float = 3.0,
        random_state: Optional[int] = None,
    ):
        self.k_neighbors = k_neighbors
        self.metric_switch_dim = metric_switch_dim
        self.activation_threshold = activation_threshold
        self.activation_temperature = activation_temperature
        self.activation_floor = activation_floor
        self.max_majority_remove = max_majority_remove
        self.prune_quantile = prune_quantile
        self.max_synthetic_ratio = max_synthetic_ratio
        self.max_prior_shift = max_prior_shift
        self.safe_anchor_quantile = safe_anchor_quantile
        self.min_safe_score = min_safe_score
        self.min_support_for_intervention = min_support_for_intervention
        self.max_support_for_intervention = max_support_for_intervention
        self.sufficiency_noop_threshold = sufficiency_noop_threshold
        self.allow_bidirectional_synthesis = allow_bidirectional_synthesis
        self.max_majority_synthetic_ratio = max_majority_synthetic_ratio
        self.max_total_synthetic_ratio = max_total_synthetic_ratio
        self.class_support_target = class_support_target
        self.min_majority_sparsity_for_synthesis = min_majority_sparsity_for_synthesis
        self.max_majority_sufficiency_for_synthesis = max_majority_sufficiency_for_synthesis
        self.max_majority_synthetic_per_split = max_majority_synthetic_per_split
        self.use_sample_weight = use_sample_weight
        self.apply_weights_in_identity = apply_weights_in_identity
        self.min_weight_signal = min_weight_signal
        self.class_weight_power = class_weight_power
        self.majority_boundary_downweight = majority_boundary_downweight
        self.minority_geometry_boost = minority_geometry_boost
        self.max_sample_weight = max_sample_weight
        self.random_state = random_state
        self.sample_weight_ = None
        self.audit_ = {}

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-z))

    def _geometry(self, X, y, minority_class, majority_class):
        sample_count = len(y)
        neighbor_count = max(1, min(self.k_neighbors, sample_count - 1))
        metric = "cosine" if X.shape[1] > self.metric_switch_dim else "euclidean"
        neighbors = NearestNeighbors(n_neighbors=neighbor_count + 1, metric=metric)
        neighbors.fit(X)
        distances, indices = neighbors.kneighbors(X, return_distance=True)
        neighbor_indices = indices[:, 1:]
        neighbor_distances = distances[:, 1:]

        weights = 1.0 / (neighbor_distances + 1e-8)
        weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
        neighbor_labels = y[neighbor_indices]
        p_minority = (weights * (neighbor_labels == minority_class)).sum(axis=1)
        p_majority = (weights * (neighbor_labels == majority_class)).sum(axis=1)
        p0 = np.clip(p_majority, 1e-8, 1.0)
        p1 = np.clip(p_minority, 1e-8, 1.0)
        entropy = -(p0 * np.log(p0) + p1 * np.log(p1)) / np.log(2.0)

        nearest = neighbor_indices[:, 0]
        mutual_conflict = (y[nearest] != y) & (nearest[nearest] == np.arange(sample_count))

        prototype_count = np.zeros(sample_count, dtype=float)
        for index in range(sample_count):
            same_class = neighbor_indices[index][y[neighbor_indices[index]] == y[index]]
            if same_class.size:
                prototype_count[same_class[0]] += 1.0
        prototype_score = prototype_count / max(1.0, prototype_count.max())
        return neighbor_indices, p_minority, p_majority, entropy, mutual_conflict, prototype_score

    def _synthesize_class(
        self,
        X,
        y,
        target_class,
        target_mask,
        intrusion_probability,
        prototype_score,
        n_generate,
        rng,
    ):
        if n_generate <= 0 or int(target_mask.sum()) < 2:
            return X, y, 0

        target_indices = np.where(target_mask)[0]
        X_target = X[target_indices]
        neighbor_count = max(1, min(5, len(X_target) - 1))
        same_class_neighbors = NearestNeighbors(n_neighbors=neighbor_count + 1, metric="euclidean")
        same_class_neighbors.fit(X_target)
        _, same_neighbor_indices = same_class_neighbors.kneighbors(X_target, return_distance=True)

        safe_score = (1.0 - intrusion_probability[target_indices]) * (
            0.75 + 0.25 * prototype_score[target_indices]
        )
        threshold = max(self.min_safe_score, float(np.quantile(safe_score, self.safe_anchor_quantile)))
        anchor_indices = np.where(safe_score >= threshold)[0]
        if anchor_indices.size == 0:
            return X, y, 0

        synthetic = np.zeros((n_generate, X.shape[1]), dtype=float)
        for row_index in range(n_generate):
            anchor = int(rng.choice(anchor_indices))
            neighbor_choices = same_neighbor_indices[anchor, 1:]
            neighbor = int(rng.choice(neighbor_choices)) if neighbor_choices.size else anchor
            interpolation = float(rng.uniform(0.20, 0.80))
            synthetic[row_index] = (1.0 - interpolation) * X_target[anchor] + interpolation * X_target[neighbor]

        synthetic_y = np.full(n_generate, target_class, dtype=y.dtype)
        return np.vstack([X, synthetic]), np.hstack([y, synthetic_y]), int(n_generate)

    def _synthesize(self, X, y, minority_class, minority_mask, local_majority_probability, prototype_score, n_generate):
        rng = np.random.default_rng(self.random_state)
        return self._synthesize_class(
            X,
            y,
            minority_class,
            minority_mask,
            local_majority_probability,
            prototype_score,
            n_generate,
            rng,
        )

    @staticmethod
    def _class_sparsity(X_target, X_other):
        if len(X_target) < 2 or len(X_other) < 2:
            return 1.0
        same_neighbor_count = max(1, min(5, len(X_target) - 1))
        same_neighbors = NearestNeighbors(n_neighbors=same_neighbor_count + 1).fit(X_target)
        same_distances, _ = same_neighbors.kneighbors(X_target)
        other_neighbors = NearestNeighbors(n_neighbors=min(5, len(X_other))).fit(X_other)
        cross_distances, _ = other_neighbors.kneighbors(X_target)
        return float(np.clip(np.mean(same_distances[:, 1:]) / (np.mean(cross_distances[:, :1]) + 1e-8), 0.0, 1.0))

    def _make_sample_weight(self, X, y, activation, imbalance_pressure):
        if not self.use_sample_weight or len(y) == 0 or len(np.unique(y)) < 2:
            return None

        classes, counts = np.unique(y, return_counts=True)
        minority_class = classes[np.argmin(counts)]
        majority_class = classes[np.argmax(counts)]
        _, p_minority, p_majority, _, _, prototype_score = self._geometry(X, y, minority_class, majority_class)

        class_counts = {cls: count for cls, count in zip(classes, counts)}
        weight_power = self.class_weight_power * float(np.clip(imbalance_pressure, 0.0, 1.0))
        sample_weight = np.ones(len(y), dtype=float)
        if weight_power > 0:
            for cls in classes:
                class_mask = y == cls
                balanced_factor = len(y) / max(1.0, len(classes) * class_counts[cls])
                sample_weight[class_mask] *= float(balanced_factor) ** weight_power

        minority_mask = y == minority_class
        majority_mask = y == majority_class
        intrusion = np.where(minority_mask, p_majority, p_minority)
        if self.majority_boundary_downweight > 0:
            sample_weight[majority_mask] *= np.exp(
                -self.majority_boundary_downweight * activation * intrusion[majority_mask]
            )
        if self.minority_geometry_boost > 0:
            safe_support = (1.0 - intrusion[minority_mask]) * (0.75 + 0.25 * prototype_score[minority_mask])
            sample_weight[minority_mask] *= 1.0 + self.minority_geometry_boost * activation * safe_support

        max_weight = max(1.0, float(self.max_sample_weight))
        sample_weight = np.clip(sample_weight, 1.0 / max_weight, max_weight)
        return sample_weight / max(1e-12, float(np.mean(sample_weight)))

    def fit_resample(self, X, y):
        """Return a rectified training split.

        ``X`` must be numeric and already encoded. ``y`` is expected to contain
        two classes; the class with fewer samples is treated as minority.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        sample_count = len(y)
        self.sample_weight_ = None
        if sample_count == 0 or len(np.unique(y)) < 2:
            self.audit_ = {
                "activation": 0.0,
                "majority_removed": 0,
                "synthetic_added": 0,
                "majority_synthetic_added": 0,
                "profile": "identity_empty",
            }
            return X, y
        if len(np.unique(y)) > 2:
            raise ValueError("GeometricManifoldRectifier currently expects binary labels.")

        classes, counts = np.unique(y, return_counts=True)
        minority_class = classes[np.argmin(counts)]
        majority_class = classes[np.argmax(counts)]
        minority_count = int(np.min(counts))
        majority_count = int(np.max(counts))
        minority_mask = y == minority_class
        majority_mask = y == majority_class

        _, p_minority, p_majority, entropy, mutual_conflict, prototype_score = self._geometry(
            X, y, minority_class, majority_class
        )
        local_majority_probability = p_majority
        local_minority_probability = p_minority
        minority_intrusion = float(np.mean(local_majority_probability[minority_mask])) if minority_mask.any() else 0.0
        majority_intrusion = float(np.mean(local_minority_probability[majority_mask])) if majority_mask.any() else 0.0
        overlap = 0.5 * (minority_intrusion + majority_intrusion)
        boundary_conflict = float(np.mean(mutual_conflict))
        entropy_mean = float(np.mean(entropy))

        imbalance_ratio = majority_count / max(1, minority_count)
        imbalance_pressure = float(np.clip(np.log1p(imbalance_ratio) / np.log1p(20.0), 0.0, 1.0))
        minority_sufficiency = float(np.clip((minority_count - 80.0) / 720.0, 0.0, 1.0))

        X_minority = X[minority_mask]
        X_majority = X[majority_mask]
        if len(X_minority) >= 2 and len(X_majority) >= 2:
            minority_neighbor_count = max(1, min(5, len(X_minority) - 1))
            minority_neighbors = NearestNeighbors(n_neighbors=minority_neighbor_count + 1).fit(X_minority)
            minority_distances, _ = minority_neighbors.kneighbors(X_minority)
            majority_neighbors = NearestNeighbors(n_neighbors=min(5, len(X_majority))).fit(X_majority)
            cross_distances, _ = majority_neighbors.kneighbors(X_minority)
            minority_sparsity = float(
                np.clip(np.mean(minority_distances[:, 1:]) / (np.mean(cross_distances[:, :1]) + 1e-8), 0.0, 1.0)
            )
            majority_sparsity = self._class_sparsity(X_majority, X_minority)
        else:
            minority_sparsity = imbalance_pressure
            majority_sparsity = imbalance_pressure

        geometric_need = float(
            np.clip(
                0.35 * overlap
                + 0.25 * entropy_mean
                + 0.20 * boundary_conflict
                + 0.20 * minority_sparsity
                + 0.20 * imbalance_pressure * (1.0 - minority_sufficiency),
                0.0,
                1.0,
            )
        )
        activation = float(self._sigmoid((geometric_need - self.activation_threshold) / self.activation_temperature))

        synthesis_pressure = float(
            np.clip(
                imbalance_pressure
                * minority_sparsity
                * (1.0 - overlap)
                * (1.0 - boundary_conflict)
                * (1.0 - minority_sufficiency),
                0.0,
                1.0,
            )
        )

        support_guard = minority_count < self.min_support_for_intervention
        large_support_guard = minority_count > self.max_support_for_intervention and geometric_need < 0.70
        sufficiency_guard = minority_sufficiency > self.sufficiency_noop_threshold and geometric_need < 0.70
        if max(activation, synthesis_pressure) < self.activation_floor or support_guard or large_support_guard or sufficiency_guard:
            weighted_identity = False
            if self.apply_weights_in_identity:
                weight_signal = max(activation, synthesis_pressure, imbalance_pressure * (1.0 - minority_sufficiency))
                if weight_signal >= self.min_weight_signal:
                    self.sample_weight_ = self._make_sample_weight(X, y, activation, imbalance_pressure)
                    weighted_identity = self.sample_weight_ is not None
            self.audit_ = {
                "activation": activation,
                "geometric_need": geometric_need,
                "majority_removed": 0,
                "synthetic_added": 0,
                "majority_synthetic_added": 0,
                "weighted": int(weighted_identity),
                "profile": "identity",
            }
            return X, y

        risk = (
            0.42 * np.where(majority_mask, local_minority_probability, local_majority_probability)
            + 0.28 * entropy
            + 0.20 * mutual_conflict.astype(float)
            - 0.25 * prototype_score
        )
        risk = np.clip(risk, 0.0, 1.0)
        remove_majority = np.zeros(sample_count, dtype=bool)
        majority_indices = np.where(majority_mask)[0]
        majority_budget = int(np.floor(self.max_majority_remove * activation * majority_indices.size))
        if majority_budget > 0 and majority_indices.size:
            risk_threshold = float(np.quantile(risk[majority_indices], self.prune_quantile))
            eligible = majority_indices[(risk[majority_indices] >= risk_threshold) & (local_minority_probability[majority_indices] > 0.0)]
            if eligible.size:
                ordered = eligible[np.argsort(risk[eligible])[::-1]]
                remove_majority[ordered[:majority_budget]] = True

        keep = ~remove_majority
        X_out, y_out = X[keep], y[keep]
        local_majority_kept = local_majority_probability[keep]
        local_minority_kept = local_minority_probability[keep]
        prototype_kept = prototype_score[keep]
        minority_kept = y_out == minority_class

        max_by_ratio = int(np.floor(self.max_synthetic_ratio * minority_count * synthesis_pressure))
        max_by_prior = int(np.floor(self.max_prior_shift * len(y_out) / max(1e-12, 1.0 - self.max_prior_shift)))
        n_generate = max(0, min(max_by_ratio, max_by_prior, majority_count - minority_count))
        rng = np.random.default_rng(self.random_state)
        X_out, y_out, synthetic_added = self._synthesize(
            X_out,
            y_out,
            minority_class,
            minority_kept,
            local_majority_kept,
            prototype_kept,
            n_generate,
        )

        majority_synthetic_added = 0
        if self.allow_bidirectional_synthesis:
            majority_sufficiency = float(
                np.clip((majority_count - self.class_support_target) / max(1.0, 4.0 * self.class_support_target), 0.0, 1.0)
            )
            conflict_majority = float(np.mean(mutual_conflict[majority_mask])) if majority_mask.any() else 0.0
            if (
                majority_sparsity < self.min_majority_sparsity_for_synthesis
                or majority_sufficiency > self.max_majority_sufficiency_for_synthesis
            ):
                majority_synthesis_pressure = 0.0
            else:
                majority_synthesis_pressure = float(
                    np.clip(
                        majority_sparsity
                        * (1.0 - majority_intrusion)
                        * (1.0 - conflict_majority)
                        * (1.0 - majority_sufficiency),
                        0.0,
                        1.0,
                    )
                )
            total_cap = int(np.floor(self.max_total_synthetic_ratio * sample_count))
            max_majority_by_ratio = int(
                np.floor(self.max_majority_synthetic_ratio * majority_count * majority_synthesis_pressure)
            )
            max_majority_by_prior = int(
                np.floor(self.max_prior_shift * len(y_out) / max(1e-12, 1.0 - self.max_prior_shift))
            )
            n_generate_majority = max(0, min(max_majority_by_ratio, max_majority_by_prior, max(0, total_cap - synthetic_added)))
            if self.max_majority_synthetic_per_split is not None:
                n_generate_majority = min(n_generate_majority, max(0, int(self.max_majority_synthetic_per_split)))
            if n_generate_majority > 0:
                if len(y_out) != len(local_minority_kept):
                    pad = len(y_out) - len(local_minority_kept)
                    local_minority_for_out = np.hstack([local_minority_kept, np.zeros(pad)])
                    prototype_for_out = np.hstack([prototype_kept, np.ones(pad)])
                else:
                    local_minority_for_out = local_minority_kept
                    prototype_for_out = prototype_kept
                majority_out = y_out == majority_class
                X_out, y_out, majority_synthetic_added = self._synthesize_class(
                    X_out,
                    y_out,
                    majority_class,
                    majority_out,
                    local_minority_for_out,
                    prototype_for_out,
                    n_generate_majority,
                    rng,
                )

        self.audit_ = {
            "activation": activation,
            "geometric_need": geometric_need,
            "overlap": overlap,
            "boundary_conflict": boundary_conflict,
            "minority_sparsity": minority_sparsity,
            "majority_sparsity": majority_sparsity,
            "imbalance_pressure": imbalance_pressure,
            "minority_sufficiency": minority_sufficiency,
            "majority_removed": int(remove_majority.sum()),
            "synthetic_added": int(synthetic_added),
            "majority_synthetic_added": int(majority_synthetic_added),
            "weighted": int(False),
            "profile": "rectified" if int(remove_majority.sum()) or int(synthetic_added) or int(majority_synthetic_added) else "identity_budget_zero",
        }
        if self.use_sample_weight:
            self.sample_weight_ = self._make_sample_weight(X_out, y_out, activation, imbalance_pressure)
            self.audit_["weighted"] = int(self.sample_weight_ is not None)
        return X_out, y_out


GMR = GeometricManifoldRectifier
