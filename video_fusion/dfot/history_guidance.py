"""
History Guidance for Diffusion Forcing Video Fusion
Simplified version adapted from diffusion-forcing-transformer
"""

from typing import List, Tuple, Literal, Optional, Callable
from collections import defaultdict
import torch
from einops import repeat, rearrange, reduce, einsum

ALL = "all"
ALLType = Literal["all"]
FreqRanges = List[Tuple[float, float] | ALLType]


class HistorySegment:
    """
    A class representing a single segment of history for guidance.

    Attributes:
        time_indices: List of frame indices to use, or "all"
        freq_ranges: Noise level ranges for each frame
        freq_ranges_if_generated: Ranges for generated (vs ground truth) frames
    """

    def __init__(
        self,
        time_indices: List[int] | ALLType = ALL,
        freq_ranges: Optional[FreqRanges] = None,
        freq_ranges_if_generated: Optional[FreqRanges] = None,
    ):
        self.time_indices = time_indices
        self.freq_ranges = freq_ranges if freq_ranges is not None else [ALL]
        self.freq_ranges_if_generated = (
            self.freq_ranges
            if freq_ranges_if_generated is None
            else freq_ranges_if_generated
        )

    def _process_freq_ranges(
        self, freq_ranges: FreqRanges, len_chosen: int
    ) -> List[Tuple[float, float]]:
        """Process frequency ranges, replacing ALL with (0.0, 1.0)"""
        freq_ranges = [
            freq_range if freq_range != ALL else (0.0, 1.0)
            for freq_range in freq_ranges
        ]

        # Case #1: exact match
        if len(freq_ranges) == len_chosen:
            return freq_ranges

        # Case #2: interpolate between two ranges
        if len(freq_ranges) == 2:
            if len_chosen == 1:
                return [freq_ranges[1]]
            first_start, first_end = freq_ranges[0]
            last_start, last_end = freq_ranges[1]
            return [
                (
                    first_start + (last_start - first_start) * t / (len_chosen - 1),
                    first_end + (last_end - first_end) * t / (len_chosen - 1),
                )
                for t in range(len_chosen)
            ]

        # Case #3: constant for all
        if len(freq_ranges) == 1:
            return freq_ranges * len_chosen

        raise ValueError(
            f"Length mismatch: history={len_chosen}, freq_ranges={len(freq_ranges)}"
        )

    def to_noise_levels(
        self, hist_mask: torch.Tensor
    ) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        """
        Convert history segment to noise levels.

        Args:
            hist_mask: (hist_len,) True=generated, False=ground truth

        Returns:
            (start_noise_levels, end_noise_levels) each of length hist_len
        """
        hist_len = hist_mask.size(0)
        generated_indices = torch.where(hist_mask)[0].tolist()

        time_indices: List[int] = (
            self.time_indices if self.time_indices != ALL else list(range(hist_len))
        )

        # Handle negative indices
        time_indices = [
            t if t >= 0 else hist_len + t for t in time_indices
        ]

        freq_ranges = self._process_freq_ranges(self.freq_ranges, len(time_indices))
        freq_ranges_if_generated = self._process_freq_ranges(
            self.freq_ranges_if_generated, len(time_indices)
        )

        # Default: full noise (1.0, 1.0)
        final_freq_ranges = [(1.0, 1.0)] * hist_len

        for i, t in enumerate(time_indices):
            final_freq_ranges[t] = (
                freq_ranges_if_generated[i]
                if t in generated_indices
                else freq_ranges[i]
            )

        return tuple(zip(*final_freq_ranges)) if hist_len > 0 else ((), ())

    @classmethod
    def full(cls) -> "HistorySegment":
        """Full history with all frequencies"""
        return cls(time_indices=ALL, freq_ranges=[ALL])

    @classmethod
    def partial_constant(cls, start_freq: float, end_freq: float) -> "HistorySegment":
        """Constant frequency range for all frames"""
        return cls(time_indices=ALL, freq_ranges=[(start_freq, end_freq)])


class HistoryGuidanceManager:
    """
    Context manager that applies history guidance during sampling.
    """

    def __init__(self, history_guidance: "HistoryGuidance", mask: torch.Tensor):
        """
        Args:
            history_guidance: Parent HistoryGuidance object
            mask: (batch_size, seq_len)
                  0=to generate, 1=ground truth, 2=generated, -1=padding
        """
        self.history_guidance = history_guidance
        self.mask = mask
        self.device = mask.device

    @property
    def nfe(self) -> int:
        """Number of Function Evaluations per step"""
        return self.num_gen * self.num_hist

    def __enter__(self):
        """Precompute all partial history conditions"""

        # 1. Get indices
        reduced_mask = self.mask[0]
        assert (self.mask == reduced_mask).all(), "Mask must be same across batch"

        self.hist_indices = torch.where(reduced_mask >= 1)[0]
        self.gen_indices = torch.where(reduced_mask == 0)[0]
        seq_len, hist_len, gen_len = (
            len(reduced_mask),
            len(self.hist_indices),
            len(self.gen_indices),
        )

        # 2. Precompute gen_mask
        gen_segments = [
            seg if seg != ALL else list(range(gen_len))
            for seg in self.history_guidance.gen_segments
        ]
        self.num_gen = len(gen_segments)
        gen_mask = torch.zeros(
            (self.num_gen, seq_len), dtype=torch.bool, device=self.device
        )
        for i, gen_segment in enumerate(gen_segments):
            gen_mask[i, self.gen_indices[gen_segment]] = True
        self.gen_mask = gen_mask

        # 3. Build history conditions dictionary
        hist_to_weights: Dict[tuple, float] = defaultdict(float)

        # Unconditional score
        hist_to_weights[
            (1.0,) * hist_len + (self.history_guidance.use_external_cond_guidance,)
        ] = 1.0

        # Add history segments
        for hist_segment, weight in zip(
            self.history_guidance.hist_segments,
            self.history_guidance.hist_weights
        ):
            noise_levels_start, noise_levels_end = hist_segment.to_noise_levels(
                reduced_mask[self.hist_indices] == 2
            )
            hist_to_weights[noise_levels_start + (False,)] += weight
            hist_to_weights[
                noise_levels_end + (self.history_guidance.use_external_cond_guidance,)
            ] -= weight

        # 4. Convert to tensors
        hist_noise_levels = []
        cond_mask = []
        weights = []

        for hist_cond, weight in hist_to_weights.items():
            if weight == 0:
                continue
            hist_noise_levels.append(hist_cond[:-1])
            cond_mask.append(hist_cond[-1])
            weights.append(weight)

        self.hist_noise_levels = (
            torch.tensor(hist_noise_levels, device=self.device)
            * self.history_guidance.timesteps
            - 1
        ).long()
        self.cond_mask = torch.tensor(cond_mask, device=self.device)
        self.weights = torch.tensor(weights, device=self.device).float()
        self.num_hist = len(self.weights)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def _extend(self, a: torch.Tensor, x: torch.Tensor):
        """Extend tensor a to match dimensions of x"""
        return rearrange(a, "... -> ..." + " 1" * (x.ndim - a.ndim))

    def prepare(
        self,
        x: torch.Tensor,
        from_noise_levels: torch.Tensor,
        to_noise_levels: torch.Tensor,
        replacement_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        replacement_only: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare input for sampling with history guidance.

        Args:
            x: (batch_size, seq_len, ...) Input tensor
            from_noise_levels: (batch_size, seq_len) Current noise levels
            to_noise_levels: (batch_size, seq_len) Target noise levels
            replacement_fn: Function to add noise to clean tokens
            replacement_only: If True, only replace without modifying noise levels

        Returns:
            Tuple of (x, from_noise_levels, to_noise_levels, cond_mask)
        """
        b, g, h = x.size(0), self.num_gen, self.num_hist

        # Repeat for NFE
        x, from_noise_levels, to_noise_levels, mask = map(
            lambda y: repeat(y, "b t ...-> b h t ...", h=h).clone(),
            (x, from_noise_levels, to_noise_levels, self.mask),
        )

        # 1. Modify noise levels of history tokens
        if not replacement_only:
            from_noise_levels[:, :, self.hist_indices] = self.hist_noise_levels
            to_noise_levels[:, :, self.hist_indices] = self.hist_noise_levels

        # 2. Replace clean history tokens with noisy versions
        replace_mask = torch.logical_and(from_noise_levels >= 0, mask >= 1)
        x = torch.where(
            self._extend(replace_mask, x),
            rearrange(
                replacement_fn(
                    rearrange(x, "b h t ... -> (b h) t ..."),
                    rearrange(from_noise_levels, "b h t -> (b h) t"),
                ),
                "(b h) t ... -> b h t ...",
                h=h,
            ),
            x,
        )

        # Repeat for gen segments
        x, from_noise_levels, to_noise_levels, mask = map(
            lambda y: repeat(y, "b h t ... -> (b h) g t ...", g=g).clone(),
            (x, from_noise_levels, to_noise_levels, mask),
        )

        # 3. Modify noise levels of excluded generated tokens
        self.gen_but_excluded_mask = torch.logical_and(~self.gen_mask, mask == 0)
        from_noise_levels, to_noise_levels = map(
            lambda y: torch.where(
                self.gen_but_excluded_mask, self.history_guidance.timesteps - 1, y
            ),
            (from_noise_levels, to_noise_levels),
        )

        # 4. Replace excluded tokens with noise
        x = torch.where(
            self._extend(self.gen_but_excluded_mask, x),
            torch.randn_like(x),
            x,
        )

        # Flatten for model input
        x, from_noise_levels, to_noise_levels = map(
            lambda y: rearrange(y, "(b h) g t ... -> (b h g) t ...", h=h),
            (x, from_noise_levels, to_noise_levels),
        )

        return (
            x,
            from_noise_levels,
            to_noise_levels,
            repeat(self.cond_mask, "h -> (b h g)", b=b, g=g).clone(),
        )

    def compose(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compose predictions from different history segments.

        Args:
            x: (batch_size * num_hist * num_gen, seq_len, ...) Predictions

        Returns:
            (batch_size, seq_len, ...) Composed prediction
        """
        x = rearrange(
            x, "(b h g) t ... -> (b h) g t ...", h=self.num_hist, g=self.num_gen
        )

        # 1. Zero out excluded gen tokens
        x = torch.where(
            self._extend(self.gen_but_excluded_mask, x),
            torch.zeros_like(x),
            x,
        )

        x = rearrange(x, "(b h) g t ... -> b h g t ...", h=self.num_hist)

        # 2. Weighted sum
        x = einsum(x, self.weights, "b h g t ..., h -> b g t ...")

        # 3. Average across gen segments
        x = reduce(x, "b g t ... -> b t ...", "sum")
        gen_mask_sum = rearrange(
            reduce(self.gen_mask.long(), "g t -> t", "sum").clamp(min=1),
            "t -> t" + " 1" * (x.ndim - 2),
        )
        x /= gen_mask_sum

        return x


class HistoryGuidance:
    """
    Main class for configuring history guidance.

    This implements the core innovation of Diffusion Forcing:
    composing multiple conditional scores with different history subsets.
    """

    def __init__(
        self,
        hist_segments: List[HistorySegment],
        hist_weights: List[float],
        gen_segments: Optional[List[List[int] | ALLType]] = None,
        timesteps: int = 1000,
        use_external_cond_guidance: bool = False,
    ):
        """
        Args:
            hist_segments: List of history segments to compose
            hist_weights: Weight for each segment
            gen_segments: Which generated tokens to predict
            timesteps: Total diffusion timesteps
            use_external_cond_guidance: Use guidance for external conditions
        """
        self.hist_segments = hist_segments
        self.hist_weights = hist_weights
        self.gen_segments = gen_segments if gen_segments is not None else [ALL]
        self.timesteps = timesteps
        self.use_external_cond_guidance = use_external_cond_guidance

        assert len(hist_segments) == len(hist_weights), \
            "hist_segments and hist_weights must have same length"
        assert len(self.gen_segments) > 0, "Need at least one gen_segment"

    def __call__(self, mask: torch.Tensor) -> HistoryGuidanceManager:
        """
        Create guidance manager for this sampling step.

        Args:
            mask: (batch_size, seq_len) Token states
                  0=to generate, 1=ground truth, 2=generated, -1=padding
        """
        return HistoryGuidanceManager(self, mask)

    # ===== Predefined guidance schemes =====

    @classmethod
    def conditional(cls, timesteps: int = 1000) -> "HistoryGuidance":
        """Standard conditional sampling (use full history)"""
        return cls(
            hist_segments=[HistorySegment.full()],
            hist_weights=[1],
            timesteps=timesteps,
            use_external_cond_guidance=False,
        )

    @classmethod
    def vanilla(
        cls,
        guidance_scale: float,
        timesteps: int = 1000,
        use_external_cond_guidance: bool = True,
    ) -> "HistoryGuidance":
        """Vanilla History Guidance (HG-v)"""
        return cls(
            hist_segments=[HistorySegment.full()],
            hist_weights=[guidance_scale],
            timesteps=timesteps,
            use_external_cond_guidance=use_external_cond_guidance,
        )

    @classmethod
    def stabilized_vanilla(
        cls,
        guidance_scale: float,
        stabilization_level: float,
        timesteps: int = 1000,
        use_external_cond_guidance: bool = True,
    ) -> "HistoryGuidance":
        """HG-v with stabilization for generated frames"""
        return cls(
            hist_segments=[
                HistorySegment(
                    time_indices=ALL,
                    freq_ranges=[ALL],
                    freq_ranges_if_generated=[(stabilization_level, 1.0)],
                )
            ],
            hist_weights=[guidance_scale],
            timesteps=timesteps,
            use_external_cond_guidance=use_external_cond_guidance,
        )
