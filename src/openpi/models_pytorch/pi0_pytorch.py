import dataclasses
import logging
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def _last_valid_indices(mask: torch.Tensor) -> torch.Tensor:
    """Return the index of the last valid (True) token for each row in a [b, s] mask."""
    seq_len = mask.shape[1]
    indices = torch.arange(seq_len, device=mask.device)[None, :].expand_as(mask)
    return torch.where(mask.bool(), indices, torch.full_like(indices, -1)).max(dim=1).values


def _scatter_sequence(
    base: torch.Tensor,
    positions: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Write a masked token block ``values`` into a padded batch sequence ``base``.

    ``positions`` (int, [b, s]) gives the target column for every element of ``values`` ([b, s]).
    Only entries where ``mask`` is True and the position lies inside ``base`` are written. This mirrors
    the JAX ``_scatter_sequence`` helper (one-hot scatter) used by the reference pi0.5 subtask code.
    On position collisions (only possible when the sequence overflows ``max_len`` and gets clamped) the
    last write wins, which matches the truncation semantics of the JAX version.
    """
    max_len = base.shape[-1]
    positions = positions.long()
    valid = mask.bool() & (positions >= 0) & (positions < max_len)
    clipped = positions.clamp(0, max_len - 1)

    put_mask = torch.zeros_like(base, dtype=torch.bool)
    put_mask.scatter_(1, clipped, valid)

    if base.dtype == torch.bool:
        src = (values.bool() & valid).to(torch.int64)
        tmp = torch.zeros_like(base, dtype=torch.int64)
        tmp.scatter_(1, clipped, src)
        tmp = tmp.bool()
    else:
        src = values.to(base.dtype) * valid.to(base.dtype)
        tmp = torch.zeros_like(base)
        tmp.scatter_(1, clipped, src)
    return torch.where(put_mask, tmp, base)


class PI0Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05
        # Optional pi0.5 two-stage (subtask) prediction, ported from the JAX implementation.
        self.train_subtask_prediction = config.train_subtask_prediction
        self.sample_subtask_prediction = config.sample_subtask_prediction
        self.flow_loss_weight = config.flow_loss_weight
        self.subtask_loss_weight = config.subtask_loss_weight
        self.max_subtask_len = config.max_subtask_len
        self.subtask_temperature = config.subtask_temperature
        self.subtask_eos_token = config.subtask_eos_token
        if (config.train_subtask_prediction or config.sample_subtask_prediction) and not config.pi05:
            raise ValueError("Subtask prediction is only supported for pi0.5 models.")

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(config.action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.action_dim)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(config.action_dim, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")
        # The staged subtask sampler uses Python-side autoregressive loops with data-dependent control
        # flow, so we skip torch.compile when it is enabled to avoid graph breaks / recompilation.
        if config.pytorch_compile_mode is not None and not self.sample_subtask_prediction:
            self.sample_actions = torch.compile(self.sample_actions, mode=config.pytorch_compile_mode)

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks, lang_ar_mask=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.

        ``lang_ar_mask`` (optional int/bool [b, num_lang]) sets the autoregressive attention flag for the
        language tokens. It is used by pi0.5 subtask prediction, where the generated subtask (and the
        action cue) must attend causally. When ``None`` the language tokens use full (bidirectional)
        attention, matching the default pi0/pi0.5 behavior.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Create attention masks so that image tokens attend to each other
            att_masks.append(torch.zeros(bsize, num_img_embs, dtype=torch.bool, device=img_emb.device))

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        bsize, num_lang_embs = lang_emb.shape[:2]
        if lang_ar_mask is None:
            # full attention between image and language inputs
            att_masks.append(torch.zeros(bsize, num_lang_embs, dtype=torch.bool, device=lang_emb.device))
        else:
            att_masks.append(lang_ar_mask.to(dtype=torch.bool, device=lang_emb.device))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.cat(att_masks, dim=1)

        return embs, pad_masks, att_masks

    def _deembed(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project hidden states back to vocab logits using the (tied) input embedding matrix.

        Mirrors the JAX ``Embedder.decode`` (a plain ``hidden @ embedding.T`` with no scaling/softcapping).
        """
        weight = self.paligemma_with_expert.paligemma.language_model.embed_tokens.weight
        return F.linear(hidden.to(weight.dtype), weight)

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        proc = _preprocessing.preprocess_observation_pytorch(observation, train=True)
        images = list(proc.images.values())
        img_masks = list(proc.image_masks.values())
        lang_tokens = proc.tokenized_prompt
        lang_masks = proc.tokenized_prompt_mask
        state = proc.state
        # When training subtask prediction, the tokenized prompt already contains the supervised subtask
        # (+ action cue) tokens and needs a causal AR mask over them.
        lang_ar_mask = proc.token_ar_mask if self.train_subtask_prediction else None

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, lang_ar_mask=lang_ar_mask
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Apply gradient checkpointing if enabled
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (prefix_out, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return prefix_out, suffix_out

        prefix_out, suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # Apply gradient checkpointing to final action projection if enabled
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        action_loss = F.mse_loss(u_t, v_t, reduction="none")
        action_loss = self.flow_loss_weight * self._mask_action_loss(action_loss, proc.action_loss_mask)
        if not self.train_subtask_prediction:
            return action_loss

        # Joint loss: flow-matching action loss + cross-entropy on the supervised subtask tokens.
        subtask_loss = self._compute_subtask_loss(prefix_out, lang_tokens, proc.token_loss_mask)
        return action_loss + self.subtask_loss_weight * subtask_loss[:, None, None]

    def _mask_action_loss(self, action_loss, action_loss_mask):
        if action_loss_mask is None:
            return action_loss
        mask = action_loss_mask.to(device=action_loss.device, dtype=action_loss.dtype)
        valid_count = mask.sum(dim=-1, keepdim=True)
        mask = torch.where(valid_count > 0, mask, torch.ones_like(mask))
        valid_count = mask.sum(dim=-1, keepdim=True)
        scale = action_loss.shape[-2] / valid_count
        return action_loss * mask.unsqueeze(-1) * scale.unsqueeze(-1)

    def _compute_subtask_loss(self, prefix_out, lang_tokens, token_loss_mask) -> torch.Tensor:
        """Cross-entropy over the (shifted) language tokens, masked to the subtask span. Returns [b]."""
        if token_loss_mask is None:
            raise ValueError("token_loss_mask is required when train_subtask_prediction is enabled.")
        text_len = lang_tokens.shape[1]
        # prefix_out = [image tokens ... language tokens]; take the language hidden states and shift by one.
        text_hidden = prefix_out[:, -text_len:-1]
        logits = self._deembed(text_hidden).float()
        logp = F.log_softmax(logits, dim=-1)
        targets = lang_tokens[:, 1:].long()
        target_logp = torch.gather(logp, -1, targets.unsqueeze(-1)).squeeze(-1)
        loss_mask = token_loss_mask[:, 1:].to(target_logp.dtype)
        return -(target_logp * loss_mask).sum(-1) / loss_mask.sum(-1).clamp_min(1.0)

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        proc = _preprocessing.preprocess_observation_pytorch(observation, train=False)
        images = list(proc.images.values())
        img_masks = list(proc.image_masks.values())
        lang_tokens = proc.tokenized_prompt
        lang_masks = proc.tokenized_prompt_mask
        state = proc.state
        # token_ar_mask is None for standard flow-matching inference (full attention); for the
        # subtask-conditioned action stage it marks the generated subtask/action-cue tokens as causal.
        lang_ar_mask = getattr(proc, "token_ar_mask", None)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, lang_ar_mask=lang_ar_mask
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )

            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)

    @torch.no_grad()
    def sample_actions_with_subtask(self, device, observation, noise=None, num_steps=10):
        """Two-stage pi0.5 inference: first autoregressively decode a subtask, then flow-match actions.

        Returns a dict with ``actions`` (and, when subtask prediction is enabled, ``subtask_tokens`` and
        ``subtask_token_mask``). Mirrors the JAX ``Pi0.sample_actions_with_subtask``.
        """
        if not self.sample_subtask_prediction:
            return {"actions": self.sample_actions(device, observation, noise=noise, num_steps=num_steps)}
        if not self.pi05:
            raise ValueError("Subtask prediction is only supported for pi0.5 models.")

        subtask_tokens, subtask_token_mask = self._sample_subtask_tokens(device, observation)
        action_observation = self._with_generated_subtask_prompt(observation, subtask_tokens, subtask_token_mask)
        actions = self.sample_actions(device, action_observation, noise=noise, num_steps=num_steps)
        return {
            "actions": actions,
            "subtask_tokens": subtask_tokens,
            "subtask_token_mask": subtask_token_mask,
        }

    @torch.no_grad()
    def _sample_subtask_tokens(self, device, observation):
        """Greedily (or with temperature) decode subtask tokens one at a time from the prefix.

        Images are embedded once and only the (growing) language prompt is re-embedded each step, which is
        a faster equivalent of the JAX reference that rebuilds the whole prefix per step.
        """
        if observation.tokenized_prompt is None or observation.tokenized_prompt_mask is None:
            raise ValueError("Tokenized prompt (and mask) are required for subtask prediction.")

        proc = _preprocessing.preprocess_observation_pytorch(observation, train=False)
        images = list(proc.images.values())
        img_masks = list(proc.image_masks.values())

        # Embed images once and cache them.
        img_embs = []
        img_pad_masks = []
        for img, img_mask in zip(images, img_masks, strict=True):
            emb = self.paligemma_with_expert.embed_image(img)
            bsize, num_img_embs = emb.shape[:2]
            img_embs.append(emb)
            img_pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

        bsize = observation.state.shape[0]
        out_tokens = torch.zeros(bsize, self.max_subtask_len, dtype=torch.int32, device=device)
        out_mask = torch.zeros(bsize, self.max_subtask_len, dtype=torch.bool, device=device)
        done = torch.zeros(bsize, dtype=torch.bool, device=device)

        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        for step_idx in range(self.max_subtask_len):
            cur = self._with_generated_subtask_prompt(observation, out_tokens, out_mask, include_action_suffix=False)
            lang_emb = self.paligemma_with_expert.embed_language_tokens(cur.tokenized_prompt)
            lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])

            embs = torch.cat([*img_embs, lang_emb], dim=1)
            pad_masks = torch.cat([*img_pad_masks, cur.tokenized_prompt_mask], dim=1)
            att_masks = torch.cat(
                [torch.zeros_like(m, dtype=torch.bool) for m in img_pad_masks]
                + [cur.token_ar_mask.to(dtype=torch.bool)],
                dim=1,
            )

            att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
            position_ids = torch.cumsum(pad_masks, dim=1) - 1
            att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

            (prefix_out, _), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[embs, None],
                use_cache=False,
            )

            last_indices = _last_valid_indices(pad_masks)
            last_hidden = prefix_out[torch.arange(bsize, device=device), last_indices][:, None, :]
            logits = self._deembed(last_hidden)[:, 0].float()

            if self.subtask_temperature > 0.0:
                probs = F.softmax(logits / self.subtask_temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)[:, 0]
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = torch.where(done, torch.zeros_like(next_token), next_token).to(torch.int32)

            positions = torch.full((bsize, 1), step_idx, dtype=torch.long, device=device)
            write = (~done)[:, None]
            out_tokens = _scatter_sequence(out_tokens, positions, next_token[:, None], write)
            out_mask = _scatter_sequence(out_mask, positions, (~done)[:, None], write)
            done = done | (next_token == self.subtask_eos_token)
            if bool(done.all()):
                break

        return out_tokens, out_mask

    def _with_generated_subtask_prompt(
        self, observation, subtask_tokens, subtask_token_mask, *, include_action_suffix=True
    ):
        """Splice generated subtask tokens (and optionally the action cue) into the tokenized prompt."""
        if observation.tokenized_prompt is None or observation.tokenized_prompt_mask is None:
            raise ValueError("Tokenized prompt (and mask) are required for subtask prediction.")

        prompt_tokens = observation.tokenized_prompt
        prompt_mask = observation.tokenized_prompt_mask
        if observation.token_ar_mask is None:
            prompt_ar_mask = torch.zeros_like(prompt_mask, dtype=torch.bool)
        else:
            prompt_ar_mask = observation.token_ar_mask.to(dtype=torch.bool)

        prompt_len = prompt_mask.sum(dim=-1)
        arange = torch.arange(subtask_tokens.shape[1], device=prompt_tokens.device)
        subtask_positions = prompt_len[:, None] + arange[None, :]

        tokens = _scatter_sequence(prompt_tokens, subtask_positions, subtask_tokens, subtask_token_mask)
        token_mask = _scatter_sequence(prompt_mask, subtask_positions, subtask_token_mask, subtask_token_mask)
        token_ar_mask = _scatter_sequence(
            prompt_ar_mask, subtask_positions, torch.ones_like(subtask_token_mask), subtask_token_mask
        )

        if include_action_suffix:
            if observation.tokenized_action_suffix is None or observation.tokenized_action_suffix_mask is None:
                raise ValueError("Tokenized action suffix (and mask) are required for subtask prediction.")
            action_suffix = observation.tokenized_action_suffix
            action_suffix_mask = observation.tokenized_action_suffix_mask
            subtask_len = subtask_token_mask.sum(dim=-1)
            arange_suffix = torch.arange(action_suffix.shape[1], device=tokens.device)
            suffix_positions = prompt_len[:, None] + subtask_len[:, None] + arange_suffix[None, :]

            tokens = _scatter_sequence(tokens, suffix_positions, action_suffix, action_suffix_mask)
            token_mask = _scatter_sequence(token_mask, suffix_positions, action_suffix_mask, action_suffix_mask)
            token_ar_mask = _scatter_sequence(
                token_ar_mask, suffix_positions, torch.ones_like(action_suffix_mask), action_suffix_mask
            )

        return dataclasses.replace(
            observation,
            tokenized_prompt=tokens,
            tokenized_prompt_mask=token_mask,
            token_ar_mask=token_ar_mask.to(torch.int32),
            token_loss_mask=None,
            tokenized_action_suffix=None,
            tokenized_action_suffix_mask=None,
        )
