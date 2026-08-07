/*
 * SPDX-License-Identifier: GPL-3.0-only
 * Copyright (C) 2026 Eamon Sippy
 */

#pragma once

#include "config/config.h"

#include <string>
#include <vector>

/**
 * @brief Inference-time sampling hyperparameters.
 */
struct SamplerParams
{
      float rep_penalty;         // >1.0 reduces repeated tokens
      int rep_window;            // Recent token window for repetition check
      std::string system_prompt; // Prepended to each user turn in chat mode

      /**
       * @brief Construct with config.h defaults.
       */
      SamplerParams()
          : rep_penalty(DEFAULT_REP_PENALTY), rep_window(DEFAULT_REP_WINDOW), system_prompt("")
      {
      }
};

/**
 * @brief Apply repetition penalty to logits.
 *
 * Positive logits divided by rep_penalty; negative logits multiplied.
 * Matches llama.cpp formula. Call before softmax.
 *
 * @param logits [vocab_size]. In-place modified.
 * @param context Recent token history.
 * @param params  SamplerParams with rep_penalty and rep_window.
 */
inline void apply_rep_penalty(std::vector<float> &logits, const std::vector<int> &context,
                              const SamplerParams &params)
{
      if (params.rep_penalty <= 1.0f || context.empty())
            return;

      int window_start = (int)context.size() - params.rep_window;
      if (window_start < 0)
            window_start = 0;

      for (int i = window_start; i < (int)context.size(); ++i)
      {
            int tok = context[i];
            if (tok < 0 || tok >= (int)logits.size())
                  continue;

            float &l = logits[tok];
            if (l > 0.0f)
                  l /= params.rep_penalty;
            else
                  l *= params.rep_penalty;
      }
}