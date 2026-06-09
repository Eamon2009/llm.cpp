#pragma once

#include "config/config.h"
#include <vector>
#include <string>

// Holds all parameters that control token sampling at inference time.
// Fields can be set from CLI flags or left at config.h defaults.
struct SamplerParams
{
    float       rep_penalty;   // values above 1.0 reduce repeated tokens
    int         rep_window;    // how many recent tokens to scan for repeats
    std::string system_prompt; // text prepended before every user turn in chat

    SamplerParams()
        : rep_penalty(DEFAULT_REP_PENALTY),
          rep_window(DEFAULT_REP_WINDOW),
          system_prompt("") {}
};

// Adjust raw logits to make recently seen tokens less likely to appear again.
// Tokens found in the recent context window have their logit reduced:
//   positive logits are divided by rep_penalty
//   negative logits are multiplied by rep_penalty
// This matches the formula used in llama.cpp repetition penalty.
// Call this on the last time step logits before softmax.
inline void apply_rep_penalty(std::vector<float>     &logits,
                               const std::vector<int> &context,
                               const SamplerParams    &params)
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
