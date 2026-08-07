/*
 * Copyright (C) 2026 Eamon Sippy
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Module: config.h
 * Description: Centralized hyperparameter and runtime configuration parameters
 * for the  training loop, and inference pipeline.
 */
//---------------------------------------------------------
//// Hyperparameters
#pragma once
#include <string>
static const std::string DEFAULT_CLEANED_PATH = "data/input.txt";
static const std::string DATA_PATH_ENV_VAR = "GPT_DATA_PATH";
static const std::string BEST_MODEL_PATH = "best_model.bin";
static const std::string MODEL_PATH_ENV_VAR = "GPT_MODEL_PATH";
static const int N_EMBD = 128;
static const int N_HEAD = 2;
static const int N_LAYER = 4;
static const float DROPOUT = 0.05f;
static const int BPE_VOCAB_SIZE = 2056;
static const unsigned int SEED = 1337;
static const double TRAIN_SPLIT = 0.9;
static const int BATCH_SIZE = 32;
static const int BLOCK_SIZE = 64;
static const int MAX_ITERS = 5000;
static const int EVAL_INTERVAL = 200;
static const float LEARNING_RATE = 5e-4f;
static const int EVAL_ITERS = 25;
static const int DEFAULT_CHAT_TOKENS = 10;
static const int DEFAULT_REP_PENALTY = 10;
static const int DEFAULT_REP_WINDOW = 10;