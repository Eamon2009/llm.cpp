#pragma once
#include <string>
static const std::string DEFAULT_CLEANED_PATH = "data/input.txt";
static const std::string DATA_PATH_ENV_VAR = "GPT_DATA_PATH";
static const unsigned int SEED = 1337;
static const double TRAIN_SPLIT = 0.9; // 90 % train, 10 % val
static const int BATCH_SIZE = 4;
static const int BLOCK_SIZE = 64; // context length
static const int MAX_ITERS = 10000;
static const int EVAL_INTERVAL = 20;
static const float LEARNING_RATE = 3e-4f;
static const int EVAL_ITERS = 1;
static const int N_EMBD = 128;
static const int N_HEAD = 4;
static const int N_LAYER = 4;
static const float DROPOUT = 0.2f; // applied during training only
static const std::string BEST_MODEL_PATH = "best_model.bin";
static const std::string MODEL_PATH_ENV_VAR = "GPT_MODEL_PATH";
