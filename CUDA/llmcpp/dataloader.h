#ifndef DATALOADER_H
#define DATALOADER_H

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <assert.h>
#include <string.h>
#include "utils.h"
#include "rand.h"
#ifndef _WIN32
#include <glob.h>
#endif
#define HEADER_SIZE 256

typedef struct
{

    int process_rank;
    int num_processes;

    size_t B;
    size_t T;
    size_t num_tokens;
    size_t shard_num_samples;

    glob_t glob_result;
    size_t current_shard_idx;
    size_t current_sample_idx;

    FILE *tokens_file;

    uint16_t *buffer;
    int *inputs;
    int *targets;

    mt19937_state shuffle_rng;
    int should_shuffle;
    int *shard_indices;
    int *intra_shard_indices;

    size_t total_batch_size_bytes;
    size_t local_batch_offset_bytes;
    size_t header_bytes;
    int64_t file_size_bytes;
} DataLoader;

int64_t dataloader_load_shard_(DataLoader *loader, int shard_index)
{
    if (loader->should_shuffle)
    {
        shard_index = loader->shard_indices[shard_index];
    }

    const char *filename = loader->glob_result.gl_pathv[shard_index];

    if (loader->tokens_file != NULL)
    {
        fcloseCheck(loader->tokens_file);
    }
    loader->tokens_file = fopenCheck(filename, "rb");

    int header[HEADER_SIZE];
    freadCheck(header, sizeof(int), HEADER_SIZE, loader->tokens_file);
    if (header[0] != 20240520)
    {

        printf("---> HINT: Are you passing in a correct file?\n");
        printf("---> HINT: The data encoding may have changed, re-run data prepro or refer again to README.\n");
        exit(EXIT_FAILURE);
    }
    if (header[1] != 1)
    {
        printf("Bad version in data file\n");
        exit(EXIT_FAILURE);
    }
    int64_t ntok = header[2]; //
    assert(ntok > 0);
    fseekCheck(loader->tokens_file, 0, SEEK_END);
    loader->file_size_bytes = ftell(loader->tokens_file);
    fseekCheck(loader->tokens_file, 0, SEEK_SET);
    int64_t expected_file_size = HEADER_SIZE * sizeof(int) + ntok * sizeof(uint16_t);
    if (loader->file_size_bytes != expected_file_size)
    {
        printf("Error: file size is not as expected\n");
        exit(EXIT_FAILURE);
    }

    loader->shard_num_samples = (ntok * sizeof(uint16_t) - sizeof(uint16_t)) / loader->total_batch_size_bytes;
    return ntok;
}

void prepare_intra_shard_indices_(DataLoader *loader)
{

    if (loader->intra_shard_indices != NULL)
    {

        free(loader->intra_shard_indices);
    }
    loader->intra_shard_indices = (int *)mallocCheck(loader->shard_num_samples * sizeof(int));
    init_identity_permutation(loader->intra_shard_indices, (int)loader->shard_num_samples);
    random_permutation(loader->intra_shard_indices, (int)loader->shard_num_samples, &loader->shuffle_rng);
}

void dataloader_reset(DataLoader *loader)
{
    loader->current_shard_idx = 0;
    loader->current_sample_idx = 0;

    if (loader->should_shuffle)
    {
        random_permutation(loader->shard_indices, (int)loader->glob_result.gl_pathc, &loader->shuffle_rng);
    }

    dataloader_load_shard_(loader, (int)loader->current_shard_idx);

    if (loader->should_shuffle)
    {
        prepare_intra_shard_indices_(loader);
    }
}

void dataloader_advance_(DataLoader *loader)
{
    if (loader->current_shard_idx == loader->glob_result.gl_pathc - 1)
    {

        dataloader_reset(loader);
        return;
    }

    loader->current_shard_idx = (loader->current_shard_idx + 1) % loader->glob_result.gl_pathc;
    loader->current_sample_idx = 0;
    dataloader_load_shard_(loader, (int)loader->current_shard_idx);

    if (loader->should_shuffle)
    {
        prepare_intra_shard_indices_(loader);
    }
}

void dataloader_init(DataLoader *loader,
                     const char *filename_pattern,
                     size_t B,
                     size_t T,
                     int process_rank,
                     int num_processes,
                     int should_shuffle)
{
    loader->process_rank = process_rank;
    loader->num_processes = num_processes;
    loader->B = B;
    loader->T = T;
    loader->tokens_file = NULL;
    loader->should_shuffle = should_shuffle;
    loader->header_bytes = HEADER_SIZE * sizeof(int);
    loader->total_batch_size_bytes = ((loader->num_processes * (loader->B * loader->T)) * sizeof(uint16_t));
    loader->local_batch_offset_bytes = loader->process_rank * loader->B * loader->T * sizeof(uint16_t);

    int glob_status = glob(filename_pattern, 0, NULL, &loader->glob_result);
    if (glob_status != 0)
    {
        printf("Error: failed to glob pattern: %s\n", filename_pattern);
        exit(EXIT_FAILURE);
    }
    if (loader->glob_result.gl_pathc == 0)
    {
        printf("Error: no files found matching the pattern: %s\n", filename_pattern);
        exit(EXIT_FAILURE);
    }

    if (should_shuffle)
    {
        mt19937_state shuffle_rng;
        manual_seed(&shuffle_rng, 42 + process_rank);
        loader->shuffle_rng = shuffle_rng;
        loader->shard_indices = (int *)mallocCheck(loader->glob_result.gl_pathc * sizeof(int));
        init_identity_permutation(loader->shard_indices, (int)loader->glob_result.gl_pathc);
        loader->intra_shard_indices = NULL;
    }

    int64_t ntok_total = 0;
    for (int shard_index = 0; shard_index < loader->glob_result.gl_pathc; shard_index++)
    {
        int64_t shard_ntok = dataloader_load_shard_(loader, shard_index);

        assert(shard_ntok >= (int64_t)(num_processes * B * T + 1));
        ntok_total += shard_ntok;
    }

    loader->buffer = (uint16_t *)mallocCheck((B * T + 1) * sizeof(uint16_t));
    loader->inputs = (int *)mallocCheck(B * T * sizeof(int));
    loader->targets = (int *)mallocCheck(B * T * sizeof(int));
    loader->num_tokens = ntok_total;

    dataloader_reset(loader);
}

void dataloader_load_batch(DataLoader *loader)
{
    assert(!loader->should_shuffle || (loader->should_shuffle && loader->intra_shard_indices != NULL));
    assert(loader->current_sample_idx < loader->shard_num_samples);
    size_t idx = loader->should_shuffle ? loader->intra_shard_indices[loader->current_sample_idx] : loader->current_sample_idx;
    size_t global_batch_offset_bytes = idx * loader->total_batch_size_bytes;
    int64_t current_offset = loader->header_bytes + global_batch_offset_bytes + loader->local_batch_offset_bytes;

    size_t B = loader->B;
    size_t T = loader->T;

    fseekCheck(loader->tokens_file, (int)current_offset, SEEK_SET);
    freadCheck(loader->buffer, sizeof(uint16_t), B * T + 1, loader->tokens_file);

    for (int i = 0; i < B * T; i++)
    {
        loader->inputs[i] = (int)loader->buffer[i];
        loader->targets[i] = (int)loader->buffer[i + 1];
    }
}

void dataloader_next_batch(DataLoader *loader)
{

    if (loader->current_sample_idx >= loader->shard_num_samples)
    {
        dataloader_advance_(loader);
    }
    dataloader_load_batch(loader);
    loader->current_sample_idx += 1;
}

void dataloader_resume(DataLoader *loader, size_t current_shard_idx, size_t current_sample_idx)
{

    loader->current_shard_idx = current_shard_idx;
    loader->current_sample_idx = current_sample_idx;
    dataloader_load_shard_(loader, (int)loader->current_shard_idx);
}

void dataloader_free(DataLoader *loader)
{
    free(loader->buffer);
    free(loader->inputs);
    free(loader->targets);
    if (loader->should_shuffle)
    {
        free(loader->shard_indices);
        free(loader->intra_shard_indices);
    }
    fcloseCheck(loader->tokens_file);
    globfree(&loader->glob_result);
}

#define ASSUMED_NUM_COMPLETIONS 4
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

typedef struct
{

    int process_rank;
    int num_processes;

    size_t B;
    size_t T;
    FILE *eval_file;
    uint16_t *buffer;
    int num_examples;
    int num_batches;
    int start_example_index;
    int end_example_index;
    int current_example_index;
    int *inputs;
    int *targets;
    char *mask;
    int *label;
    int num_completions;
} EvalLoader;

void evalloader_reset(EvalLoader *loader)
{
    int examples_per_process = CEIL_DIV(loader->num_examples, loader->num_processes);
    int can_fit_examples = (int)(loader->B / ASSUMED_NUM_COMPLETIONS);
    if (can_fit_examples == 0)
    {

        printf("HellaSwag EvalLoader: batch size %zu is < %d\n", loader->B, ASSUMED_NUM_COMPLETIONS);
        printf("---> HINT: Disable HellaSwag eval with -h 0, or increase batch size with -b\n");
        exit(EXIT_FAILURE);
    }
    loader->num_batches = CEIL_DIV(examples_per_process, can_fit_examples);

    loader->start_example_index = examples_per_process * loader->process_rank;
    loader->end_example_index = examples_per_process * (loader->process_rank + 1);

    if (loader->end_example_index > loader->num_examples)
    {
        loader->end_example_index = loader->num_examples;
    }

    int64_t header_bytes = HEADER_SIZE * sizeof(int);
    fseekCheck(loader->eval_file, (int)header_bytes, SEEK_SET);
    for (int i = 0; i < loader->start_example_index; i++)
    {
        uint16_t example_header[3];
        // read 3 uint16_t values: <START_EXAMPLE>, <EXAMPLE_BYTES>, <EXAMPLE_INDEX>
        freadCheck(&example_header[0], sizeof(uint16_t), 3, loader->eval_file);
        // validate the <START_EXAMPLE> delimiter
        assert(example_header[0] == 65535); // <START_EXAMPLE> delimiter
        // validate the <EXAMPLE_INDEX>
        assert(example_header[2] == i); // <EXAMPLE_INDEX> should match the loop index
        // skip to the next example, keeping in mind that we already read the header
        size_t remaining_bytes = example_header[1] - sizeof(uint16_t) * 3;
        assert(remaining_bytes > 0); // we expect some bytes in the example
        fseekCheck(loader->eval_file, (int)remaining_bytes, SEEK_CUR);
    }
    // now we are at the start of the example we want to start at, pointing at <START_EXAMPLE>
    loader->current_example_index = loader->start_example_index;
}

void evalloader_init(EvalLoader *loader,
                     const char *filename,
                     size_t B,
                     size_t T,
                     int process_rank,
                     int num_processes)
{
    loader->process_rank = process_rank;
    loader->num_processes = num_processes;
    loader->B = B;
    loader->T = T;

    // open the file and validate the header
    loader->eval_file = fopenCheck(filename, "rb");
    // validate the header
    int header[HEADER_SIZE];
    freadCheck(header, sizeof(int), HEADER_SIZE, loader->eval_file);
    if (header[0] != 20240522)
    {
        printf("Bad magic in eval file\n");
        exit(EXIT_FAILURE);
    }
    if (header[1] != 1)
    {
        printf("Bad version in data file\n");
        exit(EXIT_FAILURE);
    }
    loader->num_examples = header[2];              // number of examples in the file
    assert(loader->num_examples >= num_processes); // avoid headaches for now
    size_t longest_example_bytes = header[3];      // longest example in the file
    // basic sensibility check we could relax later. but roughly each example
    // contains the prompt (or "context") and 4 completions, all of these have to be
    // up to T tokens, and their tokens are uint16_t (so 2 bytes/token).
    // There's a few more things in each example but they are minor.
    // So longest example should be roughly this. Just trying to make sure it's sensible.
    assert(longest_example_bytes > 0 && longest_example_bytes < (1 + ASSUMED_NUM_COMPLETIONS) * T * 2);

    // allocate all the space we'll need
    int can_fit_examples = (int)(B / ASSUMED_NUM_COMPLETIONS);
    loader->buffer = (uint16_t *)mallocCheck(longest_example_bytes);
    loader->inputs = (int *)calloc(B * T, sizeof(int));
    loader->targets = (int *)calloc(B * T, sizeof(int));
    loader->mask = (char *)mallocCheck(B * T * sizeof(char));
    loader->label = (int *)mallocCheck(can_fit_examples * sizeof(int));

    // reset the loader, to initialize it
    evalloader_reset(loader);
}

void evalloader_next_example_(EvalLoader *loader, int example_batch_index)
{
    size_t B = loader->B;
    size_t T = loader->T;
    int batch_dim_offset = example_batch_index * ASSUMED_NUM_COMPLETIONS;
    uint16_t example_header[3];
    freadCheck(&example_header[0], sizeof(uint16_t), 3, loader->eval_file);
    assert(example_header[0] == 65535);
    assert(example_header[2] == loader->current_example_index);
    assert(example_header[2] >= loader->start_example_index && example_header[2] < loader->end_example_index);

    size_t example_bytes = example_header[1] - sizeof(uint16_t) * 3;
    freadCheck(loader->buffer, sizeof(char), example_bytes, loader->eval_file);
    int label = (int)loader->buffer[0];
    int can_fit_examples = (int)(loader->B / ASSUMED_NUM_COMPLETIONS);
    assert(label >= 0 && label < ASSUMED_NUM_COMPLETIONS);
    assert(example_batch_index >= 0 && example_batch_index < can_fit_examples);
    loader->label[example_batch_index] = label;
    int num_completions = (int)loader->buffer[1];
    assert(num_completions == ASSUMED_NUM_COMPLETIONS);
    assert(batch_dim_offset + num_completions <= B);
    loader->num_completions = num_completions;

    int context_length = (int)loader->buffer[2];
    uint16_t *context_tokens_start = &loader->buffer[3];
    assert(context_length > 0 && context_length < T);
    for (int b = 0; b < num_completions; b++)
    {
        for (int i = 0; i < context_length; i++)
        {
            int boff = batch_dim_offset + b;
            int tok_cur = (int)context_tokens_start[i];
            loader->inputs[boff * T + i] = tok_cur;
        }
    }
    uint16_t *completions_iter = loader->buffer + 3 + context_length;
    for (int c = 0; c < num_completions; c++)
    {
        int coff = batch_dim_offset + c;
        int completion_length = (int)completions_iter[0];
        uint16_t *completion_tokens_start = completions_iter + 1;
        assert(completion_length > 0 && context_length + completion_length < T);
        for (int i = 0; i < completion_length; i++)
        {
            int tok_cur = (int)completion_tokens_start[i];

            loader->inputs[coff * T + context_length + i] = tok_cur;

            loader->targets[coff * T + context_length + i - 1] = tok_cur;

            loader->mask[coff * T + context_length + i - 1] = 1;
        }
        completions_iter += 1 + completion_length;
        loader->current_example_index += 1;
    }

    void evalloader_next_batch(EvalLoader * loader)
    {
        size_t B = loader->B;
        size_t T = loader->T;
        memset(loader->mask, 0, B * T * sizeof(char));
        int can_fit_examples = (int)(B / ASSUMED_NUM_COMPLETIONS);
        for (int i = 0; i < can_fit_examples; i++)
        {
            if (loader->current_example_index >= loader->end_example_index)
            {
                break;
            }
            evalloader_next_example_(loader, i);
        }
    }

    int evalloader_stat_losses(EvalLoader * loader, float *losses)
    {
        int correct = 0;
        size_t B = loader->B;
        size_t T = loader->T;
        int can_fit_examples = (int)(B / ASSUMED_NUM_COMPLETIONS);
        for (int i = 0; i < can_fit_examples; i++)
        {
            float min_loss = 0.0f;
            int min_loss_index = -1;
            char active = 0;
            for (int b = 0; b < ASSUMED_NUM_COMPLETIONS; b++)
            {
                int boff = i * ASSUMED_NUM_COMPLETIONS + b;
                float average_loss = 0.0f;
                int count = 0;
                for (int t = 0; t < T; t++)
                {
                    char mask = loader->mask[boff * T + t];
                    if (mask == 1)
                    {
                        active = 1;
                        average_loss += losses[boff * T + t];
                        count++;
                    }
                }
                if (count > 0)
                {
                    average_loss /= count;
                }
                if (b == 0 || average_loss < min_loss)
                {
                    min_loss = average_loss;
                    min_loss_index = b;
                }
            }
            if (active && (min_loss_index == loader->label[i]))
            {
                correct += 1;
            }
        }
        return correct;
    }

    void evalloader_free(EvalLoader * loader)
    {
        free(loader->buffer);
        free(loader->inputs);
        free(loader->targets);
        free(loader->mask);
        free(loader->label);
        fcloseCheck(loader->eval_file);
    }

#endif