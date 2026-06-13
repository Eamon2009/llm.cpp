#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Config
typedef struct
{
      int vocab_size;
      int block_size;
      int n_embd;
      int n_head;
      int n_layer;
} Config;

// Tensor helpers
typedef struct
{
      float *data;
      int dims[4];
      int ndim;
} Tensor;

static Tensor read_tensor(FILE *f)
{
      Tensor t = {0};
      fread(&t.ndim, sizeof(int), 1, f);
      int total = 1;
      for (int i = 0; i < t.ndim; i++)
      {
            fread(&t.dims[i], sizeof(int), 1, f);
            total *= t.dims[i];
      }
      t.data = (float *)malloc(total * sizeof(float));
      fread(t.data, sizeof(float), total, f);
      return t;
}

static float *alloc(int n)
{
      float *p = (float *)calloc(n, sizeof(float));
      if (!p)
      {
            fprintf(stderr, "OOM\n");
            exit(1);
      }
      return p;
}

//  Math
static void softmax(float *x, int n)
{
      float max = x[0];
      for (int i = 1; i < n; i++)
            if (x[i] > max)
                  max = x[i];
      float sum = 0.0f;
      for (int i = 0; i < n; i++)
      {
            x[i] = expf(x[i] - max);
            sum += x[i];
      }
      for (int i = 0; i < n; i++)
            x[i] /= sum;
}

static void layer_norm(float *out, float *x, float *w, float *b, int n)
{
      float mean = 0.0f, var = 0.0f;
      for (int i = 0; i < n; i++)
            mean += x[i];
      mean /= n;
      for (int i = 0; i < n; i++)
            var += (x[i] - mean) * (x[i] - mean);
      var = var / n + 1e-5f;
      float inv = 1.0f / sqrtf(var);
      for (int i = 0; i < n; i++)
            out[i] = (x[i] - mean) * inv * w[i] + b[i];
}

// matmul: out[M x N] = A[M x K] @ B[K x N] (B is stored row-major as [N x K])
static void matmul(float *out, float *A, float *B, int M, int K, int N)
{
      memset(out, 0, M * N * sizeof(float));
      for (int m = 0; m < M; m++)
            for (int k = 0; k < K; k++)
            {
                  float a = A[m * K + k];
                  for (int n = 0; n < N; n++)
                        out[m * N + n] += a * B[n * K + k];
            }
}

static void add_bias(float *x, float *b, int rows, int cols)
{
      for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                  x[i * cols + j] += b[j];
}

static void relu(float *x, int n)
{
      for (int i = 0; i < n; i++)
            if (x[i] < 0)
                  x[i] = 0;
}

static int sample(float *probs, int n)
{
      float r = (float)rand() / ((float)RAND_MAX + 1.0f);
      float cdf = 0.0f;
      for (int i = 0; i < n; i++)
      {
            cdf += probs[i];
            if (r < cdf)
                  return i;
      }
      return n - 1;
}

// Model weights
typedef struct
{
      Tensor tok_emb;
      Tensor pos_emb;
      // per layer
      Tensor *head_k; // [n_layer * n_head]
      Tensor *head_q;
      Tensor *head_v;
      Tensor *sa_proj_w;
      Tensor *sa_proj_b;
      Tensor *ff_w1;
      Tensor *ff_b1;
      Tensor *ff_w2;
      Tensor *ff_b2;
      Tensor *ln1_w;
      Tensor *ln1_b;
      Tensor *ln2_w;
      Tensor *ln2_b;
      Tensor ln_f_w;
      Tensor ln_f_b;
      Tensor lm_w;
      Tensor lm_b;
} Weights;

//  Signal handler
static volatile int running = 1;
void handle_sigint(int s)
{
      (void)s;
      printf("\n\n[Stopped by user]\n");
      running = 0;
}

// Forward pass
static void forward(float *logits, int *tokens, int T, Config *cfg, Weights *W)
{
      int C = cfg->n_embd;
      int nh = cfg->n_head;
      int hs = C / nh;

      // Build x [T x C] from embeddings
      float *x = alloc(T * C);
      for (int t = 0; t < T; t++)
            for (int c = 0; c < C; c++)
                  x[t * C + c] = W->tok_emb.data[tokens[t] * C + c] + W->pos_emb.data[t * C + c];

      float *xn = alloc(T * C);
      float *tmp = alloc(T * C * 4); // scratch for ffwd

      for (int l = 0; l < cfg->n_layer; l++)
      {
            int base = l * nh;

            // ── LayerNorm 1 ──
            float *ln1_out = alloc(T * C);
            for (int t = 0; t < T; t++)
                  layer_norm(ln1_out + t * C, x + t * C, W->ln1_w[l].data, W->ln1_b[l].data, C);

            // ── Multi-head attention ──
            float *attn_out = alloc(T * C);
            float *head_out = alloc(T * C);
            memset(head_out, 0, T * C * sizeof(float));

            for (int h = 0; h < nh; h++)
            {
                  float *K = alloc(T * hs);
                  float *Q = alloc(T * hs);
                  float *V = alloc(T * hs);

                  matmul(K, ln1_out, W->head_k[base + h].data, T, C, hs);
                  matmul(Q, ln1_out, W->head_q[base + h].data, T, C, hs);
                  matmul(V, ln1_out, W->head_v[base + h].data, T, C, hs);

                  // Attention scores [T x T]
                  float *att = alloc(T * T);
                  float scale = 1.0f / sqrtf((float)hs);
                  for (int i = 0; i < T; i++)
                  {
                        for (int j = 0; j < T; j++)
                        {
                              if (j > i)
                              {
                                    att[i * T + j] = -1e9f;
                                    continue;
                              }
                              float dot = 0.0f;
                              for (int k = 0; k < hs; k++)
                                    dot += Q[i * hs + k] * K[j * hs + k];
                              att[i * T + j] = dot * scale;
                        }
                        softmax(att + i * T, T);
                  }

                  // Weighted sum of V → into head_out at offset h*hs
                  float *hv = alloc(T * hs);
                  for (int i = 0; i < T; i++)
                        for (int k = 0; k < hs; k++)
                        {
                              float s = 0.0f;
                              for (int j = 0; j <= i; j++)
                                    s += att[i * T + j] * V[j * hs + k];
                              hv[i * hs + k] = s;
                        }

                  for (int t = 0; t < T; t++)
                        for (int k = 0; k < hs; k++)
                              head_out[t * C + h * hs + k] = hv[t * hs + k];

                  free(K);
                  free(Q);
                  free(V);
                  free(att);
                  free(hv);
            }

            // Project attention output
            matmul(attn_out, head_out, W->sa_proj_w[l].data, T, C, C);
            add_bias(attn_out, W->sa_proj_b[l].data, T, C);

            // Residual
            for (int i = 0; i < T * C; i++)
                  x[i] += attn_out[i];
            free(head_out);
            free(attn_out);
            free(ln1_out);

            // LayerNorm 2
            float *ln2_out = alloc(T * C);
            for (int t = 0; t < T; t++)
                  layer_norm(ln2_out + t * C, x + t * C, W->ln2_w[l].data, W->ln2_b[l].data, C);

            // Feed forward
            int ff = 4 * C;
            float *ff1 = alloc(T * ff);
            float *ff2 = alloc(T * C);
            matmul(ff1, ln2_out, W->ff_w1[l].data, T, C, ff);
            add_bias(ff1, W->ff_b1[l].data, T, ff);
            relu(ff1, T * ff);
            matmul(ff2, ff1, W->ff_w2[l].data, T, ff, C);
            add_bias(ff2, W->ff_b2[l].data, T, C);

            // Residual
            for (int i = 0; i < T * C; i++)
                  x[i] += ff2[i];
            free(ln2_out);
            free(ff1);
            free(ff2);
      }

      // Final layer norm
      float *xf = alloc(C);
      layer_norm(xf, x + (T - 1) * C, W->ln_f_w.data, W->ln_f_b.data, C);

      // LM head - logits
      memset(logits, 0, cfg->vocab_size * sizeof(float));
      for (int v = 0; v < cfg->vocab_size; v++)
      {
            for (int c = 0; c < C; c++)
                  logits[v] += xf[c] * W->lm_w.data[v * C + c];
            logits[v] += W->lm_b.data[v];
      }

      free(x);
      free(xn);
      free(tmp);
      free(xf);
}

// Main
int main(void)
{
      signal(SIGINT, handle_sigint);
      srand((unsigned)time(NULL));

      FILE *fv = fopen("../vocab.bin", "rb");
      if (!fv)
      {
            fprintf(stderr, "[Error] Cannot open vocab.bin\n");
            return 1;
      }
      int vocab_size;
      fread(&vocab_size, sizeof(int), 1, fv);
      char *vocab = (char *)malloc(vocab_size);
      for (int i = 0; i < vocab_size; i++)
      {
            unsigned char c;
            fread(&c, 1, 1, fv);
            vocab[i] = (char)c;
      }
      fclose(fv);

      // Load weights
      FILE *fw = fopen("../weights.bin", "rb");
      if (!fw)
      {
            fprintf(stderr, "[Error] Cannot open weights.bin\n");
            return 1;
      }

      Config cfg;
      fread(&cfg.vocab_size, sizeof(int), 1, fw);
      fread(&cfg.block_size, sizeof(int), 1, fw);
      fread(&cfg.n_embd, sizeof(int), 1, fw);
      fread(&cfg.n_head, sizeof(int), 1, fw);
      fread(&cfg.n_layer, sizeof(int), 1, fw);

      Weights W = {0};
      W.tok_emb = read_tensor(fw);
      W.pos_emb = read_tensor(fw);

      int nl = cfg.n_layer, nh = cfg.n_head;
      W.head_k = (Tensor *)malloc(nl * nh * sizeof(Tensor));
      W.head_q = (Tensor *)malloc(nl * nh * sizeof(Tensor));
      W.head_v = (Tensor *)malloc(nl * nh * sizeof(Tensor));
      W.sa_proj_w = (Tensor *)malloc(nl * sizeof(Tensor));
      W.sa_proj_b = (Tensor *)malloc(nl * sizeof(Tensor));
      W.ff_w1 = (Tensor *)malloc(nl * sizeof(Tensor));
      W.ff_b1 = (Tensor *)malloc(nl * sizeof(Tensor));
      W.ff_w2 = (Tensor *)malloc(nl * sizeof(Tensor));
      W.ff_b2 = (Tensor *)malloc(nl * sizeof(Tensor));
      W.ln1_w = (Tensor *)malloc(nl * sizeof(Tensor));
      W.ln1_b = (Tensor *)malloc(nl * sizeof(Tensor));
      W.ln2_w = (Tensor *)malloc(nl * sizeof(Tensor));
      W.ln2_b = (Tensor *)malloc(nl * sizeof(Tensor));

      for (int l = 0; l < nl; l++)
      {
            for (int h = 0; h < nh; h++)
            {
                  W.head_k[l * nh + h] = read_tensor(fw);
                  W.head_q[l * nh + h] = read_tensor(fw);
                  W.head_v[l * nh + h] = read_tensor(fw);
            }
            W.sa_proj_w[l] = read_tensor(fw);
            W.sa_proj_b[l] = read_tensor(fw);
            W.ff_w1[l] = read_tensor(fw);
            W.ff_b1[l] = read_tensor(fw);
            W.ff_w2[l] = read_tensor(fw);
            W.ff_b2[l] = read_tensor(fw);
            W.ln1_w[l] = read_tensor(fw);
            W.ln1_b[l] = read_tensor(fw);
            W.ln2_w[l] = read_tensor(fw);
            W.ln2_b[l] = read_tensor(fw);
      }

      W.ln_f_w = read_tensor(fw);
      W.ln_f_b = read_tensor(fw);
      W.lm_w = read_tensor(fw);
      W.lm_b = read_tensor(fw);
      fclose(fw);

      printf("--- Model loaded ---\n");
      printf("[INFO] vocab=%d  block=%d  embd=%d  heads=%d  layers=%d\n",
             cfg.vocab_size,
             cfg.block_size,
             cfg.n_embd,
             cfg.n_head,
             cfg.n_layer);
      printf("Generating text (Ctrl+C to stop)...\n\n");
      printf("--------------------------------------------------\n");

      //  Generation loop
      int *ctx = (int *)calloc(cfg.block_size, sizeof(int));
      float *logits = alloc(cfg.vocab_size);
      int ctx_len = 1; // start with token 0
      ctx[0] = 0;

      while (running)
      {
            int T = ctx_len < cfg.block_size ? ctx_len : cfg.block_size;
            int *window = ctx + (ctx_len - T);

            forward(logits, window, T, &cfg, &W);
            softmax(logits, cfg.vocab_size);

            int next = sample(logits, cfg.vocab_size);
            printf("%c", vocab[next]);
            fflush(stdout);

            if (ctx_len < cfg.block_size)
                  ctx[ctx_len++] = next;
            else
            {
                  memmove(ctx, ctx + 1, (cfg.block_size - 1) * sizeof(int));
                  ctx[cfg.block_size - 1] = next;
            }
      }

      free(ctx);
      free(logits);
      free(vocab);
      return 0;
}