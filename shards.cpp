#include "tokenizer.h"
#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char *argv[])
{
      if (argc < 3)
      {
            std::cerr << "Usage: " << argv[0]
                      << " <input.txt> <out_dir> [vocab] [shard_size] [max_train_mb]\n"
                      << "  vocab      default = " << BPE_VOCAB_SIZE << "\n"
                      << "  shard_size default = " << SHARD_SIZE_TOKENS << " tokens\n"
                      << "  max_train_mb default = 50 (cap BPE training to N MB of text)\n";
            return 1;
      }

      std::string in = argv[1];
      std::string out = argv[2];
      int vocab = (argc >= 4) ? std::atoi(argv[3]) : BPE_VOCAB_SIZE;
      uint64_t shard_n = (argc >= 5) ? std::strtoull(argv[4], nullptr, 10) : SHARD_SIZE_TOKENS;
      int max_mb = (argc >= 6) ? std::atoi(argv[5]) : 50;
      size_t max_bytes = (size_t)max_mb * 1024 * 1024;

      try
      {
            DataLoader dl;
            std::cout << "[make]  Training BPE on first " << max_mb << " MB of: " << in << "\n";
            dl.load_text(in, vocab, TRAIN_SPLIT, max_bytes);

            std::cout << "[make]  Vocab=" << dl.vocab_size << " | Train=" << dl.train_data.size()
                      << " | Val=" << dl.val_data.size() << "\n";

            std::cout << "[make]  Writing shards to: " << out << "\n";
            dl.write_shards(out, shard_n);

            std::cout << "[make]  Done. Train with: ./llm.exe " << out << "\n";
      }
      catch (const std::exception &e)
      {
            std::cerr << "[error] " << e.what() << "\n";
            return 1;
      }
      return 0;
}