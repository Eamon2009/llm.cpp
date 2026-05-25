#include "tensor.h"
#include "torch_bridge.h"
#include <torch/torch.h>
#include <iostream>

int main()
{
      Tensor base({2, 3}, 0.0f);
      for (int i = 0; i < base.numel(); ++i)
            base.data[i] = static_cast<float>(i + 1);

      auto x = to_torch_tensor(base);
      auto y = torch::softmax(x, 1);
      auto z = torch::matmul(y, torch::randn({3, 2}));
      auto back = from_torch_tensor(z);

      std::cout << "Torch version: " << TORCH_VERSION << '\n';
      std::cout << "Input shape: [" << base.shape[0] << ", " << base.shape[1] << "]\n";
      std::cout << "Output shape: [" << back.shape[0] << ", " << back.shape[1] << "]\n";
      std::cout << back.at(0, 0) << '\n';

      return 0;
}
