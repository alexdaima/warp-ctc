int gpu_ctc(torch::Tensor probs,
            torch::Tensor grads,
            torch::Tensor labels,
            torch::Tensor label_sizes,
            torch::Tensor sizes,
            int minibatch_size,
            torch::Tensor costs,
            int blank_label);
