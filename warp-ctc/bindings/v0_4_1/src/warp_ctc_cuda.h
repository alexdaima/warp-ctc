
void ctc_cost_and_grad_cuda(THCudaTensor *th_activations,
                       THIntTensor *th_labels,
                       THIntTensor *th_lengths,
                       THIntTensor *th_label_lengths,
                       THFloatTensor *th_costs,
                       THCudaTensor *th_grads,
                       int blank_label);
