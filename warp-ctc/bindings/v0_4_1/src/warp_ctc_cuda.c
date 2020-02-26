#include <TH/TH.h>
#include <THC/THC.h>

#include "ctc.h"

/* PyTorch specific gpu global state. */
extern THCState *state;

void ctc_cost_and_grad_cuda(THCudaTensor *th_activations,
                       THIntTensor *th_labels,
                       THIntTensor *th_lengths,
                       THIntTensor *th_label_lengths,
                       THFloatTensor *th_costs,
                       THCudaTensor *th_grads,
                       int blank_label) {

    int num_examples = THCudaTensor_size(state, th_activations, 1);
    int alphabet_size = THCudaTensor_size(state, th_activations, 2);

    cudaStream_t stream;
    if (cudaStreamCreate(&stream)) {
        THError("cudaStreamCreate");
    }

    ctcOptions options;
    options.loc = CTC_GPU;
    options.stream = stream;
    options.blank_label = blank_label;

    size_t gpu_alloc_bytes;

    float *activations = THCudaTensor_data(state, th_activations);
    int *lengths = THIntTensor_data(th_lengths);
    int *labels = THIntTensor_data(th_labels);
    int *label_lengths = THIntTensor_data(th_label_lengths);

    ctcStatus_t status = get_workspace_size(label_lengths, lengths,
                           alphabet_size, num_examples, options,
                           &gpu_alloc_bytes);

    THAssertMsg(status == CTC_STATUS_SUCCESS,
                 ctcGetStatusString(status));

    void* ctc_gpu_workspace;
    if (cudaMalloc(&ctc_gpu_workspace, gpu_alloc_bytes)) {
        THError("cudaMalloc");
    }

    float *costs = THFloatTensor_data(th_costs);
    float *grads = THCudaTensor_data(state, th_grads);

    status = compute_ctc_loss(activations, grads,
                              labels, label_lengths,
                              lengths,
                              alphabet_size,
                              num_examples,
                              costs,
                              ctc_gpu_workspace,
                              options);

    THAssertMsg(status == CTC_STATUS_SUCCESS,
                 ctcGetStatusString(status));

    if (cudaFree(ctc_gpu_workspace)) {
        THError("cudaFree");
    }
    if (cudaStreamDestroy(stream)) {
        THError("cudaStreamDestroy");
    }
}
