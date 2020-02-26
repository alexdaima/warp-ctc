#include <TH/TH.h>

#include "ctc.h"

void ctc_cost_and_grad(THFloatTensor *th_activations,
                       THIntTensor *th_labels,
                       THIntTensor *th_lengths,
                       THIntTensor *th_label_lengths,
                       THFloatTensor *th_costs,
                       THFloatTensor *th_grads,
                       int blank_label) {

    int num_examples = THFloatTensor_size(th_activations, 1);
    int alphabet_size = THFloatTensor_size(th_activations, 2);

    ctcOptions options;
    options.loc = CTC_CPU;
    options.num_threads = 1;
    options.blank_label = blank_label;


    size_t cpu_alloc_bytes;

    float *activations = THFloatTensor_data(th_activations);
    int *lengths = THIntTensor_data(th_lengths);
    int *labels = THIntTensor_data(th_labels);
    int *label_lengths = THIntTensor_data(th_label_lengths);

    ctcStatus_t status = get_workspace_size(label_lengths, lengths,
                           alphabet_size, num_examples, options,
                           &cpu_alloc_bytes);

    THAssertMsg(status == CTC_STATUS_SUCCESS,
                 ctcGetStatusString(status));

    void* ctc_cpu_workspace = malloc(cpu_alloc_bytes);

    float *costs = THFloatTensor_data(th_costs);
    float *grads = THFloatTensor_data(th_grads);

    status = compute_ctc_loss(activations, grads,
                              labels, label_lengths,
                              lengths,
                              alphabet_size,
                              num_examples,
                              costs,
                              ctc_cpu_workspace,
                              options);

    THAssertMsg(status == CTC_STATUS_SUCCESS,
                 ctcGetStatusString(status));

    free(ctc_cpu_workspace);
}
