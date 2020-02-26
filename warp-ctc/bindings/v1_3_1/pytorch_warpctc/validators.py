import torch


def validate_type(var, t, name):
    if var.dtype is not t:
        raise TypeError("{} must be {}".format(name, t))


def validate_contiguous(var, name):
    if not var.is_contiguous():
        raise ValueError("{} must be contiguous".format(name))


def validate_dim(var, dim, name):
    if len(var.size()) != dim:
        raise ValueError("{} must be {}D".format(name, dim))


def validate_inputs(activations, labels, lengths, label_lengths):

    validate_type(activations, torch.float32, "activations")
    validate_type(labels, torch.int32, "labels")
    validate_type(label_lengths, torch.int32, "label_lengths")
    validate_type(lengths, torch.int32, "lengths")
    validate_contiguous(labels, "labels")
    validate_contiguous(label_lengths, "label_lengths")
    validate_contiguous(lengths, "lengths")

    if lengths.size()[0] != activations.size()[0]:
        raise ValueError("must have a length per example.")
    if label_lengths.size()[0] != activations.size()[0]:
        raise ValueError("must have a label length per example.")

    validate_dim(activations, 3, "activations")
    validate_dim(labels, 1, "labels")
    validate_dim(lengths, 1, "lenghts")
    validate_dim(label_lengths, 1, "label_lenghts")
