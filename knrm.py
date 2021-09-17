def kernel_mu(n_kernels):
    """
    get mu for each guassian kernel, Mu is the middele of each bin
    :param n_kernels: number of kernels( including exact match). first one is the exact match
    :return: mus, a list of mu
    """
    mus = [1]  # exact match
    if n_kernels == 1:
        return mus
    bin_step = (1 - (-1)) / (n_kernels - 1)  # score from [-1, 1]
    mus.append(1 - bin_step / 2)  # the margain mu value
    for k in range(1, n_kernels - 1):
        mus.append(mus[k] - bin_step)
    return mus


def kernel_sigma(n_kernels):
    """
    get sigmas for each guassian kernel.
    :param n_kernels: number of kernels(including the exact match)
    :return: sigmas, a list of sigma
    """
    sigmas = [0.001]  # exact match small variance means exact match ?
    if n_kernels == 1:
        return sigmas
    return sigmas + [0.1] * (n_kernels - 1)


class KNRM(nn.Module):
    def __init__(self, nkernels, device):
        super(KNRM, self).__init__()
        self.mus = torch.FloatTensor(kernel_mu(nkernels))
        self.mus = self.mus.view(1, 1, 1, nkernels).to(device)  # (1, 1, 1, n_kernels) view 操作是为了配合后面的 interaction matrix 的操作
        self.sigmas = torch.FloatTensor(kernel_sigma(nkernels))
        self.sigmas = self.sigmas.view(1, 1, 1, nkernels).to(device)  # (1, 1, 1, n_kernels)

    def interaction_matrix(self, match_matrix):
        # translation matrix
        # match_matrix: (batch_size * query_length * doc_length * 1)
        # RBF Kernel layers
        # kernel_pooling: batch_size * query_length * doc_length * n_kernels
        q_mask = torch.gt(torch.sum(match_matrix.squeeze(3), dim=2), 0).long()
        q_mask = q_mask.view(q_mask.size()[0], q_mask.size()[1], 1)
        t_mask = torch.gt(torch.sum(match_matrix.squeeze(3), dim=1), 0).long()
        t_mask = t_mask.view(t_mask.size()[0], 1, t_mask.size()[1], 1)
        kernel_pooling = torch.exp(-((match_matrix - self.mus) ** 2) / (2 * (self.sigmas ** 2)))
        # kernel_pooling_row: batch_size * query_length  * doc_length * n_kernels
        kernel_pooling_row = kernel_pooling * t_mask
        # pooling_row_sum -> batch_size * query_length * n_kernels
        pooling_row_sum = torch.sum(kernel_pooling_row, 2)
        # kernel_pooling -> batch_size * query_length * n_kernels
        log_pooling = torch.log(torch.clamp(pooling_row_sum, min=1e-10)) * q_mask * 0.01  # scale down the data
        # sum the value on col th: log_pooling_sum: (batch_size * n_kernels)
        log_pooling_sum = torch.sum(log_pooling, 1)
        return log_pooling_sum

    def forward(self, matrix_i, matrix_cls):
        '''
        matrix_i, matrix_cls: [bs, pad_len, pad_len]
        '''
        matrix_i = matrix_i.unsqueeze(3)
        matrix_cls = matrix_cls.unsqueeze(3)
        pooling_i = self.interaction_matrix(matrix_i)
        pooling_cls = self.interaction_matrix(matrix_cls)
        return pooling_i, pooling_cls # [bs, nkernels]