import numpy as np

import utils
from models import MeanFieldVINN, VanillaNN


def run_vcl(hidden_size, num_epochs, data_generator, coreset_method, coreset_size=0, batch_size=None, single_head=True):
    """It runs the variational continual learning algorithm presented in "Variational Continual Learning" (2018) by
    Cuong V. Nguyen et al.

    :param hidden_size:
    :param num_epochs:
    :param data_generator:
    :param coreset_method:
    :param coreset_size:
    :param batch_size:
    :param single_head:
    :return:
    """
    in_dim, out_dim = data_generator.get_dims()

    # TODO: what is difference between coresets and testsets? Maybe coresets are training sets?
    x_coresets, y_coresets = [], []

    x_testsets, y_testsets = [], []

    all_acc = np.array([])

    # max_iter corresponds to the number of tasks (?)
    for task_id in range(data_generator.max_iter):
        x_train, y_train, x_test, y_test = data_generator.next_task()

        x_testsets.append(x_test)
        y_testsets.append(y_test)

        # Set the readout head to train
        head = 0 if single_head else task_id

        bsize = x_train.shape[0] if (batch_size is None) else batch_size

        # Train network with maximum likelihood to initialize first model
        if task_id == 0:
            ml_model = VanillaNN(in_dim, hidden_size, out_dim, x_train.shape[0])

            ml_model.train(x_train, y_train, task_id, num_epochs, bsize)

            mf_weights = ml_model.get_weights()

            mf_variances = None

            ml_model.close_session()

        # Select coreset if needed
        if coreset_size > 0:
            x_coresets, y_coresets, x_train, y_train = coreset_method(x_coresets, y_coresets, x_train, y_train,
                                                                      coreset_size)

        # Train on non-coreset data
        mf_model = MeanFieldVINN(in_dim, hidden_size, out_dim, x_train.shape[0], prev_means=mf_weights,
                                 prev_log_variances=mf_variances)
        mf_model.train(x_train, y_train, head, num_epochs, bsize)
        mf_weights, mf_variances = mf_model.get_weights()

        # Incorporate coreset data and make prediction
        acc = utils.get_scores(mf_model, x_testsets, y_testsets, x_coresets, y_coresets, hidden_size, num_epochs,
                               single_head, batch_size)

        all_acc = utils.concatenate_results(acc, all_acc)

        mf_model.close_session()

    return all_acc
