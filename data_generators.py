import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from pyro import distributions

E_T_MAX = 200
E_T_MIN = 0
DELTA_PHI_MIN = -np.pi
DELTA_PHI_MAX = np.pi
M_JJ_MIN = 50
M_JJ_MAX = 200


def _get_acceptance(E_t, delta_phi, m_jj):
    # returns a bool indicating whether a particular event will be accepted by the detector
    E_t_acceptance = E_T_MIN <= E_t <= E_T_MAX
    delta_phi_acceptance = DELTA_PHI_MIN <= delta_phi <= DELTA_PHI_MAX
    m_jj_acceptance = M_JJ_MIN <= m_jj <= M_JJ_MAX
    return all([E_t_acceptance, delta_phi_acceptance, m_jj_acceptance])


class DataGenerator:

    def __init__(self, transverse_energy_distribution, delta_phi_distribution, m_jj_distribution):
        self.transverse_energy_distribution = transverse_energy_distribution
        self.delta_phi_distribution = delta_phi_distribution
        self.m_jj_distribution = m_jj_distribution

    def plot_distributions(self, n_data=1000):
        dist_limits = [(self.transverse_energy_distribution, [E_T_MIN, E_T_MAX]),
                       (self.delta_phi_distribution, [DELTA_PHI_MIN, DELTA_PHI_MAX]),
                       (self.m_jj_distribution, [M_JJ_MIN, M_JJ_MAX])]
        f, axarr = plt.subplots(len(dist_limits))
        for i, [dist, (dist_min, dist_max)] in enumerate(dist_limits):
            x = np.linspace(start=dist_min, stop=dist_max, num=n_data)
            axarr[i].plot(x, np.exp(dist.log_prob(torch.tensor(x))))
        plt.tight_layout()
        plt.show()


def generate_data(n, background_generator, signal_generator, signal_prob=6.5e-3):
    signal_dist = distributions.Bernoulli(signal_prob)
    samples = []
    i = 0
    while i < n:
        signal = signal_dist.sample().item()
        collection = signal_generator if signal else background_generator
        sample = {
            'E_t': collection.transverse_energy_distribution.sample().item(),
            'delta_phi': collection.delta_phi_distribution.sample().item(),
            'm_jj': collection.m_jj_distribution.sample().item()
        }
        accepted = _get_acceptance(**sample)
        if accepted:
            sample['signal'] = signal
            samples.append(sample)
            i += 1
            if i % int(n / 10) == 0:
                print('{}/{} samples completed'.format(i, n))
        else:
            continue
    return pd.DataFrame(samples)


class StandardModelBackgroundGenerator(DataGenerator):

    def __init__(self):
        transverse_energy_distribution = distributions.Uniform(low=E_T_MIN, high=E_T_MAX)
        delta_phi_distribution = distributions.Uniform(low=DELTA_PHI_MIN, high=DELTA_PHI_MAX)
        m_jj_distribution = distributions.HalfCauchy(120)
        super().__init__(transverse_energy_distribution, delta_phi_distribution, m_jj_distribution)


class StandardModelSignalGenerator(DataGenerator):
    width = 2

    def __init__(self, m_h=125):
        transverse_energy_distribution = distributions.Gamma(15, 0.1)
        delta_phi_distribution = distributions.Uniform(low=DELTA_PHI_MIN, high=DELTA_PHI_MAX)
        m_jj_distribution = distributions.Normal(m_h, self.width)
        super().__init__(transverse_energy_distribution, delta_phi_distribution, m_jj_distribution)


if __name__ == '__main__':
    s = StandardModelSignalGenerator()
    b = StandardModelBackgroundGenerator()
    df = generate_data(n=int(1e6), background_generator=b, signal_generator=s)
    df['m_jj'].round().value_counts().reset_index().rename(columns={'index': 'm_jj', 'm_jj': 'count'}).plot.scatter(
        x='m_jj', y='count', xlim=[1.1*M_JJ_MIN, 0.9*M_JJ_MAX])
    plt.show()
