__author__ = 'Torbjorn V Ness, torbness@gmail.com'

import os
import sys
from os.path import join
import numpy as np
import matplotlib
matplotlib.use("AGG")
import pylab as plt
import scipy.fftpack as ff
import neuron
import LFPy
from plotting_convention import *


class Conductivity:

    def __init__(self, freq_dependence_type, freqs, clr, freq_1=None,
                 freq_2=None, avrg_sigma=None, increase_factor=None, name=None):

        self.freqs = freqs
        self.freq_dependence_type = freq_dependence_type
        self.avrg_sigma = avrg_sigma
        self.clr = clr
        self.name = name

        if self.freq_dependence_type == 'linear':
            self.freq_1 = freq_1
            self.freq_2 = freq_2
            self.sigma_time = avrg_sigma
            self.increase_factor = increase_factor
            self._avrg_conductivity_specified()
        elif self.freq_dependence_type == 'linear_stop':
            self.sigma_time = avrg_sigma
            self.freq_1 = freq_1
            self.freq_2 = freq_2
            self.sigma_time = avrg_sigma
            self.increase_factor = increase_factor
            self._avrg_conductivity_specified()
            self.sigma_freqs[np.where(self.freqs > self.freq_2)[0]] = self.sigma_freq_2
        elif self.freq_dependence_type == 'average':
            self.sigma_time = avrg_sigma
            self.sigma_freqs = self.sigma_time * np.ones(len(self.freqs))

    def _avrg_conductivity_specified(self):

        # Average sigma = (sigma_1 * (increase_factor + 1)) / 2
        sigma_slope = (2 * self.sigma_time * (self.increase_factor - 1) / (self.increase_factor + 1)
                       / (self.freq_2 - self.freq_1))
        sigma_0 = self.sigma_time * (1 - (self.increase_factor - 1)/(self.increase_factor + 1) *
                                         (self.freq_2 + self.freq_1) / (self.freq_2 - self.freq_1))
        self.sigma_freqs = sigma_0 + sigma_slope * self.freqs
        self.sigma_freq_1 = self.sigma_freqs[np.argmin(np.abs(self.freqs - self.freq_1))]
        self.sigma_freq_2 = self.sigma_freqs[np.argmin(np.abs(self.freqs - self.freq_2))]


class NeuralSignalDistortion:

    def __init__(self, sim_type, plot_psd,
                 input_idx=None, simulate_cell=False):
        self.root_dir = join('')
        self.sim_folder = join(self.root_dir, 'hay', 'sim_results')
        if not os.path.isdir(self.sim_folder):
            os.mkdir(self.sim_folder)

        self.figure_folder = self.root_dir
        self.model_folder = join(self.root_dir, 'hay')
        self.sim_type = sim_type
        self.plot_psd = plot_psd

        if sim_type is 'white_noise':
            self.T = 1000
            self.time_res = 2**-5
            self.cutoff = 0
            self.repeats = 2
        else:
            self.T = 80
            self.time_res = 2**-5
            self.cutoff = 1000

        self.tvec = np.arange(int(self.T / self.time_res + 1)) * self.time_res

        if sim_type == 'synaptic':
            if input_idx is None:
                raise RuntimeError("Need input index!")
            self.input_idx = 0 if input_idx is None else input_idx
            if plot_psd:
                self.ax_dict_1 = {'ylim': [1e-9, 1e-4],
                                  'xlim': [1e0, 1e4],}
            else:
                self.ax_dict_1 = {'ylim': [-1.1, 1.1], 'xlim': [5, 70]}
                self.ax_dict_2 = {'ylim': [-1.1, 1.1],
                                  'xlim': [5, 70]}
                self.ax_dict_3 = {'ylim': [-1.1, 1.1],
                                  'xlim': [5, 70]}

        elif sim_type == 'spike':
            self.input_idx = 0

            self.ax_dict_1 = {'ylim': [-1.05, 1.05], 'xlim': [9, 27]}
            self.ax_dict_2 = {'ylim': [-0.5, 1.05], 'xlim': [9, 27]}
            self.ax_dict_3 = {'ylim': [-1.05, 0.5],
                              'xlim': [9, 27]}

        elif sim_type == 'white_noise':
            self.input_idx = input_idx
            if plot_psd:
                self.ax_dict_1 = {'ylim': [1e-3, 2e0], 'aspect': 1,
                                  'xlim': [1, 4e2], 'xscale': 'log', 'yscale': 'log'}
                self.ax_dict_2 = {'ylim': [1e-3, 2e0], 'aspect': 1,
                                  'xlim': [1, 4e2], 'xscale': 'log', 'yscale': 'log'}
                self.ax_dict_3 = {'ylim': [1e-3, 2e0], 'aspect': 1, 'xlabel': 'Frequency [Hz]',
                                  'ylabel': 'LFP power\n[normalized]',
                                  'xlim': [1, 4e2], 'xscale': 'log', 'yscale': 'log'}
        else:
            raise RuntimeError("Unrecognized sim_type!")

        self.num_tsteps = int(self.T / self.time_res + 1)
        if sim_type == 'white_noise':
            self.num_tsteps -= 1
        self.tvec = np.arange(self.num_tsteps) * self.time_res
        self.sample_freq = ff.fftfreq(self.num_tsteps, d=self.time_res/1000.)
        self.pidxs = np.where(self.sample_freq >= 0)
        self.freqs = self.sample_freq[self.pidxs]
        self.num_freqs = len(self.freqs)

        if simulate_cell:
            self._neural_simulation()

    def return_phase_from_modulus(self, modulus_f):
        # Based on Clark et al. (2005)
        # https://e-reports-ext.llnl.gov/pdf/308517.pdf
        N = len(modulus_f)
        odd = N % 2
        aa = (N+odd)/2 - 1
        bb = (N-odd)/2 - 1
        u_plus = np.r_[1, 2*np.ones(aa), 1, np.zeros(bb)]
        cn = np.real(ff.ifft(np.log(modulus_f)))
        cf = u_plus * cn
        theta = np.imag(ff.fft(cf))
        return theta

    def impact_of_freq_dependence(self, electrode_paramters):
        """ Function to try to calculate extracellular potential in frequency domain
        """
        distortion_fixed = {'freq_1': 5.,
                            'freq_2': 500.,
                            'increase_factor': 1.,
                            'avrg_sigma': 0.4,
                            'freq_dependence_type': 'average',
                            'freqs': self.freqs,
                            'clr': 'r'}

        distortion_25_percent = {'freq_1': 5.,
                                 'freq_2': 500.,
                                 'increase_factor': 1.25,
                                 'avrg_sigma': 0.4,
                                 'freq_dependence_type': 'linear',
                                 'freqs': self.freqs,
                                 'clr': 'b'}

        distortion_25_percent_stop = {'freq_1': 5.,
                                 'freq_2': 500.,
                                 'increase_factor': 1.25,
                                 'avrg_sigma': 0.4,
                                 'freq_dependence_type': 'linear_stop',
                                 'freqs': self.freqs,
                                 'clr': 'g'}

        distortion_50_percent = {'freq_1': 5.,
                                 'freq_2': 500.,
                                 'increase_factor': 1.5,
                                 'avrg_sigma': 0.4,
                                 'freq_dependence_type': 'linear',
                                 'freqs': self.freqs,
                                 'clr': 'b'}

        distortion_50_percent_stop = {'freq_1': 5.,
                                      'freq_2': 500.,
                                      'increase_factor': 1.5,
                                      'avrg_sigma': 0.4,
                                      'freq_dependence_type': 'linear_stop',
                                      'freqs': self.freqs,
                                      'clr': 'g'}

        sigma_fixed = Conductivity(**distortion_fixed)
        sigma_25 = Conductivity(**distortion_25_percent)
        sigma_25_stop = Conductivity(**distortion_25_percent_stop)
        sigma_50 = Conductivity(**distortion_50_percent)
        sigma_50_stop = Conductivity(**distortion_50_percent_stop)

        xmid = np.load(join(self.sim_folder, 'xmid.npy'))
        ymid = np.load(join(self.sim_folder, 'ymid.npy'))
        zmid = np.load(join(self.sim_folder, 'zmid.npy'))
        num_comps = len(xmid)

        elec_x = electrode_paramters['x']
        elec_y = electrode_paramters['y']
        elec_z = electrode_paramters['z']

        self.num_elecs = len(elec_x)
        self.dists = np.zeros((self.num_elecs, num_comps))
        for elec in xrange(self.num_elecs):
            for comp in xrange(num_comps):
                self.dists[elec, comp] = ((xmid[comp] - elec_x[elec])**2 +
                                          (ymid[comp] - elec_y[elec])**2 +
                                          (zmid[comp] - elec_z[elec])**2) ** -0.5

        sim_name = self.sim_type if self.sim_type == 'spike' else '%s_%s' % (self.sim_type, self.input_idx)
        imem = np.load(join(self.sim_folder, 'imem_%s.npy' % sim_name))

        method = 'sig_psd_causal'

        sig_average, power_sig_average = self._calculate_distorted_signal(imem, sigma_fixed, 'sig_avrg')
        sig_corr_25, power_sig_corr_25 = self._calculate_distorted_signal(imem, sigma_25, method)
        sig_corr_25_stop, power_sig_corr_25_stop = self._calculate_distorted_signal(imem, sigma_25_stop, method)
        sig_corr_50, power_sig_corr_50 = self._calculate_distorted_signal(imem, sigma_50, method)
        sig_corr_50_stop, power_sig_corr_50_stop = self._calculate_distorted_signal(imem, sigma_50_stop, method)

        plt.close('all')
        fig = plt.figure(figsize=[10, 10])
        fig.subplots_adjust(left=0.1, wspace=0.5)

        ax_morph = self._draw_setup_to_axis(fig, electrode_paramters, ax_pos=[0.1, 0.02, 0.2, 0.76])

        ax_c1, lines, line_names = self._draw_conductivity_profiles_to_axis([sigma_fixed, sigma_25, sigma_25_stop], fig,
                                                 ax_pos=(4, 3, 2))

        ax_E1a = fig.add_subplot(435, **self.ax_dict_1)
        ax_E2a = fig.add_subplot(4, 3, 8, **self.ax_dict_2)
        ax_E3a = fig.add_subplot(4, 3, 11, **self.ax_dict_3)

        fig.text(0.4, 0.93, '25 % increase', size=20)
        fig.text(0.72, 0.93, '50 % increase', size=20)

        ax_E1b = fig.add_subplot(436, **self.ax_dict_1)
        ax_E2b = fig.add_subplot(439, **self.ax_dict_2)
        ax_E3b = fig.add_subplot(4, 3, 12, **self.ax_dict_3)

        ax_c2, lines, line_names = self._draw_conductivity_profiles_to_axis([sigma_fixed, sigma_50, sigma_50_stop], fig,
                                                 ax_pos=(4, 3, 3))

        mark_subplots(ax_morph, 'A', xpos=0, ypos=0.9)
        mark_subplots(ax_c1, 'B', xpos=-0.2, ypos=1.3)
        mark_subplots(ax_c2, 'C', xpos=-0.2, ypos=1.3)

        text_dict = {'va': 'center', 'ha': 'center', 'size': 12,}

        x1 = 0.37
        x2 = 0.68
        if self.sim_type == 'synaptic':
            fig.text(x1, 0.18, 'III', **text_dict)
            fig.text(x1, 0.395, 'II', **text_dict)
            fig.text(x1, 0.61, 'I', **text_dict)

            fig.text(x2, 0.18, 'III', **text_dict)
            fig.text(x2, 0.395, 'II', **text_dict)
            fig.text(x2, 0.61, 'I', **text_dict)
        else:
            fig.text(x1, 0.20, 'III', **text_dict)
            fig.text(x1, 0.37, 'II', **text_dict)
            fig.text(x1, 0.61, 'I', **text_dict)

            fig.text(x2, 0.20, 'III', **text_dict)
            fig.text(x2, 0.37, 'II', **text_dict)
            fig.text(x2, 0.61, 'I', **text_dict)

        line_dict = {'lw': 2, 'clip_on': True, 'ls': '-'}

        fixed_clr = 'r'
        increase_stop_clr = 'g'
        increasing_clr = 'b'

        self._print_peak_to_peak(sig_average, 'Average')
        self._print_peak_to_peak(sig_corr_25_stop, '25_stop')
        self._print_peak_to_peak(sig_corr_25, '25_lin')
        self._print_peak_to_peak(sig_corr_50_stop, '50_stop')
        self._print_peak_to_peak(sig_corr_50, '50_lin')

        normalize = lambda sig: sig / np.max(np.abs(sig))

        ax_E3a.plot(self.tvec, normalize(sig_average[0, :]), color=fixed_clr, **line_dict)
        ax_E2a.plot(self.tvec, normalize(sig_average[1, :]), color=fixed_clr, **line_dict)
        ax_E1a.plot(self.tvec, normalize(sig_average[2, :]), color=fixed_clr, **line_dict)

        ax_E3a.plot(self.tvec, normalize(sig_corr_25_stop[0, :]), color=increase_stop_clr, **line_dict)
        ax_E2a.plot(self.tvec, normalize(sig_corr_25_stop[1, :]), color=increase_stop_clr, **line_dict)
        ax_E1a.plot(self.tvec, normalize(sig_corr_25_stop[2, :]), color=increase_stop_clr, **line_dict)

        ax_E3b.plot(self.tvec, normalize(sig_average[0, :]), color=fixed_clr, **line_dict)
        ax_E2b.plot(self.tvec, normalize(sig_average[1, :]), color=fixed_clr, **line_dict)
        ax_E1b.plot(self.tvec, normalize(sig_average[2, :]), color=fixed_clr, **line_dict)

        ax_E3b.plot(self.tvec, normalize(sig_corr_50_stop[0, :]), color=increase_stop_clr, **line_dict)
        ax_E2b.plot(self.tvec, normalize(sig_corr_50_stop[1, :]), color=increase_stop_clr, **line_dict)
        ax_E1b.plot(self.tvec, normalize(sig_corr_50_stop[2, :]), color=increase_stop_clr, **line_dict)

        line_dict['ls'] = '--'
        ax_E3a.plot(self.tvec, normalize(sig_corr_25[0, :]), color=increasing_clr, **line_dict)
        ax_E2a.plot(self.tvec, normalize(sig_corr_25[1, :]), color=increasing_clr, **line_dict)
        ax_E1a.plot(self.tvec, normalize(sig_corr_25[2, :]), color=increasing_clr, **line_dict)

        ax_E3b.plot(self.tvec, normalize(sig_corr_50[0, :]), color=increasing_clr, **line_dict)
        ax_E2b.plot(self.tvec, normalize(sig_corr_50[1, :]), color=increasing_clr, **line_dict)
        ax_E1b.plot(self.tvec, normalize(sig_corr_50[2, :]), color=increasing_clr, **line_dict)

        line_dict['ls'] = '-'
        if self.sim_type == 'synaptic':
            ax_E3b.plot([10, 25], [-0.12, -0.12], color='k', lw=4, clip_on=False)
            ax_E3b.text(10, -0.6, '15 ms')

        elif self.sim_type == 'spike':
            ax_E3b.plot([20, 25], [-0.12, -0.12], color='k', lw=4, clip_on=False)
            ax_E3b.text(20, -0.4, '5 ms')

        [ax.axis('off') for ax in [ax_E1a, ax_E2a, ax_E3a, ax_E1b, ax_E2b, ax_E3b]]

        fig.savefig(join(self.root_dir, sim_name + '.png'))

        diff_25 = normalize(sig_average[0, :]) - normalize(sig_corr_25[0, :])
        diff_50 = normalize(sig_average[0, :]) - normalize(sig_corr_50[0, :])
        diff_25_stop = normalize(sig_average[0, :]) - normalize(sig_corr_25_stop[0, :])
        diff_50_stop = normalize(sig_average[0, :]) - normalize(sig_corr_50_stop[0, :])

        print ""
        print "25 %: ", np.max(diff_25) * 100
        print "25 % stop: ", np.max(diff_25_stop)* 100
        print "50 %: ", np.max(diff_50)* 100
        print "50 % stop: ", np.max(diff_50_stop)* 100

    def impact_of_freq_dependence_white_noise(self, electrode_parameters):
        """ Function to try to calculate extracellular potential in frequency domain
        """
        distortion_fixed = {'freq_1': 5.,
                            'freq_2': 500.,
                            'increase_factor': 1.,
                            'avrg_sigma': 0.4,
                            'freq_dependence_type': 'average',
                            'freqs': self.freqs,
                            'clr': 'r',
                            'name': 'Average'}

        distortion_25_percent = {'freq_1': 5.,
                                 'freq_2': 500.,
                                 'increase_factor': 1.25,
                                 'avrg_sigma': 0.4,
                                 'freq_dependence_type': 'linear',
                                 'freqs': self.freqs,
                                 'clr': 'b',
                                 'name': '25 % increase'}

        distortion_50_percent = {'freq_1': 5.,
                                 'freq_2': 500.,
                                 'increase_factor': 1.5,
                                 'avrg_sigma': 0.4,
                                 'freq_dependence_type': 'linear',
                                 'freqs': self.freqs,
                                 'clr': 'g',
                                 'name': '50 % increase'}

        sigma_fixed = Conductivity(**distortion_fixed)
        sigma_25 = Conductivity(**distortion_25_percent)
        sigma_50 = Conductivity(**distortion_50_percent)

        elec_x = electrode_parameters['x']
        elec_y = electrode_parameters['y']
        elec_z = electrode_parameters['z']

        xmid = np.load(join(self.sim_folder, 'xmid.npy'))
        ymid = np.load(join(self.sim_folder, 'ymid.npy'))
        zmid = np.load(join(self.sim_folder, 'zmid.npy'))
        num_comps = len(xmid)

        self.num_elecs = len(elec_x)
        self.dists = np.zeros((self.num_elecs, num_comps))
        for elec in xrange(self.num_elecs):
            for comp in xrange(num_comps):
                self.dists[elec, comp] = ((xmid[comp] - elec_x[elec])**2 +
                                          (ymid[comp] - elec_y[elec])**2 +
                                          (zmid[comp] - elec_z[elec])**2) ** -0.5

        sim_name = 'white_noise_%d' % self.input_idx
        imem = np.load(join(self.sim_folder, 'imem_%s.npy' % sim_name))
        vmem = np.load(join(self.sim_folder, 'somav_%s.npy' % sim_name))

        Y = ff.fft(vmem)

        amp_vmem = np.abs(Y)/len(Y) if Y.ndim == 1 else np.abs(Y)/Y.shape[1]
        amp_vmem = amp_vmem[self.pidxs[0]]

        method = 'sig_psd_causal'
        sig_average, power_sig_average = self._calculate_distorted_signal(1000 * imem, sigma_fixed, 'sig_avrg')
        sig_corr_25, power_sig_corr_25 = self._calculate_distorted_signal(1000 * imem, sigma_25, method)
        sig_corr_50, power_sig_corr_50 = self._calculate_distorted_signal(1000 * imem, sigma_50, method)

        plt.close('all')
        fig = plt.figure(figsize=[10, 10])

        fig.subplots_adjust(left=0.1, wspace=0.05, hspace=0.5, bottom=0.1, right=0.99)

        ax_morph = self._draw_setup_to_axis(fig, electrode_parameters, ax_pos=[0.1, 0.2, 0.2, 0.8])
        ax_cond, lines, line_names = self._draw_conductivity_profiles_to_axis([sigma_fixed, sigma_25, sigma_50],
                                                                              fig, ax_pos=(4, 3, 10))

        ax_E1a = fig.add_subplot(433, **self.ax_dict_1)
        ax_E2a = fig.add_subplot(436, **self.ax_dict_2)
        ax_E3a = fig.add_subplot(439, **self.ax_dict_3)

        ax_E1b = fig.add_subplot(432, xlim=[950, 1000], frameon=False, xticks=[], ylim=[-1., 1], yticks=[])
        ax_E2b = fig.add_subplot(435, xlim=[950, 1000], frameon=False, xticks=[], ylim=[-1, 1], yticks=[])
        ax_E3b = fig.add_subplot(438, xlim=[950, 1000], frameon=False, xticks=[], ylim=[-1, 1], yticks=[])

        ax_v = fig.add_axes([0.5, 0.1, 0.2, 0.15], xlim=[1e0, 4e2], ylim=[1e-3, 1e0], aspect=1,
                            ylabel='mV/Hz', xlabel='Frequency [Hz]')
        ax_v.grid(True)
        max_freq_idx = np.argmin(np.abs(self.freqs - 500))

        l1, = ax_v.loglog(self.freqs[1:max_freq_idx], amp_vmem[1:max_freq_idx], lw=2, c='k')# / 0.0005 * 1e6)
        lines.append(l1)
        line_names.append('Somatic V$_m$')

        mark_subplots(ax_morph, 'A', xpos=0, ypos=0.8)
        mark_subplots(ax_E1b, 'C')
        mark_subplots(ax_E1a, 'D')
        mark_subplots(ax_v, 'E')

        text_dict = {'va': 'center', 'ha': 'center', 'size': 12}
        x1 = 0.35

        fig.text(x1, 0.38, 'III', **text_dict)
        fig.text(x1, 0.6, 'II', **text_dict)
        fig.text(x1, 0.8, 'I', **text_dict)

        line_dict = {'lw': 2, 'clip_on': True, 'ls': '-'}

        fixed_clr = 'r'
        increasing_clr = 'b'
        increasing_clr2 = 'g'

        normalize = lambda sig: sig / np.max(np.abs(sig[:]))
        ax_E1b.plot(self.tvec, normalize(sig_average[2]), c=fixed_clr)
        ax_E1b.plot(self.tvec, normalize(sig_corr_25[2]), c=increasing_clr)
        ax_E1b.plot(self.tvec, normalize(sig_corr_50[2]), c=increasing_clr2)

        ax_E2b.plot(self.tvec, normalize(sig_average[1]), c=fixed_clr)
        ax_E2b.plot(self.tvec, normalize(sig_corr_25[1]), c=increasing_clr)
        ax_E2b.plot(self.tvec, normalize(sig_corr_50[1]), c=increasing_clr2)

        ax_E3b.plot(self.tvec, normalize(sig_average[0]), c=fixed_clr)
        ax_E3b.plot(self.tvec, normalize(sig_corr_25[0]), c=increasing_clr)
        ax_E3b.plot(self.tvec, normalize(sig_corr_50[0]), c=increasing_clr2)

        bar = 1
        ax_E3b.plot([950, 960], [bar * 0.7, bar * 0.7], lw=3, c='k')
        ax_E3b.text(955, bar * 1.1, '10 ms', ha='center')

        freqs = self.freqs[1:max_freq_idx]
        ax_E3a.plot(freqs, normalize(power_sig_average[0, 1:max_freq_idx]), color=fixed_clr, **line_dict)
        ax_E2a.plot(freqs, normalize(power_sig_average[1, 1:max_freq_idx]), color=fixed_clr, **line_dict)
        ax_E1a.plot(freqs, normalize(power_sig_average[2, 1:max_freq_idx]), color=fixed_clr, **line_dict)

        line_dict['ls'] = '--'
        ax_E3a.plot(freqs, normalize(power_sig_corr_25[0, 1:max_freq_idx]), color=increasing_clr, **line_dict)
        ax_E2a.plot(freqs, normalize(power_sig_corr_25[1, 1:max_freq_idx]), color=increasing_clr, **line_dict)
        ax_E1a.plot(freqs, normalize(power_sig_corr_25[2, 1:max_freq_idx]), color=increasing_clr, **line_dict)

        ax_E3a.plot(freqs, normalize(power_sig_corr_50[0, 1:max_freq_idx]), color=increasing_clr2, **line_dict)
        ax_E2a.plot(freqs, normalize(power_sig_corr_50[1, 1:max_freq_idx]), color=increasing_clr2, **line_dict)
        ax_E1a.plot(freqs, normalize(power_sig_corr_50[2, 1:max_freq_idx]), color=increasing_clr2, **line_dict)

        line_dict['ls'] = '-'

        [ax.grid(True) for ax in [ax_E1a, ax_E2a, ax_E3a]]

        mark_subplots(ax_morph, 'A')
        mark_subplots(ax_cond, 'B', ypos=1.2)
        simplify_axes([ax_E1a, ax_E2a, ax_E3a, ax_v])

        save_name = 'white_noise'

        fig.legend(lines, line_names, frameon=False, loc='lower right')
        fig.savefig(join(self.root_dir, save_name + '.png'))


    def _print_peak_to_peak(self, signal, name):
        print name
        for elec in xrange(signal.shape[0]):
            print 1000 * (np.max(signal[elec]) - np.min(signal[elec]))


    def _draw_conductivity_profiles_to_axis(self, sigmas, fig, ax_pos):

        if self.sim_type is 'white_noise':
            ax = fig.add_subplot(ax_pos[0], ax_pos[1], ax_pos[2], xlim=[5, 500],
                                 ylim=[0.3, 0.5], yticks=[0.3, 0.4, 0.5], xticks=[0, 200, 400],
                             xlabel='Frequency [Hz]', ylabel='Conductivity\n[S/m]')
        else:
            ax = fig.add_subplot(ax_pos[0], ax_pos[1], ax_pos[2], xlim=[1, 1000],
                                 ylim=[0.2, 0.8], yticks=[0.2, 0.4, 0.6, 0.8], xticks=[0, 400, 800],
                             xlabel='Frequency [Hz]', ylabel='Conductivity [S/m]')

        lines = []
        line_names = []
        for sigma in sigmas:
            ls = '-' if not sigma.freq_dependence_type is 'linear' else '--'
            l, = ax.plot(self.freqs, sigma.sigma_freqs, ls=ls, lw=2, color=sigma.clr)
            lines.append(l)
            line_names.append(sigma.name)

        simplify_axes(ax)
        return ax, lines, line_names

    def _draw_setup_to_axis(self, fig, electrode_parameters, ax_pos):
        ax = fig.add_axes(ax_pos, aspect=1)
        elec_x = electrode_parameters['x']
        elec_z = electrode_parameters['z']

        xstart = np.load(join(self.sim_folder, 'xstart.npy'))
        zstart = np.load(join(self.sim_folder, 'zstart.npy'))
        xend = np.load(join(self.sim_folder, 'xend.npy'))
        zend = np.load(join(self.sim_folder, 'zend.npy'))
        xmid = np.load(join(self.sim_folder, 'xmid.npy'))
        zmid = np.load(join(self.sim_folder, 'zmid.npy'))

        synapse_clr = 'y'
        cell_clr = '0.6'
        elec_clr = 'k'

        if hasattr(self, 'input_idx') and (type(self.input_idx) != str):
            ax.plot(xmid[self.input_idx], zmid[self.input_idx], c=synapse_clr, marker='*', zorder=1, ms=15)

        [ax.plot([xstart[idx], xend[idx]], [zstart[idx], zend[idx]], lw=2,
                 color=cell_clr, zorder=0) for idx in xrange(len(xmid))]
        ax.plot(xmid[0], zmid[0], '^', color=cell_clr, zorder=0, ms=20, mec='none')

        ax.scatter(elec_x, elec_z, c=elec_clr, edgecolor='none', s=100, zorder=2)

        text_dict = {
                     'va': 'center', 'ha': 'center',
                     'size': 12,
                    }
        ax.text(elec_x[2] + 0, elec_z[2] + 50, 'I', **text_dict)
        ax.text(elec_x[1] + 0, elec_z[1] + 50, 'II', **text_dict)
        ax.text(elec_x[0] + 0, elec_z[0] + 50, 'III', **text_dict)

        ax.axis('off')
        return ax

    def _calculate_distorted_signal(self, imem, sigma, method='imem_psd'):

        if method == 'sig_psd_causal':
            sig_temp = 1./(4 * np.pi * 1) * np.dot(self.dists, imem)
            Y = ff.fft(sig_temp, axis=1)
            if self.sim_type is 'white_noise' or self.sim_type is 'distributed_synaptic':
                sigma_T = np.r_[sigma.sigma_freqs, sigma.sigma_freqs[::-1]]
            else:
                sigma_T = np.r_[sigma.sigma_freqs, sigma.sigma_freqs[::-1][1:]]
            phase = self.return_phase_from_modulus(1./sigma_T)
            Y = Y/sigma_T * np.exp(1j * phase)
            sig = np.real(ff.ifft(Y, axis=1))
            Y = Y[:, self.pidxs[0]]
            power_sig = np.abs(Y)**2/len(Y) if Y.ndim == 1 else np.abs(Y)**2/Y.shape[1]
            power_sig = power_sig[:, self.pidxs[0]]

        elif method == 'sig_avrg':
            sig = 1./(4 * np.pi * sigma.sigma_time) * np.dot(self.dists, imem)
            Y = ff.fft(sig, axis=1)[:, self.pidxs[0]]
            power_sig = np.abs(Y)**2/len(Y) if Y.ndim == 1 else np.abs(Y)**2/Y.shape[1]
        else:
            raise RuntimeError("Unrecognized 'method'")
        return sig, power_sig

    def _insert_synapse(self, cell, input_idx, weight=0.025):
        import LFPy
        # Define synapse parameters
        synapse_parameters = {
            'idx': input_idx,
            'e': 0.,                   # reversal potential
            'syntype': 'ExpSyn',       # synapse type
            'tau': 2.,                # syn. time constant
            'weight': weight,            # syn. weight
            'record_current': False,
        }
        synapse = LFPy.Synapse(cell, **synapse_parameters)
        synapse.set_spike_times(np.array([10.]))
        return cell, synapse

    def _make_WN_input(self, cell, max_freq):
        """ White Noise input ala Linden 2010 is made """
        tot_ntsteps = int(round((cell.tstopms - cell.tstartms) / cell.timeres_NEURON + 1))
        I = np.zeros(tot_ntsteps)
        tvec = np.arange(tot_ntsteps) * cell.timeres_NEURON
        for freq in xrange(1, max_freq + 1):
            I += np.sin(2 * np.pi * freq * tvec/1000. + 2*np.pi*np.random.random())
        return I

    def make_white_noise_stimuli(self, cell, input_idx, weight=0.0005, holding_potential=None):
        max_freq = 500
        plt.seed(1234)
        input_array = weight * self._make_WN_input(cell, max_freq)
        noiseVec = neuron.h.Vector(input_array)

        # plt.close('all')
        # plt.plot(input_array)
        # plt.show()
        print 1000 * np.std(input_array)
        i = 0
        syn = None
        for sec in cell.allseclist:
            for seg in sec:
                if i == input_idx:
                    print "Input inserted in ", sec.name()
                    syn = neuron.h.ISyn(seg.x, sec=sec)
                    # print "Dist: ", nrn.distance(seg.x)
                i += 1
        if syn is None:
            raise RuntimeError("Wrong stimuli index")
        syn.dur = 1E9
        syn.delay = 0 #cell.tstartms
        noiseVec.play(syn._ref_amp, cell.timeres_NEURON)
        return cell, syn, noiseVec

    def _neural_simulation(self):

        sys.path.append(self.model_folder)
        if self.sim_type == 'spike':
            from hay_active_declarations import active_declarations
            neuron_models = join(self.model_folder)
            cell_params = {
                'morphology': join(neuron_models, 'cell1.hoc'),
                'v_init': -70,
                'passive': False,           # switch off passive mechs
                'nsegs_method': 'lambda_f',  # method for setting number of segments,
                'lambda_f': 100,           # segments are isopotential at this frequency
                'timeres_NEURON': self.time_res,   # dt of LFP and NEURON simulation.
                'timeres_python': self.time_res,
                'tstartms': -self.cutoff,          # start time, recorders start at t=0
                'tstopms': self.T,
                'custom_code': [join(neuron_models, 'custom_codes.hoc')],
                'custom_fun': [active_declarations],  # will execute this function
                'custom_fun_args': [{'conductance_type': 'active',
                                     'hold_potential': -70}]
            }

        elif self.sim_type == 'synaptic':
            from hay_active_declarations import active_declarations
            neuron_models = join(self.model_folder)
            cell_params = {
                'morphology': join(neuron_models, 'cell1.hoc'),
                'v_init': -70,
                'passive': False,
                'nsegs_method': 'lambda_f',  # method for setting number of segments,
                'lambda_f': 100,           # segments are isopotential at this frequency
                'timeres_NEURON': self.time_res,   # dt of LFP and NEURON simulation.
                'timeres_python': self.time_res,
                'tstartms': -self.cutoff,          # start time, recorders start at t=0
                'tstopms': self.T,
                'custom_code': [join(neuron_models, 'custom_codes.hoc')],
                'custom_fun': [active_declarations],  # will execute this function
                'custom_fun_args': [{'conductance_type': 'active',
                                     'hold_potential': -70}]
            }
        elif self.sim_type == 'white_noise':
            from hay_active_declarations import active_declarations
            neuron_models = join(self.model_folder)
            cell_params = {
                'morphology': join(neuron_models, 'cell1.hoc'),
                'v_init': -80,
                'passive': False,
                'nsegs_method': 'lambda_f',  # method for setting number of segments,
                'lambda_f': 100,           # segments are isopotential at this frequency
                'timeres_NEURON': self.time_res,   # dt of LFP and NEURON simulation.
                'timeres_python': self.time_res,
                'tstartms': -self.cutoff,          # start time, recorders start at t=0
                'tstopms': self.T * self.repeats,
                'custom_code': [join(neuron_models, 'custom_codes.hoc')],
                'custom_fun': [active_declarations],  # will execute this function
                'custom_fun_args': [{'conductance_type': 'active',
                                     'hold_potential': -70}]
            }
        else:
            raise RuntimeError("Unknown sim_type")

        print "Making cell"
        neuron.load_mechanisms(self.model_folder)
        cell = LFPy.Cell(**cell_params)
        if self.sim_type == 'spike':
            cell, synapse = self._insert_synapse(cell, 0, 0.05)
        elif self.sim_type == 'synaptic':
            cell, synapse = self._insert_synapse(cell, self.input_idx, 0.01)
        elif self.sim_type == 'white_noise':
            cell, synapse, wn = self.make_white_noise_stimuli(cell, self.input_idx, 0.005)
        print "Running %s simulation ... " % self.sim_type

        cell.simulate(rec_imem=True, rec_vmem=False)
        self._save_neural_sim(cell)

    def _save_neural_sim(self, cell):
        sim_name = self.sim_type if self.sim_type == 'spike' else '%s_%d' % (self.sim_type, self.input_idx)

        if hasattr(self, 'repeats') and self.repeats is not None:
            cut_off_idx = (len(cell.tvec) - 1) / self.repeats
            cell.tvec = cell.tvec[-cut_off_idx:] - cell.tvec[-cut_off_idx]
            cell.imem = cell.imem[:, -cut_off_idx:]
            cell.somav = cell.somav[-cut_off_idx:]

        np.save(join(self.sim_folder, 'imem_%s.npy' % sim_name), cell.imem)
        np.save(join(self.sim_folder, 'tvec_%s.npy' % sim_name), cell.tvec)
        np.save(join(self.sim_folder, 'somav_%s.npy' % sim_name), cell.somav)

        np.save(join(self.sim_folder, 'xstart.npy'), cell.xstart)
        np.save(join(self.sim_folder, 'ystart.npy'), cell.ystart)
        np.save(join(self.sim_folder, 'zstart.npy'), cell.zstart)
        np.save(join(self.sim_folder, 'xend.npy'), cell.xend)
        np.save(join(self.sim_folder, 'yend.npy'), cell.yend)
        np.save(join(self.sim_folder, 'zend.npy'), cell.zend)
        np.save(join(self.sim_folder, 'xmid.npy'), cell.xmid)
        np.save(join(self.sim_folder, 'ymid.npy'), cell.ymid)
        np.save(join(self.sim_folder, 'zmid.npy'), cell.zmid)
        print "Simulation data saved ..."


def figure_synaptic():

    input_idx = 852
    elec_x = np.array([50., 50.,  50.])
    elec_z = np.array([0., 500., 1000.])
    elec_y = np.zeros(elec_x.shape)

    electrode_parameters = {
        'sigma': 0.4,      # extracellular conductivity
        'x': elec_x.flatten(),  # electrode requires 1d vector of positions
        'y': elec_y.flatten(),
        'z': elec_z.flatten(),
        'method': 'pointsource'
    }

    sd = NeuralSignalDistortion('synaptic', plot_psd=False,
                                input_idx=input_idx, simulate_cell=True)
    sd.impact_of_freq_dependence(electrode_parameters)

def figure_white_noise():

    input_idx = 0
    elec_x = np.array([50., 50., 50.])
    elec_z = np.array([0., 500., 1000.])
    elec_y = np.zeros(elec_x.shape)

    electrode_parameters = {
        'sigma': 0.4,      # extracellular conductivity
        'x': elec_x.flatten(),  # electrode requires 1d vector of positions
        'y': elec_y.flatten(),
        'z': elec_z.flatten(),
        'method': 'pointsource'
    }

    sd = NeuralSignalDistortion('white_noise', plot_psd=True,
                                input_idx=input_idx, simulate_cell=True)
    sd.impact_of_freq_dependence_white_noise(electrode_parameters)


def figure_spike():
    elec_x = np.array([50., 50., 50.])
    elec_z = np.array([0., 500., 1000.])
    elec_y = np.zeros(elec_x.shape)

    electrode_parameters = {
        'sigma': 0.4,      # extracellular conductivity
        'x': elec_x.flatten(),  # electrode requires 1d vector of positions
        'y': elec_y.flatten(),
        'z': elec_z.flatten(),
        'method': 'pointsource'
    }
    sd = NeuralSignalDistortion('spike', plot_psd=False, simulate_cell=True)
    sd.impact_of_freq_dependence(electrode_parameters)


if __name__ == '__main__':

    figure_synaptic()
    figure_spike()
    figure_white_noise()
