import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from qulacs import QuantumCircuit, QuantumGateMatrix, QuantumState
from qulacs.gate import CNOT, CZ, P0, RY, DenseMatrix, merge, to_matrix_gate
from qulacs.state import inner_product
from scipy import optimize


def main():
    for depth in range(2, 7):
        for inputsize in range(2, 6):
            test_190131(depth, inputsize, [0]*(inputsize-2))


def color_plot(data, figsize=(7, 6), title=None,
               clabel=None, clim=None, cmap="seismic", logscale=False,
               tick_labels=None, ticks=None, lims=None,
               aspect="equal", font='sans-serif', fontsize=16,
               tick_directions=('in', 'in'), tick_widths=(1.0, 1.0),
               tick_levels=None,
               **kwargs):

    # if imagename is not None and isinstance(imagename, Path):
    #     imagename = Path(imagename)

    # see https://qiita.com/skotaro/items/08dc0b8c5704c94eafb9
    fig, ax = plt.subplots(figsize=figsize)
    plt.rcParams['font.family'] = font  # 使用するフォント
    plt.rcParams['font.size'] = fontsize  # フォントの大きさ
    plt.rcParams['axes.labelsize'] = fontsize  # フォントの大きさ
    plt.rcParams['xtick.direction'] = tick_directions[0]
    plt.rcParams['ytick.direction'] = tick_directions[1]
    plt.rcParams['xtick.major.width'] = tick_widths[0]
    plt.rcParams['ytick.major.width'] = tick_widths[1]

    if clim is None:
        _norm = clr.LogNorm() if logscale else None
    else:
        _norm = clr.LogNorm(*clim) if logscale else clr.Normalize(*clim)

    if ticks is None:
        _im = ax.imshow(data, norm=_norm, cmap=cmap, **kwargs)
    else:
        x, y = np.meshgrid(*ticks)
        _im = ax.pcolormesh(x, y, data, norm=_norm, cmap=cmap, **kwargs)

    if title is not None:
        ax.set_title(title)

    if tick_labels is not None:
        ax.set_xlabel(tick_labels[0])
        ax.set_ylabel(tick_labels[1])

    if lims is not None:
        if lims[0] is not None:
            ax.set_xlim(*lims[0])
        if lims[1] is not None:
            ax.set_ylim(*lims[1])

    if aspect is not None:
        ax.set_aspect(aspect)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', '5%', pad='3%')
    cbar = fig.colorbar(_im, cax=cax)
    if clabel is not None:
        cbar.ax.set_ylabel(clabel)

        # fig.savefig(".".join([imagename, "png"]))

    return fig, ax


def scan_multi_bits(input_bit_size, depth, weights=[], train_data_list=None,
                    file_id="_0", **kwargs):
    # w1, w2について走査する．
    # if len(weights) != input_bit_size-2:
    #     raise ValueError("SizeError")

    qns = QuantumNeuralSystem(input_bit_size, depth)

    if train_data_list is None:
        qnum = 2 ** (input_bit_size)
        train_data_list = [i for i in range(1, qnum)]

    ws = np.linspace(-np.pi, np.pi, 100)
    array_2d = []
    array_2d_s = []
    for w2 in ws:
        temp = []
        temp_s = []
        for w1 in ws:
            loss, sampling = qns.test(train_data_list, [0.0, w1, w2, *weights],
                                      **kwargs)
            temp.append(loss)
            temp_s.append(sampling)

        array_2d.append(temp)
        array_2d_s.append(temp_s)

    title = "(input, depth)=({}, {})\n weights={}".format(
        input_bit_size, depth, [0.0, "w1", "w2", *weights])
    fig, ax = color_plot(array_2d, ticks=(ws, ws), cmap="jet",
                         tick_labels=("w1", "w2"),
                         title=title,
                         clabel="loss")
    fig2, ax2 = color_plot(array_2d_s, ticks=(ws, ws), cmap="jet",
                           tick_labels=("w1", "w2"),
                           title=title,
                           clabel="sampling counts")

    # filename = "scan_xor_input{}_d{}_{}".format(
    #     input_bit_size, depth, file_id)
    filename = "{}bit_{}_d{}".format(input_bit_size,
                                     "all",
                                     depth)
    fig.savefig(filename)
    fig2.savefig(filename + "_ps")
    plt.close()

    header = "(input, depth)=({}, {})\n".format(input_bit_size, depth)
    header += "weights={}".format([0.0, "w1", "w2", *weights])
    header += "state=\n {}\n".format(qns.state)

    np.savetxt(filename + ".txt", array_2d, header=header)


def test_190124(bias=0.0, w3=0.0, no=0):
    # 入力3bitの実験
    depth = 4
    input_bit_size = 3
    meas_circ = get_measurement_circuit("XOR", input_bit_size, depth)
    qns = QuantumNeuralSystem(meas_circ, input_bit_size, depth)

    train_data_list = [0b001, 0b011, 0b110]
    ws = np.linspace(-np.pi, np.pi, 100)
    array_2d = []
    for w2 in ws:
        temp = []
        for w1 in ws:
            temp.append(qns.test(train_data_list, [bias, w1, w2, w3]))

        array_2d.append(temp)

    str_input = ["|{:03b}>".format(bit) for bit in train_data_list]
    title = "depth={}\n input={}\n weights={}".format(
        depth, str_input, [bias, "w1", "w2", w3])

    fig, ax = color_plot(array_2d, ticks=(ws, ws), cmap="jet",
                         tick_labels=("w1", "w2"),
                         title=title,
                         clabel="loss")

    filename = "scan_xor_in{}_d{}_no{}".format(
        train_data_list, depth, no)
    fig.savefig(filename)

    np.savetxt(filename+".txt", array_2d)


def scan_weights_in2bit(depth=1, bias=0.0, train_data_list=[0b01], show=True):
    ws = np.linspace(-np.pi, np.pi, 100)
    qns = QuantumNeuralSystem(input_bit_size=2, depth=depth)

    array_2d = []
    array_2d_s = []
    for w2 in ws:
        temp = []
        temp_s = []
        for w1 in ws:
            loss, sampling = qns.test(train_data_list, [bias, w1, w2])
            temp.append(loss)
            temp_s.append(sampling)

        array_2d.append(temp)
        array_2d_s.append(temp_s)

    str_input = ["|{:02b}>".format(bit) for bit in train_data_list]

    fig, ax = color_plot(array_2d, ticks=(ws, ws), cmap="jet",
                         tick_labels=("weight 1st input", "weight 2nd input"),
                         title="depth={}, bias={} \n input={}".format(
        depth, bias, str_input,),
        clabel="loss")

    fig_s, ax = color_plot(array_2d_s, ticks=(ws, ws), cmap="jet",
                           tick_labels=("weight 1st input",
                                        "weight 2nd input"),
                           title="depth={}, bias={} \n input={}".format(
        depth, bias, str_input,),
        clabel="loss")
    if show:
        plt.show()
    else:
        fig.savefig("2bit_{}_d{}".format(train_data_list, depth))
        fig_s.savefig("2bit_{}_d{}_ps".format(train_data_list, depth))


def play_xor_simulation(input_bit_size=2, depth=1, debug_mode=True,
                        train_data_list=[0b01, 0b10], weights=None,
                        show_only_initial_state=False):
    if weights is None:
        weights = np.ones(input_bit_size+1)
    meas_circ = get_measurement_circuit("XOR", input_bit_size, depth)
    qns = QuantumNeuralSystem(meas_circ, input_bit_size, depth, debug_mode)
    qns.test(train_data_list, weights, show_only_initial_state)


class QuantumNeuralSystem:
    '''
    0~n_i-1:input_qubit,
    n_i~n_i+d-1: ancilla_qubit,
    n_i+d: output_qubit．
    '''

    def __init__(self, input_bit_size, depth=1, measurement_circuit=None,
                 debug_mode=False):
        '''
        Args:
            measurement_circuit(QuantumCircuit): 正解データ→訓練データの写像
            input_bit_size(uint): size of input bits: i
            depth(uint): depth of quantum neural net (:q-NN) :d
            debug_mode(bool): in debug_mode, outputs some variables

        memo:
            0~i-1 bit: input qubits
            i~i+d-1 bit: ancilla bits for an activation.
            i+d bit: 
        '''
        self.input_bit_size = input_bit_size
        self.depth = depth

        self.qubit_count = self.input_bit_size + self.depth + 1

        self.debug_mode = debug_mode
        if measurement_circuit is None:
            self.measurement_circuit = get_measurement_circuit(
                "XOR", input_bit_size, depth)
        else:
            self.measurement_circuit = measurement_circuit

        self.measurement_circuit.add_P0_gate(self.qubit_count-1)
        self.state = QuantumState(self.qubit_count)

        # for debug
        self.step = 3

    def test(self, train_data_list, weights=None, show_only_initialstate=False,
             step=3, show_unit_gate=False):
        """
        RUS回路のテストのため，各作用を経た後の状態を出力していく．
        Args:
            train_data_list(array-like): training data
            weights(array-like): initial weights
        return loss
        """
        # for debug of unit_gate
        if self.depth == 1:
            self.step = step

        if weights is None:
            weights = np.zeros(self.input_bit_size + 1)

        self.set_state(*train_data_list)
        if self.debug_mode or show_only_initialstate:
            print("Initial state:\n{}\n".format(self.state))

        if show_only_initialstate:
            return 0

        try:
            loss = self.weights2loss(weights)
        except Exception as e:
            print("loss calculation is failed: {}".format(e))
            loss = -99
        if show_unit_gate:
            print(self.unit_gate)

        if self.debug_mode:
            # print("After measurement:\n{}\n".format(
            #     self.state.get_vector()))

            print("Loss: {}".format(loss))

            print("Probs: {}".format(self.prob_in_each_steps))

        return loss, self.get_sampling_counts()

    def learn(self, train_data_list, weights=None,
              iteration=500, loss_tol=1e-9, circuit_name="XOR"):
        """
        Args:
            train_data_list(array-like): 訓練データのリスト
            weights(array-like): 初期重みパラメーター
            iteration(uint): イテレーション
            loss_tol(float>0): lossの更新を終える許容値
            circuit_name(str): 学習したい関数や回路の名前
        return: None
        """
        self.output_path = Path("Results")
        if not self.output_path.exists():
            os.makedirs(self.output_path)

        # 重み
        if weights is None:
            # biasの分を足す: self.num_inputbit + 1．
            self.weights = np.random.rand(self.input_bit_size + 1)
        else:
            self.weights = np.array(weights)

        weights_init = self.weights

        # 初期化
        self.set_state(*train_data_list)
        self.weights_list = []
        self.error_list = []
        self.loss_list = []
        self.post_selection_num_list = []

        # 学習
        for i in range(iteration):
            results = optimize.minimize(self.weights2loss, self.weights,
                                        method="Nelder-Mead",
                                        options={"adaptive": False,
                                                 "maxiter": 3})
            self.weights = results["x"]
            err = abs(results["fun"])
            print("{}, weight: {}, err: {}".format(i+1, results["x"], err))
            self.weights_list.append([i+1, *results["x"]])
            # self.error_list.append([i+1, err])

            if err < loss_tol:
                break

        filename = "learn_bit{}_depth{}".format(self.input_bit_size,
                                                self.depth)

        np.savetxt(self.output_path / "_".join([filename, "loss.txt"]),
                   self.loss_list,
                   header="{}".format(weights_init))

        np.savetxt(self.output_path / "_".join([filename, "postselection.txt"]),
                   self.post_selection_num_list,
                   header="{}".format(weights_init))

        self.set_weights(self.weights)

    def weights2loss(self, weights):
        """
        ある重みに対するロスを返す
        Args:
            weights(array-like): 重み
        """
        self.set_weights(weights, self.step)
        self.update_quantum_state(self.state)

        self.measurement_circuit.update_quantum_state(self.state)

        if self.debug_mode:
            print("innner product of: {}".format(self.state.get_vector()))
        loss = (1-inner_product(self.state, self.state).real /
                np.prod(self.prob_in_each_steps))

        self.state.load(self.initial_state)

        # self.loss_list.append(loss)
        # self.post_selection_num_list.append(1/np.prod(self.prob_in_each_steps))

        return loss

    def get_sampling_counts(self):
        return 1.0/np.prod(self.prob_in_each_steps)

    def update_quantum_state(self, state):
        '''
        量子状態に対して量子ニューロンのゲートを作用させる．
        各ステップごとの成功確率を返す．
        '''
        self.prob_in_each_steps = []

        self._step_update_state(self.depth, state)

    def _step_update_state(self, step, state):
        '''
        helper: 各ステップにおけるゲート作用
        '''
        _control = self.input_bit_size + step - 1
        _target = self.input_bit_size + step

        if step > 1:
            # if self.debug_mode and step == 2:
            #     print("1st 1-size RUS:")
            self._step_update_state(step - 1, state)
            # if self.debug_mode and step > 2:
            #     print("1st {}-size RUS:\n{}\n".format(step-1, state.get_vector()))
            ciY = self._get_ciYgate(_control, _target)
            ciY.update_quantum_state(state)
            # if self.debug_mode:
            #     print("In step {} iY:\n{}\n".format(step, state.get_vector()))
            # if self.debug_mode and step == 2:
            #     print("2nd 1-size RUS:")
            self._step_update_state(step - 1, state)
            # if self.debug_mode and step > 2:
            #     print("2nd {}-size RUS:\n{}\n".format(step-1, state.get_vector()))
        else:
            self.unit_gate.update_quantum_state(state)

            if self.debug_mode:
                print("unit {} RUS:\n{}\n"
                      .format(step, state.get_vector()))

        if self.debug_mode:
            print("Normalize Check: {}".format(
                inner_product(self.state, self.state)))
        prob = self._measure_0projection(_target-1)
        self.prob_in_each_steps.append(prob)
        if self.debug_mode:
            print("prob: {}".format(prob))
            print("Step {}: After projection: \n{}\n"
                  .format(step, state.get_vector()))

    def set_state(self, *input_bits):

        size = 2 ** self.qubit_count
        _amp = 1.0 / np.sqrt(len(input_bits))
        _temp = [_amp if i in input_bits else 0 for i in range(size)]

        self.state.load(_temp)
        self.initial_state = self.state.copy()

    def load_state(self, quantum_state):
        self.state.load(quantum_state)

    def set_weights(self, weights, step=3):
        '''
        weightsを設定し，RUS基本回路を更新する．
        Args:
            weights(array-like): 重み
            step(0<uint<=3): debug用．通常3でよい．
        '''
        self.weights = weights
        self.set_unit_gate(step)

    def set_unit_gate(self, step=3):
        '''
        RUSの基本回路(測定は別で行う)を作成する．
        Args:
            step(0<uint<=3): debug用．通常3でよい．
        '''
        target = self.input_bit_size
        count = 0

        # R_y
        self.unit_gate = RY(target, -2*self.weights[0])
        for control_bit, weight in enumerate(self.weights[1:]):
            c_ry_mat = self._get_cRY_gate(-2*weight, control_bit, target)
            self.unit_gate = merge(self.unit_gate, c_ry_mat)

        count += 1
        if count == step:
            print("stop@1")
            return self.unit_gate

        # controlled-iY
        y_target = target + 1
        y_control = target
        ciY = self._get_ciYgate(y_control, y_target)
        self.unit_gate = merge(self.unit_gate, ciY)

        count += 1
        if count == step:
            print("stop@2")
            return self.unit_gate

        # R_y^\dagger
        self.unit_gate = merge(self.unit_gate, RY(target, 2*self.weights[0]))
        for control_bit, weight in enumerate(self.weights[1:]):
            c_ry_mat = self._get_cRY_gate(2*weight, control_bit, target)
            self.unit_gate = merge(self.unit_gate, c_ry_mat)

        return self.unit_gate

    def _get_cRY_gate(self, angle, control, target):
        '''
        helper: controlled RY gateの作成
        Args:
            angle(float [rad]): 回転角
            control: controlビット
            target: targetビット
        '''
        c_ry = RY(target, angle)
        c_ry_mat = to_matrix_gate(c_ry)
        c_ry_mat.add_control_qubit(control, 1)

        return c_ry_mat

    def _get_ciYgate(self, control, target):
        '''
        helper: controlled iY gateの作成
        Args:
            control: controlビット
            target: targetビット
        '''
        iY = DenseMatrix(target, [[0, -1], [1, 0]])
        iY.add_control_qubit(control, 1)

        return iY

    def _measure_0projection(self, target):
        '''
        helper: P0の射影を取り，確率を返す．
        '''
        _state = self.state.copy()
        P0(target).update_quantum_state(self.state)
        prob = inner_product(_state, self.state) / \
            inner_product(_state, _state)
        prob = prob.real

        return prob

    def get_neuron_circuit(self):
        self.nc = QuantumCircuit(self.qubit_count)
        self.set_unit_gate()

        self._add_step_neuron_circuit(self.depth)

        return self.nc

    def _add_step_neuron_circuit(self, step):
        if step > 1:
            self._add_step_neuron_circuit(step - 1)

            _control = self.input_bit_size + step - 1
            _target = self.input_bit_size + step
            ciY = self._get_ciYgate(_control, _target)
            self.nc.add_gate(ciY)

            self._add_step_neuron_circuit(step - 1)

        else:
            self.nc.add_gate(self.unit_gate)


def get_measurement_circuit(name, input_bit_size, depth):
    qubit_count = input_bit_size + depth + 1
    measure_circuit = QuantumCircuit(qubit_count)

    if name.upper() == "XOR":
        for i in range(input_bit_size):
            measure_circuit.add_CNOT_gate(i, input_bit_size + depth)

    return measure_circuit


def xor_bit(bit_sequence):
    '''
    与えられたビット列について各ビットの排他的論理総和を返す
    '''
    xor = 0
    while (bit_sequence > 0):
        # print(int(bit_sequence) % 2)
        xor ^= int(bit_sequence) % 2

        bit_sequence = int(bit_sequence) / 2

    return xor


def check_nelder_mead():
    def func(x):
        return (x - 3) ** 2 - 9

    results = optimize.minimize(func, [-50, ],
                                method="Nelder-Mead",
                                options={"adaptive": True,
                                         "maxiter": 100})
    print(results["x"])


if __name__ == "__main__":
    main()
