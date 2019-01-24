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


def main_xor(states=[0b01, 0b00], iteration=500, weight=[0, 0, 0]):
    for depth in range(1, 6):
        xor_learn(depth, states, iteration, weight)


def xor_learn(depth=1, states=[0b01, 0b00], iteration=500, weight=[0, 0, 0]):
    xors = XOR_simulator(2, depth=depth)
    xors.learn(states, iteration, weight)


def test_xor(inputbit=2, depth=1, weights=[0, 0, 0], states=[0b01, 0b00],
             debug=True):
    xors = XOR_simulator(inputbit, depth, debug)
    loss = xors.test(states, weights)

    return loss


def scan_xor(depth=1, bias=0, states=[0b01],
             showfig=False, savedata=True):
    b = bias
    w1_w2_list = []
    ws = np.linspace(-np.pi, np.pi, 100)
    for w1 in ws:
        w2_list = []
        for w2 in ws:
            loss = test_xor(depth=depth, inputbit=2,
                            states=states, weights=[b, w1, w2], debug=False)
            w2_list.append(loss)

        w1_w2_list.append(w2_list)

    fig, ax = plt.subplots()
    _im = ax.pcolormesh(*np.meshgrid(ws, ws), w1_w2_list,
                        cmap="jet", label="bias={}".format(bias))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', '5%', pad='3%')
    fig.colorbar(_im, cax=cax)

    if showfig:
        plt.show()

    fig.savefig("test_xor_in{}_d{}b{}".format(states, depth, bias))

    if savedata:
        np.savetxt("test_xor_in{}_d{}b{}.txt"
                   .format(states, depth, bias), w1_w2_list)

    return w1_w2_list


# def TEST_unit_gate(step=3):
#     vects = []
#     states = []
#     for angle in np.linspace(-np.pi, np.pi, 100):
#         qns = QuantumNeuralSystem(1, 1)
#         state = qns.set_state(0b001)

#         qns.set_weights([0, angle], step)
#         qns.unit_gate.update_quantum_state(state)

#         cos = np.sqrt((np.cos(2 * angle) ** 2 + 1) / 2)
#         sin = -np.sin(2 * angle)
#         tan0 = 1 / np.sqrt(1 + np.tan(angle) ** 4)
#         tan1 = np.tan(angle) ** 2 / np.sqrt(1 + np.tan(angle) ** 4)

#         vect = np.zeros(8)
#         vect[1] = cos * tan0
#         vect[5] = cos * tan1
#         vect[3] = sin * 1/2
#         vect[7] = -sin * 1/2

#         # print("state: {}".format(state.get_vector()))
#         # print("ans: {}".format(vect))
#         # print("diff: {}".format(state.get_vector() - vect))

#         states.append(state.get_vector())
#         vects.append(vect)

#     return states, vects

def play_xor_simulation(input_bit_size=2, depth=1, debug_mode=True,
                        train_data_list=[0b01, 0b10], weights=[0, 0, 0]):
    meas_circ = get_measurement_circuit("XOR", input_bit_size, depth)
    qns = QuantumNeuralSystem(meas_circ, input_bit_size, depth, debug_mode)
    qns.test(train_data_list, weights)


class QuantumNeuralSystem:
    '''
    0~n_i-1:input_qubit,
    n_i~n_i+d-1: ancilla_qubit,
    n_i+d: output_qubit．
    '''

    def __init__(self, measurement_circuit, input_bit_size, depth=1,
                 debug_mode=False):
        '''
        Args:
            measurement_circuit(QuantumCircuit): 訓練データの逆写像
            input_bit_size(uint): number of ancilla bits
            depth(uint): depth of quantum neural net(q-NN)
            debug_mode(bool): in debug_mode, outputs some variables
        '''
        self.measurement_circuit = measurement_circuit
        self.input_bit_size = input_bit_size
        self.depth = depth

        self.qubit_count = self.input_bit_size + self.depth + 1

        self.debug_mode = debug_mode

    def test(self, train_data_list, weights):
        """
        RUS回路のテストのため，各作用を経た後の状態を出力していく．
        Args:
            train_data_list(array-like): training data
            weights(array-like): initial weights
        return loss
        """
        weights = np.array(weights)

        self.set_state(*train_data_list)
        if self.debug_mode:
            print("Initial state:\n{}\n".format(self.state.get_vector()))

        loss = self.weights2loss(weights)

        if self.debug_mode:
            print("After measurement:\n{}\n".format(
                self.measured_state.get_vector()))

            print("Loss: {}".format(loss))

            print("Probs: {}".format(self.prob_in_each_steps))

        return loss

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

        # 出力先の決定
        self.name = "{}_d{}_in{}_{}_{}".format(circuit_name,
                                               self.depth,
                                               self.input_bit_size,
                                               train_data_list,
                                               self.weights)
        self.output_path = Path("Results")/self.name
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

        # 学習
        for i in range(iteration):
            results = optimize.minimize(self.weights2loss, self.weights,
                                        method="Nelder-Mead",
                                        options={"adaptive": False,
                                                 "maxiter": 3})
            self.weights = results["x"]
            err = abs(results["fun"])
            print("{}, weight: {}, err: {}".format(i+1, results["x"], err))
            # print(self._state)
            self.weights_list.append([i+1, *results["x"]])
            self.error_list.append([i+1, err])

            if err < loss_tol:
                break

        np.savetxt(self.output_path / "_".join([self.name, "err.txt"]),
                   self.error_list,
                   header="{}".format(weights_init))

        np.savetxt(self.output_path / "_".join([self.name, "weig.txt"]),
                   self.weights_list,
                   header="{}".format(weights_init))

        self.set_weights(self.weights)

    def weights2loss(self, weights):
        """
        ある重みに対するロスを返す
        Args:
            weights(array-like): 重み
        """
        self.set_weights(weights)
        self.update_quantum_state(self.state)

        self.measurement_circuit.update_quantum_state(self.state)
        if self.debug_mode:
            self.measured_state = self.state.copy()

        _temp = inner_product(self.initial_state, self.state)

        loss = -(_temp.conjugate()*_temp).real + 1

        if np.abs(_temp) - 1 > 1e-6:
            print("Something wrong with unitary operations: {}".format(_temp))

        self.state.load(self.initial_state)

        return loss

    def set_state(self, *input_bits):
        self.state = QuantumState(self.qubit_count)

        size = 2 ** self.qubit_count
        _amp = 1.0 / np.sqrt(len(input_bits))
        _temp = [_amp if i in input_bits else 0 for i in range(size)]

        self.state.load(_temp)
        self.initial_state = self.state.copy()

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
        self.unit_gate = RY(target, -self.weights[0])
        for control_bit, weight in enumerate(self.weights[1:]):
            c_ry_mat = self._get_cRY_gate(-weight, control_bit, target)
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
        self.unit_gate = merge(self.unit_gate, RY(target, self.weights[0]))
        for control_bit, weight in enumerate(self.weights[1:]):
            c_ry_mat = self._get_cRY_gate(weight, control_bit, target)
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

    def _measure_0projection(self, target, state):
        '''
        helper: P0の射影を取り，確率を返す．
        '''
        _state = state.copy()
        P0(target).update_quantum_state(state)

        prob = inner_product(_state, state) / inner_product(_state, _state)
        prob = prob.real

        return prob

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
            self._step_update_state(step - 1, state)
            if self.debug_mode:
                print("1st {}-size RUS:\n{}\n".format(step-1, state.get_vector()))
            ciY = self._get_ciYgate(_control, _target)
            ciY.update_quantum_state(state)
            if self.debug_mode:
                print("In step {} iY:\n{}\n".format(step, state.get_vector()))

            self._step_update_state(step - 1, state)
            if self.debug_mode:
                print("2nd {}-size RUS:\n{}\n".format(step-1, state.get_vector()))

        else:
            self.unit_gate.update_quantum_state(state)

            if self.debug_mode:
                print("unit {} RUS:\n{}\n"
                      .format(step, state.get_vector()))

        prob = self._measure_0projection(_target-1, state)
        self.prob_in_each_steps.append(prob)
        if self.debug_mode:
            print("prob: {}".format(prob))
            print("Step {}: After projection: \n{}\n"
                  .format(step, state.get_vector()))

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


# class XOR_simulator:
#     def __init__(self, input_bit_size, depth=1, debug_mode=False):
#         self.qns = QuantumNeuralSystem(input_bit_size, depth, debug_mode)
#         self.input_bit_size = input_bit_size
#         self.qubit_count = input_bit_size + depth + 1
#         self.depth = depth
#         self.measure_circuit = QuantumCircuit(self.qubit_count)
#         for i in range(self.input_bit_size):
#             self.measure_circuit.add_CNOT_gate(i, input_bit_size + depth)

#         self.debug_mode = debug_mode

#     def test(self, train_data_list, weights):
#         """
#         Args:
#             debug: 各ゲートを作用させた後の状態を逐一表示する
#         """
#         weights = np.array(weights)
#         self.num_of_superpositon = len(train_data_list)
#         print("super position: {}".format(self.num_of_superpositon))

#         self.set_state(train_data_list)
#         if self.debug_mode:
#             print("Initial state:\n\t{}".format(self.state.get_vector()))

#         loss = self.weights2loss(weights)
#         if self.debug_mode:
#             print("Loss: {}".format(loss))

#         if self.debug_mode:
#             print("After measurement:\n\t{}".format(
#                 self.measured_state.get_vector()))

#         return loss

#     def learn(self, train_data_list, iteration, weights=None, loss_tol=1e-9):
#         if weights is None:
#             # biasの分を足す: self.num_inputbit + 1．
#             self.weights = np.random.rand(self.input_bit_size + 1)
#         else:
#             self.weights = np.array(weights)

#         weights_init = self.weights

#         self.name = "d{}_in{}_{}_{}".format(self.depth,
#                                             self.input_bit_size,
#                                             train_data_list,
#                                             self.weights)

#         self.output_path = Path("Results")/self.name

#         if not self.output_path.exists():
#             os.makedirs(self.output_path)

#         # print("initial weights: {}".format(self.weights))
#         self.set_state(train_data_list)

#         self.weights_list = []
#         self.error_list = []

#         for i in range(iteration):
#             results = optimize.minimize(self.weights2loss, self.weights,
#                                         method="Nelder-Mead",
#                                         options={"adaptive": False,
#                                                  "maxiter": 3})
#             self.weights = results["x"]
#             err = abs(results["fun"])
#             print("{}, weight: {}, err: {}".format(i+1, results["x"], err))
#             # print(self._state)
#             self.weights_list.append([i+1, *results["x"]])
#             self.error_list.append([i+1, err])

#             if err < loss_tol:
#                 break

#         np.savetxt(self.output_path / "_".join([self.name, "err.txt"]),
#                    self.error_list,
#                    header="{}".format(weights_init))

#         np.savetxt(self.output_path / "_".join([self.name, "weig.txt"]),
#                    self.weights_list,
#                    header="{}".format(weights_init))

#     def set_state(self, input_bits):
#         self.state = self.qns.set_state(*input_bits)
#         self.initial_state = self.state.copy()

#     def weights2loss(self, weights):
#         self.qns.set_weights(weights)
#         self.qns.update_quantum_state(self.state)
#         self._state = self.state.copy()

#         self.measure_circuit.update_quantum_state(self.state)
#         _temp = inner_product(self.initial_state, self.state)

#         self.measured_state = self.state.copy()
#         loss = -(_temp.conjugate()*_temp).real * \
#             np.sqrt(self.num_of_superpositon) + 1
#         if np.abs(_temp) - 1 > 1e-6:
#             print("Something wrong with unitary operations: {}".format(_temp))

#         self.state.load(self.initial_state)

#         return loss

#     def plot_results(self):
#         fig, axs = plt.subplots()
#         # ax.plot(self.weights_list[:, 0], self.weights_list[:, 1], "bo-")
#         axs.plot(self.error_list[:, 0], self.error_list[:, 1], "ro-")

#         fig.savefig(self.name + "_err.png", self.error_list)


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
    parser = ArgumentParser(prog=__file__, add_help=True)
    parser.add_argument('-t', '--test', action='store_true',
                        default=False, required=False, help='test')
    args = parser.parse_args()
