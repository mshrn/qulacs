import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from qulacs import QuantumCircuit, QuantumGateMatrix, QuantumState
from qulacs.gate import CNOT, CZ, P0, RY, DenseMatrix, merge, to_matrix_gate
from qulacs.state import inner_product
from scipy import optimize


def main_xor():
    for depth in range(1, 6):
        xor_learn(depth)


def xor_learn(depth=1):
    xors = XOR_simulator(2, depth=depth)
    xors.learn([0b01, 0b00], 500, [0, 0, 0])


def test_xor(depth=1, bias=0, state=[0b01]):
    b = bias
    w1_w2_list = []
    ws = np.linspace(-np.pi / 2, np.pi / 2, 100)
    for w1 in ws:
        w2_list = []
        for w2 in ws:
            xors = XOR_simulator(2, depth)
            xors.set_state(state)
            loss = xors.weights2loss([b, w1, w2])
            w2_list.append(loss)

        w1_w2_list.append(w2_list)

    fig, ax = plt.subplots()
    ax.imshow(np.meshgrid(ws, ws), w1_w2_list, cmap="jet")
    plt.show()

    return w1_w2_list


def TEST_unit_gate(step=3):
    vects = []
    states = []
    for angle in np.linspace(-np.pi, np.pi, 100):
        qns = QuantumNeuralSystem(1, 1)
        state = qns.set_state(0b001)

        qns.set_weights([0, angle], step)
        qns.unit_gate.update_quantum_state(state)

        cos = np.sqrt((np.cos(2 * angle) ** 2 + 1) / 2)
        sin = -np.sqrt((np.sin(2 * angle) ** 2) / 2)
        tan0 = 1 / np.sqrt(1 + np.tan(angle) ** 4)
        tan1 = np.tan(angle) ** 2 / np.sqrt(1 + np.tan(angle) ** 4)

        vect = np.zeros(8)
        vect[1] = cos * tan0
        vect[5] = cos * tan1
        vect[3] = sin * np.sqrt(1/2)
        vect[7] = -sin * np.sqrt(1/2)

        print("state: {}".format(state.get_vector()))
        print("ans: {}".format(vect))
        print("diff: {}".format(state.get_vector() - vect))

        states.append(state.get_vector())
        vects.append(vect)

    return states, vects


class QuantumNeuralSystem:
    '''
    入力(size: n_i), 量子回路の深さ: depth=dとすると，
    |XXXXX>の下位から数えたインデックス
    0~n_i-1:input_qubit,
    n_i~n_i+d-1: ancilla_qubit,
    n_i+d: output_qubit
    例えばdepth=3とすると
    初期設定する量子状態はinputをXXXXXで表現して
    0b000XXXXXとなる(depthの分だけ0-padされる)．
    '''

    def __init__(self, input_bit_size, depth=1):
        '''
        Args:
            ancilla_bit_count(uint): number of ancilla bits
            depth(uint): size of quantum neuron circuit
            initial_weights(array-like): weights of NN
        '''
        self.input_bit_size = input_bit_size
        self.depth = depth

        self.qubit_count = self.input_bit_size + self.depth + 1

    def set_state(self, *input_bits):
        self.state = QuantumState(self.qubit_count)

        probs = np.ones(2 ** self.qubit_count)*0
        probs[input_bits] = 1.0/np.sqrt(len(input_bits))
        self.state.load(probs)

        print("Initial state:")
        print(self.state)

        return self.state

    def set_weights(self, weights, step=3):
        '''
        weightsを設定し，RUS基本回路を更新する．
        '''
        self.weights = weights
        self.set_unit_gate(step)

    def set_unit_gate(self, step=3):
        '''
        RUSの基本回路(測定は別で行う)を作成する．
        '''
        target = self.input_bit_size
        count = 0

        # R_y
        self.unit_gate = RY(target, -self.weights[0])
        for control_bit, weight in enumerate(self.weights[1:]):
            print(weight)
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
        '''
        c_ry = RY(target, angle)
        c_ry_mat = to_matrix_gate(c_ry)
        c_ry_mat.add_control_qubit(control, 1)

        return c_ry_mat

    def _get_ciYgate(self, control, target):
        '''
        helper: controlled iY gateの作成
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

        return inner_product(_state, state) / inner_product(_state, _state)

    def update_quantum_state(self, state, **kwargs):
        '''
        量子状態に対して量子ニューロンのゲートを作用させる．
        各ステップごとの成功確率を返す．
        '''
        self.prob_in_each_steps = []

        self._step_update_state(self.depth, state, **kwargs)

        return self.prob_in_each_steps

    def _step_update_state(self, step, state, show_prob=False,
                           show_before_projection=False):
        '''
        helper: 各ステップにおけるゲート作用
        '''
        _control = self.input_bit_size + step - 1
        _target = self.input_bit_size + step

        if step > 1:
            self._step_update_state(step - 1, state)
            ciY = self._get_ciYgate(_control, _target)
            ciY.update_quantum_state(state)
            self._step_update_state(step - 1, state)

        else:
            self.unit_gate.update_quantum_state(state)

        # if show_before_projection:
        #     print("Before projection:")
        #     print(state)

        # prob = self._measure_0projection(_target-1, state)
        # self.prob_in_each_steps.append(prob)

        # if show_prob:
        #     print("prob: {}".format(prob))

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

    def debug_each_gate(self, angle=np.pi / 2):
        inputbit = 0
        ancilla = 1
        target = 2

        state = QuantumState(3)
        cRY1 = self._get_cRY_gate(-angle, inputbit, ancilla)
        ciY = self._get_ciYgate(ancilla, target)
        cRY2 = self._get_cRY_gate(angle, inputbit, ancilla)

        print("test unit_gate :Start")

        for basis in [0b000, 0b001]:
            state.set_computational_basis(basis)
            print("input_bit: {:#05b}".format(basis))
            print(state)

            cRY1.update_quantum_state(state)
            print("1st controlled RY:")
            print(state)

            ciY.update_quantum_state(state)
            print("controlled iY:")
            print(state)

            cRY2.update_quantum_state(state)
            print("2nd controlled RY:")
            print(state)

            P0(ancilla).update_quantum_state(state)
            print("Projection:")
            print(state)

            print("\n")

        print("test unit_gate :End")


class XOR_simulator:
    def __init__(self, input_bit_size, depth=1):
        self.qns = QuantumNeuralSystem(input_bit_size, depth)
        self.input_bit_size = input_bit_size
        self.qubit_count = input_bit_size + depth + 1
        self.depth = depth
        self.measure_circuit = QuantumCircuit(self.qubit_count)
        for i in range(self.input_bit_size):
            self.measure_circuit.add_CNOT_gate(i, input_bit_size + depth)

    def learn(self, train_data_list, iteration, weights=None, loss_tol=1e-9):
        if weights is None:
            # biasの分を足す: self.num_inputbit + 1．
            self.weights_ini = np.random.rand(self.input_bit_size + 1)
        else:
            self.weights_ini = np.array(weights)

        self.name = "d{}_in{}_{}_{}".format(self.depth,
                                            self.input_bit_size,
                                            train_data_list,
                                            self.weights_ini)

        self.output_path = Path("Results")/self.name

        if not self.output_path.exists():
            os.makedirs(self.output_path)

        print("initial weights: {}".format(self.weights_ini))
        self.set_state(train_data_list)

        self.weights_list = []
        self.error_list = []

        for i in range(iteration):
            results = optimize.minimize(self.weights2loss, self.weights,
                                        method="Nelder-Mead",
                                        options={"adaptive": False,
                                                 "maxiter": 3})
            self.weights = results["x"]
            err = abs(results["fun"])
            print("{}, weight: {}, err: {}".format(i+1, results["x"], err))
            print(self._state)
            self.weights_list.append([i+1, *results["x"]])
            self.error_list.append([i+1, err])

            if err < loss_tol:
                break

        np.savetxt(self.output_path / "_".join([self.name, "err.txt"]),
                   self.error_list,
                   header="{}".format(self.weights_ini))
        np.savetxt(self.output_path / "_".join([self.name, "weig.txt"]),
                   self.weights_list,
                   header="{}".format(self.weights_ini))

    def set_state(self, input_bits):
        self.state = self.qns.set_state(input_bits)
        self.initial_state = self.state.copy()

    def weights2loss(self, weights):
        print(weights)
        self.qns.set_weights(weights)
        self.qns.update_quantum_state(self.state, show_prob=False)
        self._state = self.state.copy()

        self.measure_circuit.update_quantum_state(self.state)
        _temp = inner_product(self.initial_state, self.state)
        loss = -(_temp*np.conj(_temp)).real + 1

        self.state.load(self.initial_state)

        # print("simplex: \n")
        # print(_state)
        # print("weights: {}, loss: {}".format(weights, loss))

        return loss

    def plot_results(self):
        fig, axs = plt.subplots()
        # ax.plot(self.weights_list[:, 0], self.weights_list[:, 1], "bo-")
        axs.plot(self.error_list[:, 0], self.error_list[:, 1], "ro-")

        fig.savefig(self.name + "_err.png", self.error_list)


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
    if args.test:
        TEST()

    else:
        main_xor()
        # check_nelder_mead()
