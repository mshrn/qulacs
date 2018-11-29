import sys
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
from qulacs import QuantumCircuit, QuantumGateMatrix, QuantumState
from qulacs.gate import CNOT, CZ, P0, RY, DenseMatrix, merge, to_matrix_gate
from qulacs.state import inner_product
from scipy import optimize


def xor_test(depth=1):
    xors = XOR_simulator(2, depth=depth)
    xors.learn([0b01], 2000, [0.01, 1, -np.pi/2])


def TEST():
    qns = QuantumNeuralSystem(1)
    qns.test_unit_gate()
    qns.check_output()


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

    def __init__(self, input_bit_count, depth=1):
        '''
        Args:
            ancilla_bit_count(uint): number of ancilla bits
            depth(uint): size of quantum neuron circuit
            initial_weights(array-like): weights of NN
        '''
        self.input_bit_count = input_bit_count
        self.depth = depth

        self.qubit_count = self.input_bit_count + self.depth + 1

    def set_state(self, input_bits, show_initial_state=True):
        self.state = QuantumState(self.qubit_count)

        probs = np.ones(2 ** self.qubit_count)*0
        probs[input_bits] = 1.0/len(input_bits)
        self.state.load(probs)

        print("Initial state:")
        print(self.state)

        return self.state

    def set_weights(self, weights):
        '''
        weightsを設定し，RUS基本回路を更新する．
        '''
        self.weights = weights
        self.set_unit_gate()

    def test_unit_gate(self, angle=np.pi / 2):
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

    def check_output(self):
        inputbit = 0
        ancilla = 1
        target = 2

        state = QuantumState(3)
        ciY = self._get_ciYgate(ancilla, target)

        state.set_computational_basis(0b001)
        _state = state.copy()

        for angle in np.linspace(0, 2 * np.pi, 50):

            cRY1 = self._get_cRY_gate(-angle, inputbit, ancilla)
            cRY2 = self._get_cRY_gate(angle, inputbit, ancilla)

            cRY1.update_quantum_state(state)
            ciY.update_quantum_state(state)
            cRY2.update_quantum_state(state)
            P0(ancilla).update_quantum_state(state)

            prob = inner_product(state, state)
            ans = (np.cos(2*angle)**2+1)/2
            print("angle: {}, prob: {}, err: {}".format(angle, prob, prob-ans))
            print(state)

            state.load(_state)

    def debug_gate(self, input_bit, angle=np.pi/2):
        weights = np.ones(self.input_bit_count + 1) * angle
        self.set_weights(weights)
        self.set_state(input_bit)

        self.update_quantum_state(self.state, show_before_projection=True)

        print("Updated(debug) state:")
        print(self.state)

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
        iY = DenseMatrix(target, [[0, 1], [-1, 0]])
        iY.add_control_qubit(control, 1)

        return iY

    def _measure_0projection(self, target, state):
        '''
        helper: P0の射影を取り，確率を返す．
        '''
        _state = state.copy()
        P0(target).update_quantum_state(state)

        return inner_product(_state, state) / inner_product(_state, _state)

    def set_unit_gate(self):
        '''
        RUSの基本回路(測定は別で行う)を作成する．
        '''
        if self.weights is None:
            self.weights = np.ones(self.input_bit_count + 1) * np.pi / 2

        target = self.input_bit_count

        # R_y
        self.unit_gate = RY(target, self.weights[0])
        for control_bit, weight in enumerate(self.weights[1:]):
            c_ry_mat = self._get_cRY_gate(weight, control_bit, target)
            self.unit_gate = merge(self.unit_gate, c_ry_mat)

        # controlled-iY
        y_target = target + 1
        y_control = target
        ciY = self._get_ciYgate(y_control, y_target)
        self.unit_gate = merge(self.unit_gate, ciY)

        # R_y^\dagger
        self.unit_gate = merge(self.unit_gate, RY(target, -self.weights[0]))
        for control_bit, weight in enumerate(-self.weights[1:]):
            c_ry_mat = self._get_cRY_gate(weight, control_bit, target)
            self.unit_gate = merge(self.unit_gate, c_ry_mat)

        return self.unit_gate

    def update_quantum_state(self, state, **kwargs):
        '''
        量子状態に対して量子ニューロンのゲートを作用させる．
        各ステップごとの成功確率を返す．
        '''
        self.prob_in_each_steps = []

        self._step_update_state(self.depth, state, **kwargs)

        return self.prob_in_each_steps

    def _step_update_state(self, step, state, show_prob=True,
                           show_before_projection=False):
        '''
        helper: 各ステップにおけるゲート作用
        '''
        _control = self.input_bit_count + step - 1
        _target = self.input_bit_count + step

        if step > 1:
            self._step_update_state(step - 1, state)
            ciY = self._get_ciYgate(_control, _target)
            ciY.update_quantum_state(state)
            self._step_update_state(step - 1, state)

        else:
            self.unit_gate.update_quantum_state(state)

        if show_before_projection:
            print("Before projection:")
            print(state)

        prob = self._measure_0projection(_target-1, state)
        self.prob_in_each_steps.append(prob)

        if show_prob:
            print("prob: {}".format(prob))

    def get_neuron_circuit(self):
        self.nc = QuantumCircuit(self.qubit_count)
        self.set_unit_gate()

        self._add_step_neuron_circuit(self.depth)

        return self.nc

    def _add_step_neuron_circuit(self, step):
        if step > 1:
            self._add_step_neuron_circuit(step - 1)

            _control = self.input_bit_count + step - 1
            _target = self.input_bit_count + step
            ciY = self._get_ciYgate(_control, _target)
            self.nc.add_gate(ciY)

            self._add_step_neuron_circuit(step - 1)

        else:
            self.nc.add_gate(self.unit_gate)


class XOR_simulator:
    def __init__(self, input_bit_count, depth=1):
        self.qns = QuantumNeuralSystem(input_bit_count, depth)
        self.input_bit_count = input_bit_count
        self.qubit_count = input_bit_count + depth + 1
        self.measure_circuit = QuantumCircuit(self.qubit_count)
        for i in range(self.input_bit_count):
            self.measure_circuit.add_CNOT_gate(i, input_bit_count + depth)

    def learn(self, train_data_list, iteration, weights=None, loss_tol=1e-9):

        self.state = self.qns.set_state(train_data_list)
        self.initial_state = self.state.copy()

        if weights is None:
            # biasの分を足す: self.num_inputbit + 1．
            weights = np.random.rand(self.input_bit_count + 1)
        else:
            weights = np.array(weights)

        self.weights_list = []
        self.error_list = []

        for i in range(iteration):
            results = optimize.minimize(self.weights2loss, weights,
                                        method="Nelder-Mead",
                                        options={"adaptive": False,
                                                 "maxiter": 3})
            err = abs(results["fun"]+1)
            print("{}, weight: {}, err: {}".format(i, results["x"], err))
            print(self._state)
            self.weights_list.append([i, *results["x"]])
            self.error_list.append([i, err])

            if err < loss_tol:
                break

    def weights2loss(self, weights):
        self.qns.set_weights(weights)
        self.qns.update_quantum_state(self.state, show_prob=False)
        self._state = self.state.copy()

        self.measure_circuit.update_quantum_state(self.state)
        _temp = inner_product(self.initial_state, self.state)
        loss = -(_temp*np.conj(_temp)).real

        self.state.load(self.initial_state)

        # print("simplex: \n")
        # print(_state)
        # print("weights: {}, loss: {}".format(weights, loss))

        return loss

    def plot_results(self):
        fig, axs = plt.subplots(2)
        # ax.plot(self.weights_list[:, 0], self.weights_list[:, 1], "bo-")
        axs[0].plot(self.error_list[:, 0], self.error_list[:, 1], "ro-")
        for weight in self.weights_list[:, 1]:
            axs[1].plot(self.weights_list[:, 0])
        fig.show()


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
        xor_test()
        # check_nelder_mead()
