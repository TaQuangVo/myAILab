"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const ml_matrix_1 = require("ml-matrix");
const util_1 = require("./util");
class NNLayer {
    constructor(nrOfInput, nrOfOutput) {
        this._w = new ml_matrix_1.Matrix((0, util_1.createRandArray)(nrOfOutput, nrOfInput));
        this._b = new ml_matrix_1.Matrix((0, util_1.createRandArray)(nrOfOutput, 1));
        this.lastInput = new ml_matrix_1.Matrix(nrOfInput, 1);
        this.lastOutput = new ml_matrix_1.Matrix(nrOfOutput, 1);
    }
    feedForward(input) {
        this.lastInput = new ml_matrix_1.Matrix((0, util_1.toOneColArray)(input));
        const sum = (0, util_1.scalarProduct)(this._w, this.lastInput);
        sum.add(this._b);
        sum.apply((r, c) => {
            const sigm = (0, util_1.sigmoid)(sum.get(r, c));
            sum.set(r, c, sigm);
        });
        this.lastOutput = sum;
        return sum.to1DArray();
    }
    backwardPropagation(err, lr) {
        const gradient = new ml_matrix_1.Matrix((0, util_1.toOneColArray)(err));
        /*console.log("____________________________")
        console.log({"lr":lr})
        console.log({"Error":gradient})
        console.log({"Input":this.lastInput})
        console.log({"Output":this.lastOutput})*/
        gradient.apply((r, c) => {
            const E = gradient.get(r, c);
            const O = this.lastOutput.get(r, c);
            const g = lr * E * (O * (1 - O));
            gradient.set(r, c, g);
        });
        const inputT = this.lastInput.transpose();
        const dW = (0, util_1.scalarProduct)(gradient, inputT);
        this._w.add(dW);
        this._b.add(gradient);
        return this.getInputError(err);
    }
    getInputError(err) {
        const outputErr = new ml_matrix_1.Matrix((0, util_1.toOneColArray)(err));
        const wT = this._w.transpose();
        const inputErr = (0, util_1.scalarProduct)(wT, outputErr);
        return inputErr.to1DArray();
    }
}
exports.default = NNLayer;
