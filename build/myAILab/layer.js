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
    feefForward(input) {
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
    view() {
        console.log("weight: ", this._w);
        console.log("Bios: ", this._b);
    }
}
exports.default = NNLayer;
