"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const NNlayer_1 = __importDefault(require("./NNlayer"));
const util_1 = require("./util");
class default_1 {
    constructor(layerDef) {
        this._layers = [];
        for (let i = 0; i < layerDef.length - 1; i++)
            this._layers[i] = new NNlayer_1.default(layerDef[i], layerDef[i + 1]);
    }
    guess(input) {
        for (let i = 0; i < this._layers.length; i++)
            input = this._layers[i].feedForward(input);
        return input;
    }
    train(input, target, lr) {
        const guess = this.guess(input);
        let error = (0, util_1.arraySub)(target, guess);
        //console.log({guess})
        for (let i = this._layers.length - 1; i >= 0; i--)
            error = this._layers[i].backwardPropagation(error, lr);
    }
}
exports.default = default_1;
