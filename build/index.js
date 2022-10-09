"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const NNetwork_1 = __importDefault(require("./myAILab/NNetwork"));
const nn = new NNetwork_1.default([2, 2, 1]);
const guessb = nn.guess([0.2, 0.9]);
console.log(guessb);
for (let i = 0; i < 10000; i++) {
    const x = Math.random();
    const y = Math.random();
    const t = x > y ? 1 : 0;
    nn.train([x, y], [t], 0.07);
}
const guessf = nn.guess([0.1, 0.9]);
console.log(guessf);
