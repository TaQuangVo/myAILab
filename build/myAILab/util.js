"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.arraySub = exports.scalarProduct = exports.sigmoid = exports.toOneColArray = exports.createRandArray = void 0;
const ml_matrix_1 = __importDefault(require("ml-matrix"));
const createRandArray = (row, cols) => {
    const arr = [];
    for (let y = 0; y < row; y++) {
        const temp = [];
        for (let x = 0; x < cols; x++)
            temp[x] = Math.random();
        arr[y] = temp;
    }
    return arr;
};
exports.createRandArray = createRandArray;
const arraySub = (arr1, arr2) => {
    const r = [];
    for (let i = 0; i < arr1.length; i++) {
        r[i] = arr1[i] - arr2[i];
    }
    return r;
};
exports.arraySub = arraySub;
const scalarProduct = (matA, matB) => {
    const A = matA.to2DArray();
    const B = matB.to2DArray();
    const result = [];
    for (let y = 0; y < matA.rows; y++) {
        const row = [];
        for (let x = 0; x < matB.columns; x++) {
            let tot = 0;
            for (let t = 0; t < matA.columns; t++)
                tot += A[y][t] * B[t][x];
            row[x] = tot;
        }
        result[y] = row;
    }
    return new ml_matrix_1.default(result);
};
exports.scalarProduct = scalarProduct;
const toOneColArray = (a) => {
    const arr = [];
    for (let y = 0; y < a.length; y++) {
        arr[y] = [a[y]];
    }
    return arr;
};
exports.toOneColArray = toOneColArray;
const sigmoid = (num) => {
    return 1 / (1 + Math.exp(-num));
};
exports.sigmoid = sigmoid;
