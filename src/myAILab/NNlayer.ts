import ml_matrix, { Matrix } from "ml-matrix"
import { createRandArray, scalarProduct, sigmoid, toOneColArray } from "./util"

export default class NNLayer{
    private _w:Matrix
    private _b:Matrix
    private lastInput:Matrix
    private lastOutput:Matrix

    constructor(nrOfInput:number, nrOfOutput:number){
        this._w = new Matrix(createRandArray(nrOfOutput, nrOfInput))
        this._b = new Matrix(createRandArray(nrOfOutput, 1))

        this.lastInput = new Matrix(nrOfInput, 1)
        this.lastOutput = new Matrix(nrOfOutput, 1)
    }

    feedForward(input:number[]):number[]{
        this.lastInput = new Matrix(toOneColArray(input))
        
        const sum = scalarProduct(this._w, this.lastInput)
        sum.add(this._b)

        sum.apply((r, c)=>{
            const sigm = sigmoid(sum.get(r,c))
            sum.set(r, c, sigm)
        })

        this.lastOutput = sum

        return sum.to1DArray();
    }

    backwardPropagation(err:number[], lr:number){
        const gradient = new Matrix(toOneColArray(err))

        /*console.log("____________________________")
        console.log({"lr":lr})
        console.log({"Error":gradient})
        console.log({"Input":this.lastInput})
        console.log({"Output":this.lastOutput})*/

        gradient.apply((r,c) => {
            const E = gradient.get(r,c)
            const O = this.lastOutput.get(r,c)
            const g = lr * E * (O*(1-O))
            gradient.set(r,c,g)
        })

        const inputT = this.lastInput.transpose()
        const dW = scalarProduct(gradient, inputT)

        this._w.add(dW)
        this._b.add(gradient)
        
        return this.getInputError(err)
    }

    private getInputError(err:number[]){
        const outputErr = new Matrix(toOneColArray(err))
        const wT = this._w.transpose()
        const inputErr = scalarProduct(wT, outputErr)
        return inputErr.to1DArray()
    }
}