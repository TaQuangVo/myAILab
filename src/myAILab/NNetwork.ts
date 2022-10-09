import NNLayer from "./NNlayer"
import { arraySub } from "./util"

export default class {
    _layers:NNLayer[]

    constructor(layerDef:number[]){
        this._layers = []
        for(let i = 0; i < layerDef.length-1; i++)
            this._layers[i] = new NNLayer(layerDef[i], layerDef[i+1])
    }

    guess(input:number[]){
        for(let i = 0; i < this._layers.length; i ++)
            input = this._layers[i].feedForward(input)

        return input;
    }

    train(input:number[], target:number[], lr:number){
        const guess = this.guess(input)

        let error = arraySub(target, guess)

        //console.log({guess})

        for(let i = this._layers.length-1; i >= 0; i--)
            error = this._layers[i].backwardPropagation(error, lr)
    }

}