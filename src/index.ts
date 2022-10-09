import NNLayer from "./myAILab/NNlayer"
import NNetwork from "./myAILab/NNetwork"




const nn = new NNetwork([2,2,1])


/*
const guessb = nn.guess([0.2,0.9])
console.log(guessb)

for(let i = 0; i < 10000; i++){
    const x = Math.random()
    const y = Math.random()
    const t = x>y?1:0

    nn.train([x,y],[t],0.07)
}

const guessf = nn.guess([0.1,0.9])
console.log(guessf)

*/

const dataSet = [
    {
        input:[0,0],
        target:[0]
    },{
        input:[1,0],
        target:[1]
    },{
        input:[0,1],
        target:[1]
    },{
        input:[1,1],
        target:[0]
    }
]

const guessb = nn.guess([0,0])
console.log(guessb)

for(let i = 0; i < 50000; i++){
    const x = Math.floor(Math.random()*4)

    const data = dataSet[x]

    nn.train(data.input,data.target,0.2)
}

console.log(nn.guess([0,0]))
console.log(nn.guess([1,0]))
console.log(nn.guess([0,1]))
console.log(nn.guess([1,1]))