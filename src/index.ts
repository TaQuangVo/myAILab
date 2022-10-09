import NNLayer from "./myAILab/NNlayer"
import NNetwork from "./myAILab/NNetwork"




const nn = new NNetwork([2,2,1])

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