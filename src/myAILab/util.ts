import Matrix from "ml-matrix";

const createRandArray = (row:number, cols:number) : number[][] => {
    const arr = []
    for(let y = 0; y < row; y++){
        const temp = []
        for(let x = 0; x < cols; x++)
            temp[x] = Math.random()
        arr[y] = temp;
    }
    return arr
}

const arraySub = (arr1:number[], arr2:number[]) : number[] => {
    const r = []
    for (let i = 0; i < arr1.length; i++) {
        r[i] = arr1[i] - arr2[i]
    }
    return r;
}

const scalarProduct = (matA:Matrix, matB:Matrix):Matrix => {
    const A = matA.to2DArray();
    const B = matB.to2DArray();

    const result = []
    for(let y = 0; y < matA.rows; y++){
        const row = []
        for(let x = 0; x < matB.columns; x++){
            let tot = 0;
            for(let t = 0; t < matA.columns; t++)
                tot += A[y][t] * B[t][x]
            row[x] = tot
        }
        result[y] = row
    }
        
    return new Matrix(result)
}

const toOneColArray = (a:number[]):number[][] => {
    const arr = []
    for(let y = 0; y < a.length; y++){
        arr[y] = [a[y]];
    }
    return arr
}

const sigmoid = (num:number):number => {
    return 1/(1+Math.exp(-num))
}


export {createRandArray, toOneColArray, sigmoid, scalarProduct, arraySub}