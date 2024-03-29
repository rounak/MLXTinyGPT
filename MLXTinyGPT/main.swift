//
//  main.swift
//  MLXTinyGPT
//
//  Created by Rounak Jain on 3/16/24.
//

import Foundation
import MLX
import MLXRandom
import MLXNN
import MLXOptimizers

// Validation loss in Karpathy's video was 1.4873.
// With the params below, I was able to hit a loss of 1.5579436
enum HyperParameters {
    static let batchSize = 64
    static let blockSize = 128 // 256 in Karpathy's video, but going beyond 128 spits out nans on my Mac
    static let maxIters = 5000
    static let evalInterval = 500
    static let learningRate: Float = 3e-4
    static let evalIters = 200
    static let nEmbedPerHead = 32 // 64 in Karpathy's video, but going beyond 32 spits out nans on my Mac
    static var nEmbed: Int { nEmbedPerHead * nHead }
    static let nHead = 6
    static let nLayer = 6
    static let dropout: Float = 0.2
}

let text = try String(
    contentsOf: URL(string: "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")!
)
let sortedChars = Set(text).sorted()
let vocabSize = sortedChars.count
var characterToInt: [Character: Int] = [:]
var intToCharacter: [Int: Character] = [:]
for (i, character) in sortedChars.enumerated() {
    characterToInt[character] = i
    intToCharacter[i] = character
}
func encode(_ s: String) -> [Int] { s.map { characterToInt[$0]! } }
func decode(_ i: MLXArray) -> String {
    let castedArray: [any BinaryInteger] = switch i.dtype {
    case .int32:
        i.asArray(Int32.self)
    case .int64:
        i.asArray(Int64.self)
    default:
        fatalError("\(i.dtype) not supported")
    }
    return String(castedArray.map { intToCharacter[Int($0)]!})
}

let data = MLXArray(encode(text))

let n = Int(0.9*Double(data.count))
let trainData = data[..<n] // 90% train
let valData = data[n...] // 10% test

MLXRandom.seed(1337) // determinism

enum Split: CaseIterable {
    case train, validation
}

func getBatch(_ split: Split, of batchSize: Int) -> (MLXArray, MLXArray) {
    let data = switch split {
    case .train:
        trainData
    case .validation:
        valData
    }
    let ix = MLXRandom.randInt((0..<(data.count - HyperParameters.blockSize)), [batchSize]).asArray(Int.self)
    let x = MLX.stacked(ix.map { i in
        data[i..<(i+HyperParameters.blockSize)]
    })

    let y = MLX.stacked(ix.map { i in
        data[i+1..<(i+HyperParameters.blockSize+1)]
    })
    return (x, y)
}

let mask = MLXNN.MultiHeadAttention.createAdditiveCausalMask(HyperParameters.blockSize)

class Head: Module, UnaryLayer {
    let query: Linear
    let key: Linear
    let value: Linear
    let dropout: Dropout

    init(headSize: Int) {
        let nEmbed = HyperParameters.nEmbed
        self.query = Linear(nEmbed, headSize, bias: false)
        self.key = Linear(nEmbed, headSize, bias: false)
        self.value = Linear(nEmbed, headSize, bias: false)
        dropout = Dropout(p: HyperParameters.dropout)
        super.init()
    }

    func callAsFunction(_ x: MLX.MLXArray) -> MLX.MLXArray {
        let (T, C) = (x.dim(1), x.dim(2))
        let k = key(x) // Dimensions: Batch, Time, headSize
        let q = query(x) // Dimensions: B, T, headSize
        let v = value(x)

        var weights = q.matmul(k.transposed(0, 2, 1)) * pow(MLXArray(C), -0.5)
        // mask is filled with -inf in one triangle and 0 in other triangle. x + (-inf) = -inf. x + 0 = x
        weights = weights + mask[..<T, ..<T].asType(weights.dtype)
        weights = softMax(weights, axis: -1)
        weights = dropout(weights)
        return weights.matmul(v)
    }
}

class MultiHeadAttention: Module, UnaryLayer {
    let heads: [Head]
    let projection: Linear
    let dropout: Dropout

    init(numberOfHeads: Int, headSize: Int) {
        heads = (0..<numberOfHeads).map { _ in
            Head(headSize: headSize)
        }
        projection = Linear(HyperParameters.nEmbed, HyperParameters.nEmbed)
        dropout = Dropout(p: HyperParameters.dropout)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLX.MLXArray {
        var out = concatenated(self.heads.map { $0(x) }, axis: -1)
        out = dropout(projection(out))
        return out
    }
}

class Block: Module, UnaryLayer {
    let attention: MultiHeadAttention
    let feedForward: Sequential
    let layerNorm1: LayerNorm
    let layerNorm2: LayerNorm

    init(nEmbed: Int, numberOfHeads: Int) {
        let headSize = nEmbed / numberOfHeads
        attention = MultiHeadAttention(numberOfHeads: numberOfHeads, headSize: headSize)
        feedForward = Sequential(layers: [
            Linear(nEmbed, 4*nEmbed),
            ReLU(),
            Linear(4*nEmbed, nEmbed),
            Dropout(p: HyperParameters.dropout)
        ])
        layerNorm1 = LayerNorm(dimensions: nEmbed)
        layerNorm2 = LayerNorm(dimensions: nEmbed)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x + attention(layerNorm1(x))
        x = x + feedForward(layerNorm2(x))
        return x
    }
}

public class LanguageModel: Module, UnaryLayer {
    let tokenEmbeddingTable: Embedding
    let positionEmbeddingTable: Embedding
    let blocks: Sequential
    let layerNorm: LayerNorm
    let languageModelingHead: Linear

    public init(vocabSize: Int, numberOfBlocks: Int, numberOfHeads: Int) {
        let nEmbed = HyperParameters.nEmbed
        tokenEmbeddingTable = Embedding(embeddingCount: vocabSize, dimensions: nEmbed)
        positionEmbeddingTable = Embedding(embeddingCount: vocabSize, dimensions: nEmbed)
        blocks = Sequential(layers: (0..<numberOfBlocks).map { _ in
            Block(nEmbed: nEmbed, numberOfHeads: numberOfHeads)
        })
        layerNorm = LayerNorm(dimensions: nEmbed)
        languageModelingHead = Linear(nEmbed, vocabSize)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let tokenEmbeddings = tokenEmbeddingTable(x)
        let positionEmbedding = positionEmbeddingTable(MLXArray(0..<x.shape[1]))
        var x = tokenEmbeddings + positionEmbedding
        x = blocks(x)
        x = layerNorm(x)
        let logits = languageModelingHead(x)
        return logits
    }

    func generate(idx: MLXArray, maxNewTokens: Int) -> MLXArray {
        var newIdx = idx
        for _ in 0..<maxNewTokens {
            // get most recent blockSize logits
            var logits = self(newIdx[from: -HyperParameters.blockSize, stride: 1, axis: -1])
            logits = logits[-1, axis: 1]
            let idxNext = categorical(logits, count: 1)
            newIdx = concatenated([newIdx, idxNext], axis: 1)
            eval(newIdx) // Eval the array as it grows to avoid a large eval at the end
        }
        return newIdx
    }
}

let model = LanguageModel(
    vocabSize: vocabSize,
    numberOfBlocks: HyperParameters.nLayer,
    numberOfHeads: HyperParameters.nHead
)

func train() {
    let optimizer = AdamW(learningRate: HyperParameters.learningRate)

    func loss(model: LanguageModel, x: MLXArray, y: MLXArray) -> MLXArray {
        crossEntropy(logits: model(x), targets: y, reduction: .mean)
    }

    let trainStart = Date.timeIntervalSinceReferenceDate

    let lg = valueAndGrad(model: model, loss)
    func estimateLoss() -> [Split: Float32] {
        model.train(false)
        var out: [Split: Float32] = [:]
        for split in Split.allCases {
            let losses = MLX.zeros([HyperParameters.evalIters])
            for k in 0..<HyperParameters.evalIters {
                let (xb, yb) = getBatch(split, of: HyperParameters.batchSize)
                let logits = model(xb)
                losses[k] = crossEntropy(logits: logits, targets: yb, reduction: .mean)
            }
            out[split] = losses.mean().item()
        }
        model.train()
        return out
    }

    for epoch in 1...HyperParameters.maxIters {
        if epoch % HyperParameters.evalInterval == 0 {
            let losses = estimateLoss()
            print("Step: \(epoch): Train loss: \(losses[.train]!) Val loss: \(losses[.validation]!)")
        }
        let (xb, yb) = getBatch(.train, of: HyperParameters.batchSize)

        let (_, grads) = lg(model, xb, yb) // TODO: Figure out if this can be MLX.compile'd
        optimizer.update(model: model, gradients: grads)

        eval(model, optimizer)
    }
    print("Training complete in \((Date.timeIntervalSinceReferenceDate - trainStart).formatted())")
}

train()

let generationStart = Date.timeIntervalSinceReferenceDate
print(decode(model.generate(idx: MLX.zeros([1, 1], type: Int32.self), maxNewTokens: 300)))
print("Generation complete in \((Date.timeIntervalSinceReferenceDate - generationStart).formatted())")
