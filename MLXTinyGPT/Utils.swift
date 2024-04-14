//
//  Utils.swift
//  MLXTinyGPT
//
//  Created by Rounak Jain on 4/13/24.
//

import Foundation
import MLX

func ask(prompt: String) -> Bool {
    print(prompt + " Y/n")
    // Wait for user input
    if let choice = readLine()?.lowercased() {
        switch choice {
        case "y":
            return true
        default:
            return false
        }
    }
    return false // Return nil if no input was provided
}

func loadWeights() throws -> [String: MLXArray] {
    guard !FileManager.default.fileExists(atPath: path.absoluteString) else {
        return try MLX.loadArrays(url: path)
    }
    let data = try Data(contentsOf: URL(string: "https://github.com/rounak/MLXTinyGPT/raw/main/tinygptweights.safetensors")!)
    try data.write(to: path)
    return try MLX.loadArrays(url: path)
}
